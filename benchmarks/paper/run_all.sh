#!/usr/bin/env bash
# Runs the measurements behind results/runs/ and builds results/claims/.
#
#   bash benchmarks/paper/run_all.sh <experiment> [<experiment> ...]
#
# Experiments: e1 dispatch_cost kernel c1 table2 lam7 lattice ablation13
#              ablation7 c2 c3 table3 table5 table5_tp2 summarize, or "all".
#
# Each experiment writes to the same results/runs/ directory its measured run
# lives in. Drivers skip tags that are already recorded, so to measure again
# either point OUT_ROOT somewhere else (OUT_ROOT=/tmp/out) or delete the
# rows first. Single-GPU experiments use GPU 0 unless GPU is set; c2 and
# table5_tp2 need both A6000s.
#
# Prerequisites (README.md): models/, adapters*/, data/sharegpt in the repo
# root; vLLM 0.5.0 in venvs/vllm050 (or VLLM050_PYTHON); S-LoRA, Punica and
# dLoRA as in benchmarks/sota/REPRODUCE_SOTA.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
R=${OUT_ROOT:-results/runs}
P=benchmarks/paper
export BENCH_TIMEOUT=${BENCH_TIMEOUT:-3600}
# Scripts run against this tree, whatever else is installed.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

M7="MODEL=./models/llama-7b ADIR=./adapters RANK=32 K=10"
M13="MODEL=./models/llama-13b ADIR=./adapters_13b_r64 RANK=64 K=16"
AS_7B="SEGMODE=3 AS_ADMIT_GROUPED=1"
AS_13B="SEGMODE=3 AS_KV_ADMISSION=1 AS_ADMIT_GROUPED=1"

e1() {
    # Mixed-batch loss of SGMV (E1) and the Nsight Compute profile behind it.
    $PY benchmarks/micro/m1_sgmv_throughput_scale.py --model ./models/llama-7b \
        --N-values 512 1024 2048 4096 --K-values 2 4 8 16 --conditions S A D --n-runs 100 --warmup 10 \
        --output results/e1/scale/scale_sweep_a6000_tp1.csv
    for c in A B S C D; do
        bash scripts/reproduce/ncu_e1.sh $c ./models/llama-7b a6000
    done
}

dispatch_cost() {
    # CASH dispatch cost per scheduler tick against K, on the CPU.
    $PY scripts/experiments/cash_dispatch_cost.py --hardware-label a6000_single \
        --output results/cash_dispatch/a6000_single/overhead_scaling.csv
}

kernel() {
    RANK=32 SHAPE=13b OUT=$R/kernel $PY $P/kernel_layer.py
    RANK=32 SHAPE=13b_tp2 OUT=$R/kernel $PY $P/kernel_layer.py
    RANK=64 SHAPE=13b OUT=$R/kernel $PY $P/kernel_layer.py
    RANK=32 OUT=$R/kernel $PY $P/kernel_tiles.py
    OUT=$R/kernel $PY $P/output_check.py
    OUT=$R/kernel $PY $P/l3_check.py
    $PY scripts/experiments/fused_kernel_crossover_benchmark.py --output-dir $R/kernel/fused_crossover \
        --hardware a6000_tp1 --ranks 16 32 --n-tokens 4 8 16 32 64 128 256 --n-reps 500 --warmup 50
}

c1() {
    # C1: AdapterSlots (C7) and vLLM 0.6.3 (C0), K=10, rank 32, saturated.
    for S in 42 43 44; do
        env $M7 SEED=$S CELLS=128:256:C0 OUT=$R/c1_k10/cells $PY $P/cell.py
        env $M7 $AS_7B SEED=$S CELLS=128:256:C7 OUT=$R/c1_k10/cells $PY $P/cell.py
    done
}

table2() {
    # Table II baselines on the C1 traces: vLLM 0.5.0, S-LoRA, Punica, dLoRA.
    RATE=300 NP=400 WU=30 ARMS=v050,slora SEEDS=42,43,44 OUT=$R/c1_k10/baselines $PY $P/k10.py
    for S in 42 43 44; do
        SEED=$S OUT=$R/c1_k10/punica $PY $P/punica_engine.py
        SEED=$S PORT=$((8300 + S)) OUT=$R/c1_k10/dlora $PY $P/dlora_engine.py
    done
}

lam7() {
    # C1 at the paper's arrival rate, with outputs long enough to load the GPU.
    env RATE=7 NP=1000 WU=30 MT=384 $AS_7B ARMS=as,v063 SEEDS=42,43,44 OUT=$R/c1_lam7_mt384 $PY $P/k10.py
    RATE=7 NP=1000 WU=30 MT=384 ARMS=v050 SEEDS=42,43 OUT=$R/c1_lam7_mt384 $PY $P/k10.py
}

lattice() {
    # Table I lattice, modes C1..C7, with per-batch WAR / GWAR(8).
    env RATE=300 NP=400 WU=30 $AS_7B AS_LOG_WAR=1 ARMS=asC1,asC2,asC3,asC4,asC5,asC6,asC7 \
        SEEDS=42,43,44 OUT=$R/table1_lattice $PY $P/k10.py
}

ablation13() {
    # Table I as a cumulative ablation on LLaMA-13B: each row adds one mechanism.
    local D=$R/table1_ablation_13b
    env $M13 CELLS=64:256:C0 OUT=$D/L0_vllm $PY $P/cell.py
    env $M13 AS_LOG_WAR=1 CELLS=64:256:C6 OUT=$D/L1_wgkp $PY $P/cell.py
    env $M13 AS_LOG_WAR=1 AS_KV_ADMISSION=1 CELLS=64:256:C6 OUT=$D/L2_kvadm $PY $P/cell.py
    env $M13 AS_LOG_WAR=1 AS_KV_ADMISSION=1 SEGMODE=2 CELLS=64:256:C7 OUT=$D/L3_grouped $PY $P/cell.py
    env $M13 AS_LOG_WAR=1 AS_KV_ADMISSION=1 SEGMODE=3 AS_SEGGEMM_PREFILL=0 CELLS=64:256:C7 \
        OUT=$D/L4_multislice $PY $P/cell.py
    env $M13 AS_LOG_WAR=1 AS_KV_ADMISSION=1 SEGMODE=3 CELLS=64:256:C7 OUT=$D/L5_prefill $PY $P/cell.py
    env $M13 AS_LOG_WAR=1 $AS_13B CELLS=64:256:C7 OUT=$D/L6_full $PY $P/cell.py
}

ablation7() {
    # The same ablation on the C1 configuration (row 0 is the C1 vLLM 0.6.3 cell).
    local D=$R/table1_ablation_7b
    env $M7 AS_LOG_WAR=1 CELLS=128:256:C6 OUT=$D/L1_wgkp $PY $P/cell.py
    env $M7 AS_LOG_WAR=1 SEGMODE=2 CELLS=128:256:C7 OUT=$D/L3_grouped $PY $P/cell.py
    env $M7 AS_LOG_WAR=1 SEGMODE=3 AS_SEGGEMM_PREFILL=0 CELLS=128:256:C7 OUT=$D/L4_multislice $PY $P/cell.py
    env $M7 AS_LOG_WAR=1 SEGMODE=3 CELLS=128:256:C7 OUT=$D/L5_prefill $PY $P/cell.py
    env $M7 AS_LOG_WAR=1 $AS_7B CELLS=128:256:C7 OUT=$D/L6_full $PY $P/cell.py
}

c2() {
    # C2: APIS against vLLM TP=2 and topology-matched vLLM, K=50, rank 128, two GPUs.
    env RANK=128 ARMS=apis,tp2,topo_rr25 SEEDS=42,43,44 $AS_7B OUT=$R/c2_apis $PY $P/apis.py
    RANK=128 ARMS=topo SEEDS=42 OUT=$R/c2_apis $PY $P/apis.py
}

c3() {
    # C3: alignment-deadline sweep, two runs per point on two traces.
    for S in 42 43; do
        for RUN in 0 1; do
            SEED=$S C3_NP=2000 OUT=$R/c3_deadline/s${S}_run$RUN $PY $P/c3_deadline.py
        done
    done
}

table3() {
    # Table III: K=4, rank 16, lambda 7, engine-side and client-side TTFT.
    VARIANTS=sync ARMS=as SEEDS=42,43,44 OUT=$R/table3_k4 $PY $P/k4.py
    VARIANTS=engine ARMS=vllm SEEDS=42 OUT=$R/table3_k4 $PY $P/k4.py
}

table5() {
    # Table V: LLaMA-13B on one A6000, rank 64, 64-token outputs, 2000 requests.
    # Both systems get the same LoRA slot budget: K slots, then 16 slots.
    local D=$R/table5_13b B="MODEL=./models/llama-13b ADIR=./adapters_13b_r64 RANK=64 NP=2000"
    for S in 42 43 44; do
        for K in 4 8 16; do
            for SL in $K 16; do
                X="--max-loras $SL --max-cpu-loras 16"
                env $B K=$K SEED=$S EXTRA_ARGS="$X" CELLS=64:256:C0 OUT=$D/k${K}_v_ml$SL $PY $P/cell.py
                env $B K=$K SEED=$S $AS_13B AS_MAX_LORAS=$SL EXTRA_ARGS="$X" CELLS=64:256:C7 \
                    OUT=$D/k${K}_a_ml$SL $PY $P/cell.py
            done
        done
    done
}

table5_tp2() {
    # Table V setting with TP=2 on both A6000s, rank 32.
    SEGMODE=3 AS_ADMIT_GROUPED=1 RANKS=32 KS=4,8,16 OUT=$R/table5_13b_tp2 $PY $P/tp2_13b.py
    for S in 43 44; do
        SEGMODE=3 AS_ADMIT_GROUPED=1 RANKS=32 KS=4 SEED=$S OUT=$R/table5_13b_tp2 $PY $P/tp2_13b.py
    done
}

summarize() {
    $PY $P/summarize.py
}

[ $# -gt 0 ] || { sed -n 2,16p "$0"; exit 1; }
for x in "$@"; do
    if [ "$x" = all ]; then
        for e in e1 dispatch_cost kernel c1 table2 lam7 lattice ablation13 ablation7 c2 c3 table3 table5 table5_tp2 summarize; do
            $e
        done
    else
        "$x"
    fi
done
