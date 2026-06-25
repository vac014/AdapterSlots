#!/usr/bin/env bash
# run_e12_pp2_dp2.sh -- E12 PP=2 and DP=2 experiments (3 reps each)
#
# Motivation: TP=2 PCIe τ_iter=100ms collapses WAR (T_max < τ_iter makes deferral
# trivially short).  PP=2 (pipeline parallel) and DP=2 (data parallel) both bring
# τ_iter to ~30-45ms -- the same regime where AB10 showed +80% single-A6000 gains.
#
# PP=2: one server, both GPUs, layers split vertically.  τ_iter ≈ 45ms estimated.
#   cold_boost = ceil(96.3/45)+1 = ceil(2.14)+1 = 4  (pcie-auto-cold-boost handles this)
# DP=2: two single-GPU servers, adapter-aware routing (adapter_k → server k%2).
#   τ_iter ≈ 30ms per server.
#   cold_boost = ceil(96.3/30)+1 = 5  (pcie-auto-cold-boost handles this)
#
# Each rep runs all 4 conditions (C0..C3) sequentially.  All CSV outputs go to
# results/adapter_prefetching/{pp2,dp2}/reps/.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/reproduce/run_e12_pp2_dp2.sh

set -euo pipefail

MODEL="${MODEL:-./models/llama-7b}"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters}"
DATASET="${DATASET:-./data/sharegpt/sharegpt.jsonl}"
N_REPS="${N_REPS:-3}"
K="${K:-50}"
K_WARM="${K_WARM:-25}"
LAMBDA="${LAMBDA:-4.0}"
NUM_PROMPTS="${NUM_PROMPTS:-400}"
WARMUP="${WARMUP:-20}"
MAX_TOKENS="${MAX_TOKENS:-128}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# pyairports stub (required, may not persist across envs)
PYAIRPORTS_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/pyairports
if [ ! -f "$PYAIRPORTS_DIR/__init__.py" ]; then
    mkdir -p "$PYAIRPORTS_DIR"
    touch "$PYAIRPORTS_DIR/__init__.py"
    echo "AIRPORT_LIST = []" > "$PYAIRPORTS_DIR/airports.py"
    echo "[setup] pyairports stub created"
fi

echo "============================================================"
echo "E12 PP=2 + DP=2 Experiments"
echo "  K=${K}  K_warm=${K_WARM}  λ=${LAMBDA}  reps=${N_REPS}"
echo "  num_prompts=${NUM_PROMPTS}  warmup=${WARMUP}  max_tokens=${MAX_TOKENS}"
echo "  $(date)"
echo "============================================================"

# PP=2 (3 reps, K=50)
PP2_DIR="results/adapter_prefetching/pp2/reps"
mkdir -p "$PP2_DIR"

echo ""
echo "━━━ PP=2 experiments (τ_iter≈45ms, --tp-size 1 --pp-size 2) ━━━"
echo "  cold_boost = ceil(96.3/45)+1 = ceil(2.14)+1 = 4  (auto)"

for rep in $(seq 1 "$N_REPS"); do
    OUT="${PP2_DIR}/e12_pp2_k${K}_lam${LAMBDA%.*}_r${rep}.csv"
    echo ""
    echo "--- PP=2 Rep ${rep}/${N_REPS} ($(date +%H:%M)) ---"
    CUDA_VISIBLE_DEVICES=0,1 python scripts/experiments/prefetch_policy_ablation.py \
        --mode policy-ablation \
        --model "$MODEL" \
        --adapter-dir "$ADAPTER_DIR" \
        --K "$K" --K-warm "$K_WARM" \
        --lambda-total "$LAMBDA" \
        --hardware-label "two_a6000_pp2" \
        --tp-size 1 --pp-size 2 \
        --tau-iter-ms 45.0 \
        --pcie-auto-cold-boost --pcie-min-deferral \
        --num-prompts "$NUM_PROMPTS" --warmup-prompts "$WARMUP" --max-tokens "$MAX_TOKENS" \
        --dataset-path "$DATASET" \
        --output-dir "$PP2_DIR" \
        --port 8240 2>&1 | tee -a "${PP2_DIR}/pp2_rep${rep}.log"

    # Rename output to rep-stamped name if present
    LATEST="${PP2_DIR}/e12_prefetch_ablation_two_a6000_pp2.csv"
    [ -f "$LATEST" ] && mv "$LATEST" "$OUT" && echo "  → saved: $OUT"
done

echo ""
echo "━━━ PP=2 3-rep aggregate ━━━"
python - <<'PYEOF'
import csv, glob, statistics, os
rdir = "results/adapter_prefetching/pp2/reps"
files = sorted(glob.glob(f"{rdir}/e12_pp2_*.csv"))
if not files:
    print("  No PP=2 result files found")
else:
    from collections import defaultdict
    rows_by_label = defaultdict(list)
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                rows_by_label[row["label"]].append(float(row["tput_tok_s"]))
    base_mean = statistics.mean(rows_by_label.get("C0:vLLM-LRU", [0]))
    print(f"  {'Condition':<25} {'Mean±Std':>14}  {'Gain':>8}")
    for label in ["C0:vLLM-LRU", "C1:AdapterSlots-WAR", "C2:AdapterSlots-PredLFU", "C3:AdapterSlots-Combined"]:
        vals = rows_by_label.get(label, [])
        if not vals:
            print(f"  {label:<25}  no data")
            continue
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        gain = (m - base_mean) / max(base_mean, 1.0) * 100
        ec = "EC PASS ✓" if gain >= 5.0 and label == "C3:AdapterSlots-Combined" else ""
        print(f"  {label:<25}  {m:6.1f}±{s:4.1f} tok/s  {gain:+6.2f}%  {ec}")
PYEOF

# DP=2 (3 reps, K=50)
DP2_DIR="results/adapter_prefetching/dp2/reps"
mkdir -p "$DP2_DIR"

echo ""
echo "━━━ DP=2 experiments (τ_iter≈30ms, two single-GPU servers) ━━━"
echo "  cold_boost = ceil(96.3/30)+1 = 5  (auto)"
echo "  Routing: adapter_k → server k%2 (interleaved, balanced Zipf load)"

for rep in $(seq 1 "$N_REPS"); do
    OUT="${DP2_DIR}/e12_dp2_k${K}_lam${LAMBDA%.*}_r${rep}.csv"
    echo ""
    echo "--- DP=2 Rep ${rep}/${N_REPS} ($(date +%H:%M)) ---"
    CUDA_VISIBLE_DEVICES=0,1 python scripts/experiments/prefetch_policy_ablation.py \
        --mode policy-ablation \
        --model "$MODEL" \
        --adapter-dir "$ADAPTER_DIR" \
        --K "$K" --K-warm "$K_WARM" \
        --lambda-total "$LAMBDA" \
        --hardware-label "two_a6000_dp2" \
        --tp-size 1 --dp-mode \
        --tau-iter-ms 30.0 \
        --pcie-auto-cold-boost --pcie-min-deferral \
        --num-prompts "$NUM_PROMPTS" --warmup-prompts "$WARMUP" --max-tokens "$MAX_TOKENS" \
        --dataset-path "$DATASET" \
        --output-dir "$DP2_DIR" \
        --port 8250 2>&1 | tee -a "${DP2_DIR}/dp2_rep${rep}.log"

    LATEST="${DP2_DIR}/e12_prefetch_ablation_two_a6000_dp2.csv"
    [ -f "$LATEST" ] && mv "$LATEST" "$OUT" && echo "  → saved: $OUT"
done

echo ""
echo "━━━ DP=2 3-rep aggregate ━━━"
python - <<'PYEOF'
import csv, glob, statistics
from collections import defaultdict
rdir = "results/adapter_prefetching/dp2/reps"
files = sorted(glob.glob(f"{rdir}/e12_dp2_*.csv"))
if not files:
    print("  No DP=2 result files found")
else:
    rows_by_label = defaultdict(list)
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                rows_by_label[row["label"]].append(float(row["tput_tok_s"]))
    base_mean = statistics.mean(rows_by_label.get("C0:vLLM-LRU", [0]))
    print(f"  {'Condition':<25} {'Mean±Std':>14}  {'Gain':>8}")
    for label in ["C0:vLLM-LRU", "C1:AdapterSlots-WAR", "C2:AdapterSlots-PredLFU", "C3:AdapterSlots-Combined"]:
        vals = rows_by_label.get(label, [])
        if not vals:
            print(f"  {label:<25}  no data")
            continue
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        gain = (m - base_mean) / max(base_mean, 1.0) * 100
        ec = "EC PASS ✓" if gain >= 5.0 and label == "C3:AdapterSlots-Combined" else ""
        print(f"  {label:<25}  {m:6.1f}±{s:4.1f} tok/s  {gain:+6.2f}%  {ec}")
PYEOF

echo ""
echo "============================================================"
echo "All E12 PP=2 + DP=2 experiments done  ($(date))"
echo "  PP=2 results: results/adapter_prefetching/pp2/reps/"
echo "  DP=2 results: results/adapter_prefetching/dp2/reps/"
echo "============================================================"
