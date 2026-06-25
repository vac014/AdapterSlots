#!/usr/bin/env bash
# run_e12_dp2_only.sh -- E12 DP=2 experiments (3 reps at K=50 and K=100)
#
# PP=2 was ruled out: pipeline bubbles in small-batch decode mode cause ~56×
# throughput regression (8.2 tok/s vs 460 tok/s baseline) with vLLM 0.6.x.
#
# DP=2: two independent single-GPU servers, adapter-aware routing (k%2).
#   - τ_iter ≈ 30ms per server (same as single A6000 AB10 which showed +80%)
#   - cold_boost = 2.0  (pcie-min-deferral is the hard 96ms block; cold_boost=2.0
#     is the soft Whittle penalty -- matches K=50 TP=2 that gave +5.77%/+4.21%)
#   - K=50: each server gets 25 adapters, λ_server≈2 req/s → λ_per_adapter≈0.08 req/s
#   - K=100: each server gets 50 adapters, λ_server≈2 req/s → λ_per_adapter≈0.04 req/s
#
# K=50 is the target: per-adapter rate matches the regime where TP=2 PCIe gave
# +3.16% (C3 ±0.07%).  With τ_iter=30ms instead of 100ms, WAR should be stronger.

set -euo pipefail

MODEL="${MODEL:-./models/llama-7b}"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters}"
DATASET="${DATASET:-./data/sharegpt/sharegpt.jsonl}"
N_REPS="${N_REPS:-3}"
LAMBDA="${LAMBDA:-4.0}"
NUM_PROMPTS="${NUM_PROMPTS:-400}"
WARMUP="${WARMUP:-20}"
MAX_TOKENS="${MAX_TOKENS:-128}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# pyairports stub
PYAIRPORTS_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/pyairports
if [ ! -f "$PYAIRPORTS_DIR/__init__.py" ]; then
    mkdir -p "$PYAIRPORTS_DIR"
    touch "$PYAIRPORTS_DIR/__init__.py"
    echo "AIRPORT_LIST = []" > "$PYAIRPORTS_DIR/airports.py"
    echo "[setup] pyairports stub created"
fi

echo "============================================================"
echo "E12 DP=2 Experiments"
echo "  λ=${LAMBDA}  reps=${N_REPS}  num_prompts=${NUM_PROMPTS}"
echo "  AS_MODE=sort (warm-first adapter sort, no deferral)  cold_boost=1.0 (unused)"
echo "  Routing: adapter_k → server k%2 (interleaved Zipf balance)"
echo "  $(date)"
echo "============================================================"

run_dp2_config() {
    local K="$1"
    local K_WARM="$2"
    local label="k${K}"
    local outdir="results/adapter_prefetching/dp2/reps_${label}"
    mkdir -p "$outdir"

    echo ""
    echo "━━━ DP=2  K=${K}  K_warm=${K_WARM}  (3 reps) ━━━"

    for rep in $(seq 1 "$N_REPS"); do
        OUT="${outdir}/e12_dp2_${label}_lam${LAMBDA%.*}_r${rep}.csv"
        echo ""
        echo "--- DP=2 K=${K} Rep ${rep}/${N_REPS} ($(date +%H:%M)) ---"
        CUDA_VISIBLE_DEVICES=0,1 python scripts/experiments/prefetch_policy_ablation.py \
            --mode policy-ablation \
            --model "$MODEL" \
            --adapter-dir "$ADAPTER_DIR" \
            --K "$K" --K-warm "$K_WARM" \
            --lambda-total "$LAMBDA" \
            --hardware-label "two_a6000_dp2" \
            --tp-size 1 --dp-mode \
            --tau-iter-ms 30.0 \
            --cold-boost 1.0 \
            --num-prompts "$NUM_PROMPTS" --warmup-prompts "$WARMUP" --max-tokens "$MAX_TOKENS" \
            --dataset-path "$DATASET" \
            --output-dir "$outdir" \
            --port 8250 2>&1 | tee -a "${outdir}/dp2_${label}_rep${rep}.log"

        LATEST="${outdir}/e12_prefetch_ablation_two_a6000_dp2.csv"
        [ -f "$LATEST" ] && mv "$LATEST" "$OUT" && echo "  → saved: $OUT"
    done

    echo ""
    echo "━━━ DP=2 K=${K} 3-rep aggregate ━━━"
    python - "$outdir" "$label" <<'PYEOF'
import csv, glob, statistics, sys
from collections import defaultdict
rdir, label = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{rdir}/e12_dp2_{label}_*.csv"))
if not files:
    print("  No result files found")
    sys.exit(0)
rows_by_label = defaultdict(list)
for f in files:
    with open(f) as fh:
        for row in csv.DictReader(fh):
            rows_by_label[row["label"]].append(float(row["tput_tok_s"]))
base_vals = rows_by_label.get("C0:vLLM-LRU", [1.0])
base_mean = statistics.mean(base_vals)
print(f"  {'Condition':<25} {'Reps':>5} {'Mean±Std':>14}  {'Gain':>8}  {'All>0':>6}")
for lbl in ["C0:vLLM-LRU", "C1:AdapterSlots-WAR", "C2:AdapterSlots-PredLFU", "C3:AdapterSlots-Combined"]:
    vals = rows_by_label.get(lbl, [])
    if not vals: continue
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    gain = (m - base_mean) / max(base_mean, 1.0) * 100
    gains_pos = all((v - base_mean) / base_mean * 100 > 0 for v in vals) if lbl != "C0:vLLM-LRU" else None
    ec = " EC12.3 PASS ✓" if gain >= 5.0 and "Combined" in lbl else ""
    pos_str = "YES" if gains_pos else ("NO" if gains_pos is False else "--")
    print(f"  {lbl:<25} {len(vals):>5}  {m:6.1f}±{s:4.1f} tok/s  {gain:+6.2f}%  {pos_str:>6}{ec}")
PYEOF
}

# K=50 (primary -- matches per-adapter rate regime of successful K=50 TP=2)
run_dp2_config 50 25

# K=100 (secondary -- EC 12.3 target, per-adapter rate halved vs K=50)
run_dp2_config 100 50

echo ""
echo "============================================================"
echo "All E12 DP=2 experiments done  ($(date))"
echo "  K=50: results/adapter_prefetching/dp2/reps_k50/"
echo "  K=100: results/adapter_prefetching/dp2/reps_k100/"
echo "============================================================"
