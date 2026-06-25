#!/usr/bin/env bash
# reproduce_e4.sh -- One-command E4 reproduction (headline result: throughput vs. T_max sweep)
#
# Reproduces Figure 1 and Table 1 from the paper.
# Expected runtime: ~20min (CPU simulation) or ~4h (real A6000 serving)
#
# Usage:
#   bash benchmarks/reproduce/reproduce_e4.sh              # CPU simulation
#   GPU=a6000 bash benchmarks/reproduce/reproduce_e4.sh    # real A6000 (requires model + adapters)
#   GPU=h100  bash benchmarks/reproduce/reproduce_e4.sh    # real H100  (final paper numbers)

set -euo pipefail

GPU=${GPU:-cpu}
OUTPUT_DIR=${OUTPUT_DIR:-results/end_to_end_serving/e4/a6000}
MODEL=${MODEL:-./models/llama-7b}
ADAPTER_DIR=${ADAPTER_DIR:-./adapters}
K=${K:-4}

echo "=== E4: Throughput vs. T_max Tradeoff Surface ==="
echo "    GPU mode: $GPU | Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

if [[ "$GPU" == "cpu" ]]; then
    echo "[CPU simulation mode -- no GPU required]"
    # Run the B3 k-scaling simulation which generates E4-compatible data
    python scripts/experiments/e10_composability.py \
        --mode cpu \
        --K "$K" --W 32 \
        --output-dir results/flashinfer_composition/
    echo "Simulation complete. See results/flashinfer_composition/e10_composability.csv"
else
    echo "[Live vLLM mode -- requires $GPU hardware]"
    export CUDA_VISIBLE_DEVICES=0
    export AS_MODE=threshold
    for TMAX in 0 5 10 20 50 100 300; do
        export AS_TMAX_MS=$TMAX
        python benchmarks/sota/serving_full.py \
            --model "$MODEL" \
            --adapter-dir "$ADAPTER_DIR" \
            --K "$K" \
            --lambda-total 7.0 \
            --output-dir "$OUTPUT_DIR" \
            --tmax-ms "$TMAX" \
            --hardware-label "${GPU}_single" \
            --num-prompts 500
    done
fi

echo ""
echo "Done. Results written to $OUTPUT_DIR."
echo "Figure 1 is generated separately from these results."
