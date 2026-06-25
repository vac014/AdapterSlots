#!/usr/bin/env bash
# Full reproduction from a clean checkout.
# Runtime: ~45 min (CPU simulation) or ~15-18 GPU-hours (real hardware).
#
# Usage:
#   bash benchmarks/reproduce/reproduce_all.sh                     # CPU simulation
#   GPU=a6000_single bash benchmarks/reproduce/reproduce_all.sh    # real A6000
#   GPU=two_h100_nvlink bash benchmarks/reproduce/reproduce_all.sh # real H100

set -euo pipefail

GPU=${GPU:-cpu}
SKIP_TESTS=${SKIP_TESTS:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-results}

echo "AdapterSlots reproduction -- GPU=$GPU, output=$OUTPUT_ROOT"

echo "[1/4] Verifying environment..."
conda run -n adapter_env python -c "import adapter_slots; print('AdapterSlots', adapter_slots.__version__)" \
    2>/dev/null || python -c "import adapter_slots; print('AdapterSlots', adapter_slots.__version__)"

if [[ "$SKIP_TESTS" != "1" ]]; then
    echo "[2/4] Running unit tests..."
    python -m pytest tests/ -q --tb=short 2>&1 | tail -5
else
    echo "[2/4] Skipping tests (SKIP_TESTS=1)"
fi

echo "[3/4] Running core experiments..."

echo "  E10: FlashInfer composability..."
GPU="$GPU" OUTPUT_DIR=results/flashinfer_composition bash benchmarks/reproduce/reproduce_e10.sh 2>&1 | tail -3

echo "  Multi-GPU correctness..."
case "$GPU" in
    cpu|a6000_single) TAU_ITER_MS=30.0 ;;
    two_a6000_pcie)   TAU_ITER_MS=100.0 ;;
    two_h100_nvlink)  TAU_ITER_MS=5.0 ;;
    *)                TAU_ITER_MS=30.0 ;;
esac
case "$GPU" in
    a6000_single|two_a6000_pcie|two_h100_nvlink) TP_MODE=live ;;
    *) TP_MODE=simulation ;;
esac
python scripts/test_tp_correctness.py --mode "$TP_MODE" --tau-iter-ms "$TAU_ITER_MS" \
    --output-dir results/multi_gpu_correctness/ 2>&1 | tail -3
python scripts/test_pp_correctness.py --mode simulation \
    --output-dir results/multi_gpu_correctness/ 2>&1 | tail -3

echo "  AB7: Preemption safety..."
python scripts/experiments/ab7_preemption.py --mode cpu \
    --K 4 --W 32 --output-dir results/multi_gpu_correctness/ 2>&1 | tail -3

echo "  KV stress test..."
python scripts/kv_stress_test.py --mode cpu \
    --K 4 --W 32 --output-dir results/multi_gpu_correctness/ 2>&1 | tail -3

echo "[4/4] Running ablations..."
bash benchmarks/reproduce/reproduce_ablations.sh 2>&1 | tail -5

echo ""
echo "Reproduction complete. Results in $OUTPUT_ROOT/."
echo "Figure generation and theorem validation are performed separately from results/."
