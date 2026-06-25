#!/usr/bin/env bash
# reproduce_e10.sh -- E10 FlashInfer + AdapterSlots composability reproduction
#
# Reproduces Figure 6 and Section 5 of the paper.
# EC 11.1: Gain(FlashInfer+AdapterSlots) >= Gain(FlashInfer) + Gain(AdapterSlots) - 0.05
#
# Usage:
#   bash benchmarks/reproduce/reproduce_e10.sh                  # CPU simulation
#   GPU=a6000_single bash benchmarks/reproduce/reproduce_e10.sh  # single A6000
#   GPU=two_a6000_pcie bash benchmarks/reproduce/reproduce_e10.sh # two A6000 PCIe (simulation)

set -euo pipefail

GPU=${GPU:-cpu}
OUTPUT_DIR=${OUTPUT_DIR:-results/flashinfer_composition}
K=${K:-4}

echo "=== E10: FlashInfer + AdapterSlots Composability Experiment ==="
echo "    GPU mode: $GPU | Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

case "$GPU" in
    cpu|a6000_single)
        python scripts/experiments/e10_composability.py \
            --mode a6000_single \
            --K "$K" --W 32 \
            --lambda-total 7.0 \
            --output-dir "$OUTPUT_DIR"
        ;;
    two_a6000_pcie)
        # Simulation only for TP=2 (no FlashInfer install required)
        python scripts/experiments/e10_composability.py \
            --mode two_a6000_pcie \
            --K "$K" --W 32 \
            --lambda-total 7.0 \
            --output-dir "$OUTPUT_DIR/two_a6000_pcie"
        ;;
    h100|two_h100_nvlink)
        python scripts/experiments/e10_composability.py \
            --mode two_h100_nvlink \
            --K "$K" --W 32 \
            --lambda-total 7.0 \
            --output-dir "${OUTPUT_DIR}_h100"
        ;;
    *)
        echo "Unknown GPU mode: $GPU (use: cpu, a6000_single, two_a6000_pcie, h100)"
        exit 1
        ;;
esac

# Check EC 11.1
python -c "
import csv
rows = list(csv.DictReader(open('$OUTPUT_DIR/e10_composability.csv')))
fi_gain = next(float(r['gain_vs_vllm']) for r in rows if r['system'] == 'flashinfer')
as_gain = next(float(r['gain_vs_vllm']) for r in rows if r['system'] == 'adapterslots')
comb_gain = next(float(r['gain_vs_vllm']) for r in rows if r['system'] == 'combined')
threshold = fi_gain + as_gain - 0.05
ec11 = comb_gain >= threshold
print(f'EC 11.1: Gain(comb)={comb_gain:.1%} >= Gain(FI)+Gain(AdapterSlots)-5%={threshold:.1%}: {\"PASS\" if ec11 else \"FAIL\"}')
"

echo ""
echo "Done. Figure 6 is generated separately from these results."
echo "Data  → $OUTPUT_DIR/e10_composability.csv"
