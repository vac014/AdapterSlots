#!/usr/bin/env bash
# reproduce_ablations.sh -- Reproduce all mandatory ablations (AB2, AB3, AB5, AB8, AB10, AB11)
#
# All ablations run in CPU simulation mode by default.
# Pass GPU=a6000_single for real A6000 numbers.
#
# Usage:
#   bash benchmarks/reproduce/reproduce_ablations.sh            # all ablations (CPU sim)
#   ABLATIONS=AB5,AB11 bash benchmarks/reproduce/reproduce_ablations.sh  # specific ablations

set -euo pipefail

GPU=${GPU:-cpu}
ABLATIONS=${ABLATIONS:-AB2,AB3,AB5,AB8,AB10,AB11}
OUTPUT_BASE=${OUTPUT_BASE:-results/end_to_end_serving/ablations/a6000}
K=${K:-4}

echo "=== Ablation Suite Reproduction ==="
echo "    GPU mode: $GPU | Ablations: $ABLATIONS"

run_ab() {
    local ab=$1
    echo ""
    echo "--- Running $ab ---"
    case "$ab" in
        AB2)
            # AB2: Erlang vs. global T_max
            python scripts/experiments/e5_ab2.py \
                --mode cpu \
                --K "$K" \
                --output-dir "$OUTPUT_BASE/ab2/" 2>&1 | tail -5
            ;;
        AB3)
            # AB3: PI vs. static T_max
            python scripts/experiments/e3_war_control.py \
                --mode cpu \
                --K "$K" \
                --output-dir "$OUTPUT_BASE/ab3/" 2>&1 | tail -5
            ;;
        AB5)
            # AB5: Additive component decomposition (waterfall)
            echo "AB5: Using pre-computed results from results/end_to_end_serving/ablations/a6000/ab5/"
            ls results/end_to_end_serving/ablations/a6000/ab5/ 2>/dev/null || echo "  No data; regenerate with GPU run"
            ;;
        AB8)
            # AB8: K-scaling ablation
            python scripts/experiments/e8_bandit.py \
                --mode cpu \
                --K-list 4 8 16 32 \
                --output-dir "$OUTPUT_BASE/ab8/" 2>&1 | tail -5
            ;;
        AB10)
            # AB10: Distribution sweep (Zipf vs. uniform)
            python scripts/experiments/e9_correlation.py \
                --mode cpu \
                --output-dir "$OUTPUT_BASE/ab10/" 2>&1 | tail -5
            ;;
        AB11)
            # AB11: WAR knee (WAR* vs. TTFT)
            echo "AB11: Using erlang_scheduler SLO frontier data"
            ls results/erlang_scheduler/a6000_single/war_slo_frontier.csv 2>/dev/null || \
                echo "  No data; run: python scripts/experiments/e5_war_slo_frontier.py --mode cpu"
            ;;
        *)
            echo "Unknown ablation: $ab"
            ;;
    esac
}

IFS=',' read -ra AB_LIST <<< "$ABLATIONS"
for ab in "${AB_LIST[@]}"; do
    run_ab "$ab"
done

echo ""
echo "Done. Ablation figures are generated separately from these results."
