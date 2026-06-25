#!/bin/bash
# run_k_sweep.sh -- overnight K sweep for alignment_buffer WAR improvement experiment.
# Runs K=2,3,4,5 sequentially; each K is 18 cells (6 T_max × 3 rates × 5 min).
# Total wall-clock: ~12 hours.
set -euo pipefail
cd /workspace

for K in 2 3 4 5; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Starting K=${K} (18 cells, ~3 h) ==="
    python scripts/experiments/war_improvement_serving_benchmark.py \
        --model ./models/llama-7b \
        --adapter-dir ./adapters \
        --K "$K" \
        --tmax-values 0 1 2 5 10 20 \
        --request-rates 3 7 10 \
        --dataset-path ./data/sharegpt/sharegpt.jsonl \
        --output-dir "results/alignment_buffer/a6000_k${K}" \
        --label "Single A6000 K=${K} (TP=1)" \
        --duration 300
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === K=${K} done -- results/alignment_buffer/a6000_k${K}/war_improvement.csv ==="
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === ALL K SWEEPS COMPLETE ==="
