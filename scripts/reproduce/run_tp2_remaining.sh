#!/bin/bash
# run_tp2_remaining.sh -- finishes §4.11 and §4.12 on Two A6000 PCIe.
# §4.9 Phase 1 (K=4 TP=2, 18 cells) is ALREADY DONE -- this script does NOT re-run it.
# §4.10 analysis uses throughput_comparison.csv already present from Phase 1 -- no GPU run.
# §4.11 dispatch overhead: pure Python, no GPU, <1 min.
# §4.12 K=16 TP=2 sweep: 6 cells, ~30 min.
# Total wall-clock: ~35 min. Disconnect-safe via nohup.
set -euo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Kill any stale sweep or vLLM processes from a prior run before we start.
pkill -9 -f "war_improvement_serving_benchmark"    2>/dev/null || true
pkill -9 -f "alignment_buffer_dispatch_overhead"  2>/dev/null || true
pkill -9 -f "patched_api_server"       2>/dev/null || true
sleep 2

mkdir -p results/alignment_buffer/two_a6000_pcie
mkdir -p results/alignment_buffer/two_a6000_pcie_k16
mkdir -p logs

# §4.11 Dispatch overhead at K=16, TP=2 (pure Python, no GPU)
# Skip if already generated (idempotent).
if [ -f results/alignment_buffer/two_a6000_pcie/dispatch_overhead_k16.csv ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === §4.11: dispatch_overhead_k16.csv already present -- skipping ==="
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === §4.11: dispatch overhead K=16 TP=2 ==="
    python scripts/experiments/alignment_buffer_dispatch_overhead.py \
        --K-values 4 8 16 \
        --warp-size 32 \
        --n-reps 10000 \
        --output results/alignment_buffer/two_a6000_pcie/dispatch_overhead_k16.csv \
        --label "Two A6000 PCIe (TP=2)"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === §4.11 done ==="
fi

# §4.12 Theorem 11.1 at K=16 -- TP=2 sweep (6 cells, ~30 min)
# rate=7 matches §4.9 (K=4 TP=2) -- same rate, same hardware, only K changes.
# K=4: N/K≈64 >> W=32 → WAR=0.55.  K=16: N/K≈16 < W=32 → WAR≈0.
# Both cells at rate=7 validate Theorem 11.1 across below- and above-threshold
# regimes without confounding K with rate.  K=16 above-threshold validation
# (N/K≥W) requires H100 throughput (§4.15) -- not achievable on PCIe A6000.
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === §4.12: K=16 TP=2 sweep (6 cells, ~30 min) ==="
python scripts/experiments/war_improvement_serving_benchmark.py \
    --model ./models/llama-7b \
    --adapter-dir ./adapters \
    --K 16 \
    --tensor-parallel-size 2 \
    --tmax-values 0 1 2 5 10 20 \
    --request-rates 7 \
    --dataset-path ./data/sharegpt/sharegpt.jsonl \
    --output-dir results/alignment_buffer/two_a6000_pcie_k16 \
    --label "Two A6000 PCIe K=16 (TP=2)" \
    --duration 300 \
    --gpu-memory-utilization 0.82

cp results/alignment_buffer/two_a6000_pcie_k16/war_improvement.csv \
   results/alignment_buffer/two_a6000_pcie/war_transparency_k16.csv

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === §4.12 done -- results/alignment_buffer/two_a6000_pcie/war_transparency_k16.csv ==="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === §4.11–§4.12 COMPLETE ==="
