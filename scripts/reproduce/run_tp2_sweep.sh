#!/bin/bash
# run_tp2_sweep.sh -- overnight TP=2 WAR sweep for alignment_buffer §5.5 (Two A6000 PCIe).
# Phase 1: K=4, TP=2, 18 cells (6 T_max × 3 rates × 5 min) -- covers §4.9 and §4.10.
# Phase 2: K=16, TP=2, 6 cells (6 T_max × rate=7 × 5 min) -- covers §4.12 (Theorem 11.1).
# Total wall-clock: ~2 hours. Disconnect-safe via nohup.
set -euo pipefail
cd /workspace
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Kill any stale sweep or vLLM processes from a prior run before we start.
# Without this, an orphaned war_improvement_serving_benchmark.py (left behind when bash
# is killed via `kill <PID>` while the Python child keeps running) will
# compete for port 8000 and truncate batch log files mid-write → WAR=nan.
pkill -9 -f "war_improvement_serving_benchmark" 2>/dev/null || true
pkill -9 -f "patched_api_server"    2>/dev/null || true
sleep 2  # let the kernel release port 8000 and file locks

mkdir -p results/alignment_buffer/two_a6000_pcie
mkdir -p results/alignment_buffer/two_a6000_pcie_k16
mkdir -p logs

# Phase 1: K=4 (18 cells, ~90 min)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Phase 1: K=4 TP=2 (18 cells, ~90 min) ==="
python scripts/experiments/war_improvement_serving_benchmark.py \
    --model ./models/llama-7b \
    --adapter-dir ./adapters \
    --K 4 \
    --tensor-parallel-size 2 \
    --tmax-values 0 1 2 5 10 20 \
    --request-rates 3 7 10 \
    --dataset-path ./data/sharegpt/sharegpt.jsonl \
    --output-dir results/alignment_buffer/two_a6000_pcie \
    --label "Two A6000 PCIe K=4 (TP=2)" \
    --duration 300

# war_transparency.csv (§4.9) and throughput_comparison.csv (§4.10) are the same sweep.
cp results/alignment_buffer/two_a6000_pcie/war_improvement.csv \
   results/alignment_buffer/two_a6000_pcie/war_transparency.csv
cp results/alignment_buffer/two_a6000_pcie/war_improvement.csv \
   results/alignment_buffer/two_a6000_pcie/throughput_comparison.csv

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Phase 1 done -- results/alignment_buffer/two_a6000_pcie/war_improvement.csv ==="

# Phase 2: K=16 (6 cells, ~30 min) -- Theorem 11.1 at large K
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Phase 2: K=16 TP=2 (6 cells, ~30 min) ==="
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
    --duration 300

# Copy into the shared two_a6000_pcie dir for §4.12 and §4.14 checks.
cp results/alignment_buffer/two_a6000_pcie_k16/war_improvement.csv \
   results/alignment_buffer/two_a6000_pcie/war_transparency_k16.csv

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Phase 2 done -- results/alignment_buffer/two_a6000_pcie/war_transparency_k16.csv ==="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === ALL TP=2 SWEEPS COMPLETE ==="
