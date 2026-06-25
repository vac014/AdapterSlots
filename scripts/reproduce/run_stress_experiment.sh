#!/usr/bin/env bash
# run_stress_experiment.sh -- K=16 stress experiment for §2.7.3 WARτ/TTFT correlation.
#
# Uses stress_k16_rate2000_n20000.jsonl replayed at speed_mult=0.0125 so the
# effective arrival rate is 2000×0.0125 = 25 req/s (~1.8× server capacity).
# This causes adapters 0–3 (Zipf-dominant) to accumulate ≥32 concurrent requests
# (condition A fires, WARτ≈0ms) while adapters 4–15 wait T_max (condition B,
# WARτ≈T_max).  The bimodal WARτ distribution gives ρ(WARτ,TTFT) ≥ 0.5.
#
# Replay duration at 0.0125×: 10 s trace span / 0.0125 ≈ 800 s.
# The server is kept alive until the harness exits (no fixed DURATION).
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash scripts/reproduce/run_stress_experiment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}"

PORT=8000
LOG_DIR="results/instrumentation/stress_logs_a6000_pcie"
WORKLOAD="workloads/stress_k16_rate2000_n20000.jsonl"
# speed_mult = 25 / 2000 = 0.0125  →  effective λ ≈ 25 req/s on the server
SPEED_MULT="0.0125"
MODEL="./models/llama-7b"
ADAPTERS="./adapters"
REPLAY_OUT="${LOG_DIR}/replay_client.csv"
EC5_OUT="results/instrumentation/stress_ec5_pairs.csv"

echo "[stress] =========================================="
echo "[stress] K=16 stress experiment (λ≈25 req/s effective, T_max=3000ms)"
echo "[stress] trace: ${WORKLOAD}  speed_mult: ${SPEED_MULT}"
echo "[stress] =========================================="

# Backup old results
if [[ -d "${LOG_DIR}" ]]; then
    TS=$(date +%Y%m%d_%H%M%S)
    mv "${LOG_DIR}" "${LOG_DIR}_backup_${TS}"
fi
mkdir -p "${LOG_DIR}/gpu0"

# Build lora-modules args (K=16)
LORA_ARGS=()
for i in $(seq 0 15); do
    SEED=$((42 + i))
    AP="${ADAPTERS}/adapter_r16_k${i}_s${SEED}"
    if [[ -d "${AP}" ]]; then
        LORA_ARGS+=("adapter_${i}=${AP}")
    fi
done
echo "[stress] Registering ${#LORA_ARGS[@]} adapters"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export AS_SCHEDULER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export AS_TMAX_MS=12000
export AS_TTFT_SLO_MS=20000
export AS_MODE=threshold
export AS_LOG_WAR=1
export AS_METRICS_PATH="${LOG_DIR}/gpu0/metrics.jsonl"

echo "[stress] Starting vLLM server ..."
python scripts/vllm_serve_adapter_slots.py \
    --model "${MODEL}" \
    --enable-lora \
    --max-loras 16 \
    --max-lora-rank 16 \
    --lora-modules "${LORA_ARGS[@]}" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.88 \
    --port "${PORT}" \
    > "${LOG_DIR}/gpu0/stdout.log" 2> "${LOG_DIR}/gpu0/stderr.log" &
SERVER_PID=$!

echo "[stress] Waiting for server (PID=${SERVER_PID}) ..."
SERVER_READY=0
for i in $(seq 1 300); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[stress] Ready after ${i}s"
        SERVER_READY=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[stress] ERROR: server died. Check ${LOG_DIR}/gpu0/stderr.log" >&2
        tail -20 "${LOG_DIR}/gpu0/stderr.log" >&2; exit 1
    fi
    sleep 1
done
if [[ "${SERVER_READY}" -eq 0 ]]; then
    echo "[stress] ERROR: server not ready after 300s. Check ${LOG_DIR}/gpu0/stderr.log" >&2
    kill -9 "${SERVER_PID}" 2>/dev/null || true; exit 1
fi

echo "[stress] Running replay at speed_mult=${SPEED_MULT} (effective λ≈25 req/s, ~800s replay) ..."
python scripts/replay_harness.py \
    --trace "${WORKLOAD}" \
    --endpoint "http://localhost:${PORT}/v1/completions" \
    --speed-multiplier "${SPEED_MULT}" \
    --model "llama-7b" \
    --adapter-prefix "adapter_" \
    --warp-size 32 \
    --output "${REPLAY_OUT}" \
    --timeout-s 300.0 \
    --max-concurrent 150 \
    2>&1 | tee "${LOG_DIR}/harness.log"

echo "[stress] Stopping server ..."
kill -TERM "${SERVER_PID}" 2>/dev/null || true
sleep 5; kill -9 "${SERVER_PID}" 2>/dev/null || true

echo "[stress] Computing EC5 correlation ..."
python analysis/compute_ec5_correlation.py \
    --metrics "${LOG_DIR}/gpu0/metrics.jsonl" \
    --harness "${REPLAY_OUT}" \
    --output "${EC5_OUT}" \
    --update-csv "results/instrumentation/wartau_stress_correlations_a6000_pcie.csv"

echo "[stress] Done. EC5 pairs: ${EC5_OUT}"
