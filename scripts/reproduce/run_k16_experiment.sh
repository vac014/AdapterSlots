#!/usr/bin/env bash
# run_k16_experiment.sh -- End-to-end K=16 TP=2 serving experiment for instrumentation §2.7.2
#
# 1. Backs up old k16_logs_a6000_pcie results
# 2. Launches vLLM (TP=2, K=16 adapters) and waits for ready
# 3. Runs replay harness with zipf_k16_lam7_n5000.jsonl
# 4. Stops server, then builds real wartau/ttft CSV and logs summary stats
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash scripts/reproduce/run_k16_experiment.sh [--speed-mult 5.0] [--duration 800]
#
# Output files (in results/instrumentation/k16_logs_a6000_pcie/):
#   gpu0/metrics.jsonl      -- per-batch WAR/WARτ metrics
#   replay_client.csv       -- per-request TTFT from harness
#   server.log              -- server start/stop log
# After the run, also writes:
#   results/instrumentation/ttft_per_request.csv  -- real (wartau_ms, ttft_ms) pairs for EC5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}"

SPEED_MULT=5.0
SERVER_DURATION=900
PORT=8000
LOG_DIR="results/instrumentation/k16_logs_a6000_pcie"
WORKLOAD="workloads/zipf_k16_lam7_n5000.jsonl"
MODEL="./models/llama-7b"
ADAPTERS="./adapters"
REPLAY_OUT="results/instrumentation/k16_harness_results.csv"
TTFT_OUT="results/instrumentation/ttft_per_request.csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --speed-mult)  SPEED_MULT="$2"; shift 2 ;;
        --duration)    SERVER_DURATION="$2"; shift 2 ;;
        --port)        PORT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

echo "[run_k16] =========================================="
echo "[run_k16] K=16 TP=2 serving experiment"
echo "[run_k16] Workload:   ${WORKLOAD}"
echo "[run_k16] Speed mult: ${SPEED_MULT}x"
echo "[run_k16] Server dur: ${SERVER_DURATION}s"
echo "[run_k16] =========================================="

# 1. Backup old results and start fresh
if [[ -d "${LOG_DIR}" ]]; then
    TS=$(date +%Y%m%d_%H%M%S)
    BACKUP="${LOG_DIR}_backup_${TS}"
    echo "[run_k16] Backing up ${LOG_DIR} → ${BACKUP}"
    mv "${LOG_DIR}" "${BACKUP}"
fi
mkdir -p "${LOG_DIR}/gpu0"

# 2. Build lora-modules args
LORA_ARGS=()
for i in $(seq 0 15); do
    SEED=$((42 + i))
    AP="${ADAPTERS}/adapter_r16_k${i}_s${SEED}"
    if [[ -d "${AP}" ]]; then
        LORA_ARGS+=("adapter_${i}=${AP}")
    else
        echo "[run_k16] WARN: missing adapter ${AP}" >&2
    fi
done
N_ADAPTERS=${#LORA_ARGS[@]}
echo "[run_k16] Registering ${N_ADAPTERS} adapters (0–$((N_ADAPTERS-1)))"
if [[ ${N_ADAPTERS} -lt 16 ]]; then
    echo "[run_k16] ERROR: only ${N_ADAPTERS}/16 adapters found" >&2
    exit 1
fi

# 3. NCCL config for PCIe multi-GPU (no SHM needed with 63GB /dev/shm)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
echo "[run_k16] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 4. AdapterSlots scheduler env
export AS_SCHEDULER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export AS_TMAX_MS="${AS_TMAX_MS:-3000}"
export AS_MODE="${AS_MODE:-threshold}"
export AS_LOG_WAR=1
export AS_METRICS_PATH="${LOG_DIR}/gpu0/metrics.jsonl"

# 5. Start vLLM server
echo "[run_k16] Starting vLLM server (TP=2, max-loras=16) on port ${PORT} ..."
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
echo "[run_k16] Server PID=${SERVER_PID}"

# 6. Wait for server ready
echo "[run_k16] Waiting for vLLM to be ready (may take 60–120s) ..."
READY=0
for i in $(seq 1 300); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        READY=1
        echo "[run_k16] Server ready after ${i}s"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[run_k16] ERROR: vLLM server exited. Check ${LOG_DIR}/gpu0/stderr.log" >&2
        tail -20 "${LOG_DIR}/gpu0/stderr.log" >&2
        exit 1
    fi
    sleep 1
done

if [[ "${READY}" -eq 0 ]]; then
    echo "[run_k16] ERROR: Server not ready after 300s" >&2
    kill -9 "${SERVER_PID}" 2>/dev/null || true
    exit 1
fi

# Record server ready timestamp
echo "[run_k16] Server ready at $(date)" | tee "${LOG_DIR}/server.log"

# 7. Run replay harness
echo "[run_k16] Running replay harness (speed=${SPEED_MULT}x) ..."
python scripts/replay_harness.py \
    --trace "${WORKLOAD}" \
    --endpoint "http://localhost:${PORT}/v1/completions" \
    --speed-multiplier "${SPEED_MULT}" \
    --model "llama-7b" \
    --adapter-prefix "adapter_" \
    --output "${REPLAY_OUT}" \
    --timeout-s 180.0 \
    --max-concurrent 300 \
    2>&1 | tee "${LOG_DIR}/harness.log"
HARNESS_RC=${PIPESTATUS[0]}
echo "[run_k16] Harness finished (exit code ${HARNESS_RC})"

# 8. Stop server
echo "[run_k16] Stopping server (PID=${SERVER_PID}) ..."
kill -TERM "${SERVER_PID}" 2>/dev/null || true
TRIES=0
while kill -0 "${SERVER_PID}" 2>/dev/null && [[ ${TRIES} -lt 30 ]]; do
    sleep 1; TRIES=$((TRIES + 1))
done
kill -9 "${SERVER_PID}" 2>/dev/null || true
echo "[run_k16] Server stopped"

# 9. Build real EC5 pairs (wartau_ms, ttft_ms) via per-adapter WARτ
echo ""
echo "[run_k16] Computing EC5 WARτ/TTFT correlation ..."
python analysis/compute_ec5_correlation.py \
    --metrics  "${LOG_DIR}/gpu0/metrics.jsonl" \
    --harness  "${REPLAY_OUT}" \
    --output   "${TTFT_OUT}" \
    --update-csv "results/instrumentation/k16_correlations_a6000_pcie.csv" \
    --also-update-metric-correlations

# 10. Summary
echo ""
echo "[run_k16] ========== EXPERIMENT COMPLETE =========="
echo "[run_k16] Metrics:  ${LOG_DIR}/gpu0/metrics.jsonl"
echo "[run_k16] Harness:  ${REPLAY_OUT}"
echo "[run_k16] EC5 CSV:  ${TTFT_OUT}"
echo ""
python3 - <<'EOF'
import csv, json
from scipy.stats import spearmanr

# Harness summary
rows = list(csv.DictReader(open("results/instrumentation/k16_harness_results.csv")))
n_ok = sum(1 for r in rows if r.get("success", "") == "True")
print(f"Harness: {len(rows)} requests, {n_ok} successful ({100*n_ok/max(len(rows),1):.1f}%)")
adapter_ids = sorted(set(int(r["adapter_id"]) for r in rows))
print(f"Adapter IDs used: {adapter_ids[:5]}...{adapter_ids[-5:]} (min={min(adapter_ids)}, max={max(adapter_ids)})")

# Metrics summary
batches = [json.loads(l) for l in open("results/instrumentation/k16_logs_a6000_pcie/gpu0/metrics.jsonl") if l.strip()]
wars = [b["war"] for b in batches]
from collections import Counter
war_dist = Counter(round(w, 2) for w in wars)
print(f"Batches: {len(batches)}  WAR distribution: {dict(sorted(war_dist.items()))}")
print(f"WAR range: {min(wars):.3f} – {max(wars):.3f}  mean={sum(wars)/len(wars):.3f}")

# EC5 summary
pairs = list(csv.DictReader(open("results/instrumentation/ttft_per_request.csv")))
wt = [float(r["wartau_ms"]) for r in pairs]
tt = [float(r["ttft_ms"]) for r in pairs]
rho, p = spearmanr(wt, tt)
print(f"EC5: n={len(pairs)}  rho={rho:.4f}  p={p:.3e}  PASS={rho >= 0.5}")
EOF
