#!/usr/bin/env bash
# run_noloss_check.sh -- Section 4.5b no-loss verification, Single A6000 (EC 9.1.6)
#
# Submits 10 000 requests and verifies 0 are dropped.
# Uses vllm_serve_adapter_slots.py (AS_SCHEDULER=1 env-var approach) because
# vLLM 0.6.x does not support --scheduler-class; same fix applied in §2.7.3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}"

PORT=8000
MODEL="./models/llama-7b"
RESULT_DIR="results/alignment_buffer/a6000_single"
RESULT_FILE="noloss_check_10000.json"
SERVER_LOG="logs/noloss_server.log"

mkdir -p "${RESULT_DIR}" logs

# AdapterSlots scheduler configuration
export AS_SCHEDULER=1
export AS_TMAX_MS=5.0
export AS_TTFT_SLO_MS=200.0
export AS_MODE=threshold
export AS_LOG_WAR=1

echo "[noloss] Starting vLLM server with AlignmentAwareScheduler (AS_SCHEDULER=1) ..."
python scripts/vllm_serve_adapter_slots.py \
    --model "${MODEL}" \
    --enable-lora \
    --max-loras 4 \
    --lora-modules \
        "adapter_0=./adapters/adapter_r16_k0_s42" \
        "adapter_1=./adapters/adapter_r16_k1_s43" \
        "adapter_2=./adapters/adapter_r16_k2_s44" \
        "adapter_3=./adapters/adapter_r16_k3_s45" \
    --max-lora-rank 16 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.88 \
    --port "${PORT}" \
    --disable-frontend-multiprocessing \
    --served-model-name "llama-7b" \
    > "${SERVER_LOG}" 2>&1 &
VLLM_PID=$!
echo "[noloss] Server PID=${VLLM_PID}  log: ${SERVER_LOG}"

# Wait for server ready
echo "[noloss] Waiting for server on port ${PORT} ..."
SERVER_READY=0
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[noloss] Server ready after ${i}s"
        SERVER_READY=1
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[noloss] ERROR: server died. Last 30 lines of ${SERVER_LOG}:" >&2
        tail -30 "${SERVER_LOG}" >&2
        exit 1
    fi
    sleep 1
done
if [[ "${SERVER_READY}" -eq 0 ]]; then
    echo "[noloss] ERROR: server not ready after 120s. Check ${SERVER_LOG}" >&2
    kill -9 "${VLLM_PID}" 2>/dev/null || true
    exit 1
fi

# Run benchmark
echo "[noloss] Running 10 000-request benchmark at 7 req/s ..."
cd benchmarks/vllm_upstream
python benchmark_serving.py \
    --backend openai \
    --model llama-7b \
    --tokenizer "../../${MODEL}" \
    --dataset-name sharegpt \
    --dataset-path ../../data/sharegpt/sharegpt.jsonl \
    --request-rate 7 \
    --num-prompts 10000 \
    --save-result \
    --result-dir "../../${RESULT_DIR}" \
    --result-filename "${RESULT_FILE}"
cd "${ROOT}"

# Verify no-loss
python3 -c "
import json, sys
d = json.load(open('${RESULT_DIR}/${RESULT_FILE}'))
completed = d.get('completed', d.get('num_prompts', '?'))
print(f'Submitted: 10000  Completed: {completed}')
if int(completed) == 10000:
    print('EC 9.1.6: PASS -- no requests dropped')
else:
    print(f'EC 9.1.6: FAIL -- {10000 - int(completed)} requests dropped', file=sys.stderr)
    sys.exit(1)
"

# Cleanup
echo "[noloss] Stopping server ..."
kill "${VLLM_PID}" 2>/dev/null || true
sleep 3
kill -9 "${VLLM_PID}" 2>/dev/null || true
echo "[noloss] Done."
