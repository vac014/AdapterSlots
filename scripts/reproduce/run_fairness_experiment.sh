#!/usr/bin/env bash
# run_fairness_experiment.sh -- GPU-validated fairness experiment (§5.B.4 / EC 11.2.4)
#
# Runs two Erlang conditions back-to-back and compares per-adapter WARτ distributions:
#
#   NoFair: AS_MODE=erlang, SLO cap = ∞ (1 000 000 ms)
#           Rare adapters starve: T_max*(15) ≈ 4.7 s  >> 2 s SLO
#
#   Fair:   AS_MODE=erlang, SLO cap = 2 000 ms
#           All adapters bounded: T_max*(k) = min(T_erlang*, 2 s) ≤ 2 s
#
# Workload: Zipf-α=1.5, K=16, arrival-rate=7 req/s, output-len=150 tok
#   → token load ≈ 1 050 tok/s (~1.2× A6000 PCIe TP=2 capacity -- light overload
#     intentional so dominant adapters saturate warps while rare adapters starve)
#
# Metrics: per-adapter WARτ (dispatch_time - arrival_time) from metrics.jsonl
#   → independent of TTFT measurement issues (AlignmentBuffer acts on decode,
#     not prefill; WARτ is the native metric for dispatch-alignment starvation)
#
# Wall-clock: ~35 min (two conditions × ~15 min each incl. server startup + drain)
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash scripts/reproduce/run_fairness_experiment.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}"

# Configuration
K=16
N_GPUS=2
PORT=8000
MODEL="./models/llama-7b"
ADAPTERS="./adapters"
WORKLOAD="workloads/fairness_k16_zipf1.5_n3000.jsonl"
LOG_DIR="results/erlang_scheduler/two_a6000_pcie/fairness_logs"
RESULT_DIR="results/erlang_scheduler/two_a6000_pcie"
WAR_TARGET=0.8
TTFT_SLO_MS=2000
EWMA_ALPHA=0.1
ARRIVAL_RATE=7
OUTPUT_LEN=150
N_REQUESTS=3000
DRAIN_S=45   # seconds to drain AlignmentBuffer after harness finishes

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 0. Workload
if [[ ! -f "${WORKLOAD}" ]]; then
    echo "[fairness] Generating Zipf-1.5 K=${K} workload ..."
    python scripts/generate_workload.py \
        --pattern zipf \
        --K "${K}" \
        --zipf-alpha 1.5 \
        --arrival-rate "${ARRIVAL_RATE}" \
        --n-requests "${N_REQUESTS}" \
        --output-len "${OUTPUT_LEN}" \
        --seed 42 \
        --output "${WORKLOAD}"
fi
echo "[fairness] Workload: ${WORKLOAD}"

# 1. Build lora-modules args
LORA_ARGS=()
for i in $(seq 0 $((K - 1))); do
    SEED=$((42 + i))
    AP="${ADAPTERS}/adapter_r16_k${i}_s${SEED}"
    if [[ ! -d "${AP}" ]]; then
        echo "[fairness] ERROR: adapter ${AP} not found -- run step 0.3 first" >&2
        exit 1
    fi
    LORA_ARGS+=("adapter_${i}=${AP}")
done
echo "[fairness] Registered ${#LORA_ARGS[@]} adapters"

# 2. run_condition function
run_condition() {
    local COND="$1"       # "nofair" or "fair"
    local SLO_CAP="$2"    # effective AS_TTFT_SLO_MS value

    local COND_LOG="${LOG_DIR}/${COND}"
    mkdir -p "${COND_LOG}/gpu0"

    echo ""
    echo "[fairness] ════════════════════════════════════════"
    echo "[fairness] Condition : ${COND}  (SLO cap = ${SLO_CAP} ms)"
    echo "[fairness] Log dir   : ${COND_LOG}"
    echo "[fairness] ════════════════════════════════════════"

    # Scheduler environment
    export AS_SCHEDULER=1
    export AS_MODE=erlang
    export AS_WAR_TARGET="${WAR_TARGET}"
    export AS_EWMA_ALPHA="${EWMA_ALPHA}"
    export AS_TTFT_SLO_MS="${SLO_CAP}"
    export AS_LOG_WAR=1
    export AS_METRICS_PATH="${COND_LOG}/gpu0/metrics.jsonl"

    # Start vLLM server
    echo "[fairness] Starting vLLM server (TP=${N_GPUS}, K=${K}) ..."
    python scripts/vllm_serve_adapter_slots.py \
        --model "${MODEL}" \
        --enable-lora \
        --max-loras "${K}" \
        --max-lora-rank 16 \
        --lora-modules "${LORA_ARGS[@]}" \
        --tensor-parallel-size "${N_GPUS}" \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.88 \
        --port "${PORT}" \
        > "${COND_LOG}/gpu0/stdout.log" 2> "${COND_LOG}/gpu0/stderr.log" &
    local SERVER_PID=$!
    echo "[fairness] Server PID=${SERVER_PID}"

    # Wait for /health
    local READY=0
    for i in $(seq 1 300); do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "[fairness] Server ready after ${i}s"
            READY=1
            break
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[fairness] ERROR: server died -- check ${COND_LOG}/gpu0/stderr.log" >&2
            tail -20 "${COND_LOG}/gpu0/stderr.log" >&2
            exit 1
        fi
        sleep 1
    done
    if [[ "${READY}" -eq 0 ]]; then
        echo "[fairness] ERROR: server not ready after 300s" >&2
        kill -9 "${SERVER_PID}" 2>/dev/null || true
        exit 1
    fi

    # Replay Zipf-1.5 workload (real-time, speed-mult=1.0)
    echo "[fairness] Replaying ${N_REQUESTS} requests at ${ARRIVAL_RATE} req/s ..."
    python scripts/replay_harness.py \
        --trace "${WORKLOAD}" \
        --endpoint "http://localhost:${PORT}/v1/completions" \
        --speed-multiplier 1.0 \
        --model "llama-7b" \
        --adapter-prefix "adapter_" \
        --warp-size 32 \
        --output "${COND_LOG}/harness.csv" \
        --timeout-s 300.0 \
        --max-concurrent 150 \
        2>&1 | tee "${COND_LOG}/harness.log"

    # Drain: allow AlignmentBuffer to flush remaining tokens before stopping
    echo "[fairness] Draining AlignmentBuffer (${DRAIN_S}s) ..."
    sleep "${DRAIN_S}"

    # Stop server
    echo "[fairness] Stopping server ..."
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    local TRIES=0
    while kill -0 "${SERVER_PID}" 2>/dev/null && [[ ${TRIES} -lt 30 ]]; do
        sleep 1; TRIES=$((TRIES + 1))
    done
    kill -9 "${SERVER_PID}" 2>/dev/null || true
    echo "[fairness] ${COND} done. Batches: $(wc -l < "${COND_LOG}/gpu0/metrics.jsonl" 2>/dev/null || echo 0)"

    # Brief pause between conditions so GPU memory is fully released
    sleep 15
}

# 3. Create output directories
mkdir -p "${LOG_DIR}/nofair/gpu0" "${LOG_DIR}/fair/gpu0" "${RESULT_DIR}"

# 4. Run both conditions
run_condition "nofair" 1000000        # Erlang uncapped  → starvation
run_condition "fair"   "${TTFT_SLO_MS}"  # Erlang capped at 2 000 ms → fairness

# 5. Fairness metrics (per-adapter TTFT, starvation, SLO compliance) are computed
#    offline from the per-condition logs written under ${LOG_DIR}.
echo ""
echo "[fairness] COMPLETE"
echo "[fairness] nofair logs : ${LOG_DIR}/nofair/"
echo "[fairness] fair logs   : ${LOG_DIR}/fair/"
echo "[fairness] NoFair metrics  : ${LOG_DIR}/nofair/gpu0/metrics.jsonl"
echo "[fairness] Fair metrics    : ${LOG_DIR}/fair/gpu0/metrics.jsonl"
