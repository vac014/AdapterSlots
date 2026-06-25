#!/usr/bin/env bash
# launch_k16_serving.sh -- Start a TP=N vLLM HTTP server for K=16 serving experiment.
#
# Designed for the Two RTX A6000 PCIe setup (2 GPUs, 8 adapters per GPU = K=16 total).
# Also usable for the Two H100 NVLink setup with --n-gpus 2 --K-per-gpu 16 (K=32).
#
# Starts a single vLLM server in tensor-parallel mode. The AlignmentAwareScheduler
# is used so per-batch WAR metrics are written to <log-dir>/gpu0/metrics.jsonl.
#
# Usage (Two A6000 PCIe, K=16):
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash scripts/reproduce/launch_k16_serving.sh \
#       --model ./models/llama-7b \
#       --adapters ./adapters \
#       --n-gpus 2 \
#       --K-per-gpu 8 \
#       --workload workloads/zipf_k4_lam7_n5000.jsonl \
#       --log-dir results/instrumentation/k16_logs_a6000_pcie \
#       --duration 600
#
# Usage (Two H100 NVLink, K=32):
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash scripts/reproduce/launch_k32_serving.sh \
#       --model ./models/llama-7b \
#       --adapters ./adapters \
#       --n-gpus 2 \
#       --K-per-gpu 16 \
#       --workload workloads/zipf_k4_lam7_n5000.jsonl \
#       --log-dir results/instrumentation/k32_logs_h100_nvlink \
#       --duration 600

set -euo pipefail

MODEL=""
ADAPTERS=""
N_GPUS=2
K_PER_GPU=8
WORKLOAD=""
LOG_DIR="results/instrumentation/k16_logs_a6000_pcie"
DURATION=600
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";    shift 2 ;;
        --adapters)    ADAPTERS="$2"; shift 2 ;;
        --n-gpus)      N_GPUS="$2";   shift 2 ;;
        --K-per-gpu)   K_PER_GPU="$2";shift 2 ;;
        --workload)    WORKLOAD="$2"; shift 2 ;;
        --log-dir)     LOG_DIR="$2";  shift 2 ;;
        --duration)    DURATION="$2"; shift 2 ;;
        --port)        PORT="$2";     shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

: "${MODEL:?--model required}"
: "${ADAPTERS:?--adapters required}"

TOTAL_K=$((N_GPUS * K_PER_GPU))
TP_SIZE=${N_GPUS}

# When /dev/shm is too small for NCCL's SHM transport (common in containers
# launched without --shm-size or --ipc=host), disable SHM transport so NCCL
# falls back to CUDA P2P over PCIe -- the correct intra-node path for A6000s.
# Do NOT fall back to TP=1: that silently discards the multi-GPU topology.
if [[ ${N_GPUS} -gt 1 ]]; then
    SHM_AVAIL_KB=$(df -k /dev/shm | awk 'NR==2 {print $4}')
    if [[ ${SHM_AVAIL_KB} -lt 512000 ]]; then
        echo "[launch_k16] WARN: /dev/shm only ${SHM_AVAIL_KB}kB (need ≥512MB for NCCL SHM transport)"
        echo "[launch_k16] INFO: setting NCCL_SHM_DISABLE=1 -- NCCL will use CUDA P2P over PCIe instead"
        echo "[launch_k16] INFO: to permanently fix, restart the container with --shm-size=16g or add shm_size in docker-compose.yml"
        export NCCL_SHM_DISABLE=1
    fi
fi
echo "[launch_k16] N_GPUS=${N_GPUS}  K_PER_GPU=${K_PER_GPU}  TOTAL_K=${TOTAL_K}  TP=${TP_SIZE}"
echo "[launch_k16] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set -- set externally}"
mkdir -p "${LOG_DIR}/gpu0"

# Build --lora-modules args: adapter_{i} → {ADAPTERS}/adapter_r16_k{i}_s{42+i}
# Uses r16 adapters; silently skips indices where the directory does not exist.
LORA_MODULES_ARGS=()
for i in $(seq 0 $((TOTAL_K - 1))); do
    SEED=$((42 + i))
    ADAPTER_PATH="${ADAPTERS}/adapter_r16_k${i}_s${SEED}"
    if [[ -d "${ADAPTER_PATH}" ]]; then
        LORA_MODULES_ARGS+=("adapter_${i}=${ADAPTER_PATH}")
    fi
done
N_ADAPTERS=${#LORA_MODULES_ARGS[@]}
if [[ ${N_ADAPTERS} -eq 0 ]]; then
    echo "[launch_k16] ERROR: no adapters found under ${ADAPTERS}/adapter_r16_k*" >&2
    exit 1
fi
echo "[launch_k16] Registering ${N_ADAPTERS} adapter(s): adapter_0 … adapter_$((N_ADAPTERS - 1))"

# AdapterSlots scheduler configuration -- inherit from environment if already set.
export AS_SCHEDULER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export AS_TMAX_MS="${AS_TMAX_MS:-3000}"
export AS_MODE="${AS_MODE:-threshold}"
export AS_LOG_WAR=1
export AS_METRICS_PATH="${LOG_DIR}/gpu0/metrics.jsonl"

echo "[launch_k16] Starting vLLM server (TP=${TP_SIZE}, max-loras=${TOTAL_K}) on port ${PORT} ..."
python scripts/vllm_serve_adapter_slots.py \
    --model "${MODEL}" \
    --enable-lora \
    --max-loras "${TOTAL_K}" \
    --max-lora-rank 16 \
    --lora-modules "${LORA_MODULES_ARGS[@]}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.88 \
    --port "${PORT}" \
    > "${LOG_DIR}/gpu0/stdout.log" 2> "${LOG_DIR}/gpu0/stderr.log" &
SERVER_PID=$!

# Wait for the vLLM HTTP server to be ready (model load + CUDA graph capture ≈ 60–90 s).
# Poll /health; only emit "Started ... workers" once the server can actually serve.
echo "[launch_k16] Waiting for vLLM server on port ${PORT} ..."
READY=0
for i in $(seq 1 300); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        READY=1
        echo "[launch_k16] Server ready after ${i}s"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[launch_k16] ERROR: vLLM server exited unexpectedly. Check ${LOG_DIR}/gpu0/stderr.log" >&2
        exit 1
    fi
    sleep 1
done

if [[ "${READY}" -eq 0 ]]; then
    echo "[launch_k16] ERROR: Server not ready after 300s. Check ${LOG_DIR}/gpu0/stderr.log" >&2
    kill -9 "${SERVER_PID}" 2>/dev/null || true
    exit 1
fi

# Emit the sentinel that the caller's wait-loop greps for.
echo "[launch_k16] Started 1 workers (PIDs: ${SERVER_PID})"
echo "[launch_k16] Running for ${DURATION}s ..."
sleep "${DURATION}"

echo "[launch_k16] Stopping server ..."
kill -TERM "${SERVER_PID}" 2>/dev/null || true
TRIES=0
while kill -0 "${SERVER_PID}" 2>/dev/null && [[ ${TRIES} -lt 30 ]]; do
    sleep 1
    TRIES=$((TRIES + 1))
done
kill -9 "${SERVER_PID}" 2>/dev/null || true

echo "[launch_k16] All workers stopped. Logs in: ${LOG_DIR}"
echo "[launch_k16] Per-GPU metric files:"
ls "${LOG_DIR}"/gpu*/metrics.jsonl 2>/dev/null || echo "  (no metrics.jsonl found -- check ${LOG_DIR}/gpu0/stderr.log)"
