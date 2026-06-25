#!/usr/bin/env bash
# launch_k32_serving.sh -- Start N single-GPU vLLM workers for K=32 serving experiment.
#
# Designed for the Two H100 NVLink setup (2 GPUs, 16 adapters per GPU = K=32 total).
# NVLink 4.0 (~900 GB/s) handles all-reduce across both H100 80 GB GPUs.
#
# Each GPU hosts K_PER_GPU adapters.  Workers run in background; the script waits
# for DURATION seconds, then sends SIGTERM to all workers and exits.
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
K_PER_GPU=16
WORKLOAD=""
LOG_DIR="results/instrumentation/k32_logs_h100_nvlink"
DURATION=600

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";    shift 2 ;;
        --adapters)    ADAPTERS="$2"; shift 2 ;;
        --n-gpus)      N_GPUS="$2";   shift 2 ;;
        --K-per-gpu)   K_PER_GPU="$2";shift 2 ;;
        --workload)    WORKLOAD="$2"; shift 2 ;;
        --log-dir)     LOG_DIR="$2";  shift 2 ;;
        --duration)    DURATION="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

: "${MODEL:?--model required}"
: "${ADAPTERS:?--adapters required}"
: "${WORKLOAD:?--workload required}"

TOTAL_K=$((N_GPUS * K_PER_GPU))
echo "[launch_k32] N_GPUS=${N_GPUS}  K_PER_GPU=${K_PER_GPU}  TOTAL_K=${TOTAL_K}"
echo "[launch_k32] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set -- set externally}"
mkdir -p "${LOG_DIR}"

PIDS=()

for GPU_ID in $(seq 0 $((N_GPUS - 1))); do
    GPU_LOG="${LOG_DIR}/gpu${GPU_ID}"
    mkdir -p "${GPU_LOG}"

    # Adapter IDs assigned to this GPU: [GPU_ID*K_PER_GPU, ..., GPU_ID*K_PER_GPU + K_PER_GPU - 1]
    ADAPTER_START=$((GPU_ID * K_PER_GPU))
    ADAPTER_IDS=$(seq -s, ${ADAPTER_START} $((ADAPTER_START + K_PER_GPU - 1)))

    echo "[launch_k32] GPU ${GPU_ID} → adapters ${ADAPTER_IDS}  log: ${GPU_LOG}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python benchmarks/sota/serving_full.py \
        --systems adapter_slots \
        --model "${MODEL}" \
        --adapters "${ADAPTERS}" \
        --adapter-ids "${ADAPTER_IDS}" \
        --workloads "${WORKLOAD}" \
        --log-war \
        --output-dir "${GPU_LOG}" \
        > "${GPU_LOG}/stdout.log" 2> "${GPU_LOG}/stderr.log" &

    PIDS+=($!)
done

echo "[launch_k32] Started ${#PIDS[@]} workers (PIDs: ${PIDS[*]})"
echo "[launch_k32] Running for ${DURATION}s ..."
sleep "${DURATION}"

echo "[launch_k32] Stopping workers ..."
for PID in "${PIDS[@]}"; do
    kill -TERM "${PID}" 2>/dev/null || true
done

# Wait up to 30s for graceful shutdown.
for PID in "${PIDS[@]}"; do
    TRIES=0
    while kill -0 "${PID}" 2>/dev/null && [[ ${TRIES} -lt 30 ]]; do
        sleep 1
        TRIES=$((TRIES + 1))
    done
    kill -9 "${PID}" 2>/dev/null || true
done

echo "[launch_k32] All workers stopped. Logs in: ${LOG_DIR}"
echo "[launch_k32] Per-GPU metric files:"
ls "${LOG_DIR}"/gpu*/metrics.jsonl 2>/dev/null || echo "  (no metrics.jsonl found -- check stdout/stderr logs)"
