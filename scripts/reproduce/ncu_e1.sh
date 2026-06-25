#!/usr/bin/env bash
# ncu_e1.sh -- Nsight Compute profile of SGMV for one E1 batch layout.
#
# Usage:
#   bash scripts/reproduce/ncu_e1.sh <A|B|S|C|D> ./models/llama-7b a6000
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/reproduce/ncu_e1.sh A ./models/llama-7b two_a6000_pcie --tp 2
#
# With --tp the workload runs under torchrun and NCU profiles every rank. Hardware
# counters need permission to read them (perf_event_paranoid); run with sudo -E
# if NCU reports ERR_NVGPUCTRPERM.

set -euo pipefail

TP=1
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp)
            TP="${2:?--tp requires a value (e.g. --tp 2)}"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

CONDITION="${POSITIONAL[0]:-D}"
MODEL_PATH="${POSITIONAL[1]:-./models/llama-7b}"
GPU_TAG="${POSITIONAL[2]:-a6000}"

# Workload: N tokens over K adapters at LoRA rank RANK, N_RUNS timed batches.
N=4096
K=4
RANK=16
N_RUNS=20

# Profile 80 kernel launches after skipping the first 10.
LAUNCH_SKIP=10
LAUNCH_COUNT=80

NCU_BIN=$(which ncu 2>/dev/null || find /usr/local/cuda* -name ncu -type f 2>/dev/null | head -1)
PYTHON_BIN=$(which python3 2>/dev/null || which python)

OUTPUT_DIR="results/e1/ncu/${GPU_TAG}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/ncu_condition_${CONDITION}_${GPU_TAG}_latest.csv"
SAFE_COND="$(printf '%s' "${CONDITION}" | tr -cs 'A-Za-z0-9_-' '_')"
TMP_RAW="$(mktemp "${OUTPUT_DIR}/tmp_ncu_${SAFE_COND}_${GPU_TAG}_XXXXXX.csv")"

METRICS="\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__warps_eligible.avg.pct_of_peak_sustained_active,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l2tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
smsp__inst_executed_pipe_tensor.sum,\
launch__grid_size,\
launch__block_size"

echo "E1 Nsight Compute profile: condition=${CONDITION} model=${MODEL_PATH} gpu=${GPU_TAG} tp=${TP}"
echo "output: ${OUTPUT_FILE}"

if [[ "${TP}" -gt 1 ]]; then
    "${NCU_BIN}" \
      --metrics "${METRICS}" \
      --launch-skip "${LAUNCH_SKIP}" \
      --launch-count "${LAUNCH_COUNT}" \
      --target-processes all \
      --csv \
      --log-file "${TMP_RAW}" \
      "${PYTHON_BIN}" -m torch.distributed.run \
        --nproc_per_node "${TP}" \
        --master_port 29501 \
        scripts/experiments/isolation_batch_conditions.py \
          --condition "${CONDITION}" \
          --N "${N}" \
          --K "${K}" \
          --rank "${RANK}" \
          --model "${MODEL_PATH}" \
          --tp "${TP}" \
          --n-runs "${N_RUNS}"
else
    "${NCU_BIN}" \
      --metrics "${METRICS}" \
      --launch-skip "${LAUNCH_SKIP}" \
      --launch-count "${LAUNCH_COUNT}" \
      --target-processes all \
      --csv \
      --log-file "${TMP_RAW}" \
      "${PYTHON_BIN}" scripts/experiments/isolation_batch_conditions.py \
        --condition "${CONDITION}" \
        --N "${N}" \
        --K "${K}" \
        --rank "${RANK}" \
        --model "${MODEL_PATH}" \
        --n-runs "${N_RUNS}"
fi

# Keep the CSV header and the SGMV kernels.
grep -E 'Kernel|sgmv_shrink|precompute_sgmv_args|^"ID"' "${TMP_RAW}" > "${OUTPUT_FILE}"
rm -f "${TMP_RAW}"
echo "wrote ${OUTPUT_FILE}"
