#!/usr/bin/env bash
# e1_tp.sh -- E1 isolation experiment with TP=2 on Two RTX A6000 PCIe or Two H100 NVLink.
#
# Runs condition A or D with TP=2 to test whether adapter-mixing effects persist
# under tensor-parallel weight sharding.  Only TP=2 is supported because both
# available multi-GPU setups have exactly 2 GPUs.
#
# Usage:
#   bash scripts/reproduce/e1_tp.sh <CONDITION> <MODEL_PATH> <GPU_NAME> [N_RUNS]
#
# GPU_NAME is used as the output directory suffix and must be one of:
#   two_a6000_pcie   -- Two RTX A6000 PCIe (CUDA_VISIBLE_DEVICES=0,1 must be set externally)
#   h100_nvlink      -- Two H100 NVLink    (CUDA_VISIBLE_DEVICES=0,1 must be set externally)
#
# Examples:
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash scripts/reproduce/e1_tp.sh A ./models/llama-7b  two_a6000_pcie
#   bash scripts/reproduce/e1_tp.sh D ./models/llama-70b h100_nvlink
#
# Outputs: results/e1/tp2_<GPU_NAME>/throughput_condition_<COND>_tp2_<GPU_NAME>_<TS>.csv

set -euo pipefail

COND=${1:?Usage: e1_tp.sh <A|D> <model_path> <gpu_name> [n_runs]}
MODEL=${2:?}
GPU_NAME=${3:?}
N_RUNS=${4:-200}
TP=2  # Only TP=2 supported: both setups have exactly 2 GPUs.

if [[ "${COND}" != "A" && "${COND}" != "D" ]]; then
    echo "Error: CONDITION must be A or D for tensor-parallel runs." >&2
    exit 1
fi

if [[ "${GPU_NAME}" != "two_a6000_pcie" && "${GPU_NAME}" != "h100_nvlink" ]]; then
    echo "Error: GPU_NAME must be 'two_a6000_pcie' or 'h100_nvlink'." >&2
    exit 1
fi

OUTDIR="results/e1/tp2_${GPU_NAME}"
mkdir -p "${OUTDIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="${OUTDIR}/throughput_condition_${COND}_tp2_${GPU_NAME}_${TIMESTAMP}.csv"

echo "[e1_tp.sh] COND=${COND}  TP=2  GPU=${GPU_NAME}  MODEL=${MODEL}  N_RUNS=${N_RUNS}"
echo "[e1_tp.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set -- both GPUs must be visible}"
echo "[e1_tp.sh] Output: ${OUTFILE}"

python benchmarks/isolation/benchmark_e1.py \
    --model "${MODEL}" \
    --adapter-dir ./adapters \
    --condition "${COND}" \
    --n-tokens 512 \
    --K 2 \
    --n-runs "${N_RUNS}" \
    --warmup 20 \
    --tensor-parallel-size "${TP}" \
    --output "${OUTFILE}"

echo "[e1_tp.sh] Done: ${OUTFILE}"
