#!/usr/bin/env bash
# nsys_e1.sh -- Nsight Systems timeline profiling for E1 isolation experiment.
#
# Usage:
#   bash scripts/reproduce/nsys_e1.sh A ./models/llama-3-8b a6000
#   bash scripts/reproduce/nsys_e1.sh D ./models/llama-3-8b h100
#
# Output: results/infrastructure/nsys/nsys_condition_<COND>_<GPU>_<TS>.nsys-rep
# Open the .nsys-rep in the Nsight Systems GUI to view the timeline.
#
# Key things to look for in the timeline:
#   - SGMV kernel launch boundaries (cuda kernel rows)
#   - CTA assignment sequence visible in nvtx markers
#   - Gap between CPU dispatch and GPU kernel start
#   - Memory transfer events (cudaMemcpyAsync)

# nsys_e1.sh -- Nsight Systems timeline profiling for E1 isolation experiment.
#
# Usage:
#   sudo bash scripts/reproduce/nsys_e1.sh A ./models/llama-3-8b a6000
#   sudo bash scripts/reproduce/nsys_e1.sh D ./models/llama-3-8b h100
#
# Output:
#   results/e1/nsys/<gpu_tag>/
# nsys_condition_<COND>_<GPU>_<TS>.nsys-rep
# nsys_condition_<COND>_<GPU>_<TS>.sqlite
# nsys_condition_<COND>_<GPU>_<TS>_filtered_kernels.txt
#
# Open report:
#   nsys-ui <report>.nsys-rep

set -euo pipefail

# Arguments

CONDITION="${1:-A}"
MODEL_PATH="${2:-./models/llama-3-8b}"
GPU_TAG="${3:-a6000}"

# Experiment configuration

N=4096
K=4
RANK=16
N_RUNS=20

# Paths

NSYS_BIN=$(find / -name nsys 2>/dev/null | head -n 1)
PYTHON_BIN=$(which python3 2>/dev/null || which python)

# Output paths

OUTPUT_DIR="results/e1/nsys/${GPU_TAG}"
mkdir -p "${OUTPUT_DIR}"

OUTPUT_PREFIX="${OUTPUT_DIR}/nsys_condition_${CONDITION}_${GPU_TAG}_latest"

FILTERED_KERNELS="${OUTPUT_DIR}/nsys_condition_${CONDITION}_${GPU_TAG}_filtered_kernels_latest.txt"

TMP_STATS="$(mktemp "${OUTPUT_DIR}/tmp_nsys_${CONDITION}_${GPU_TAG}_XXXXXX.txt")"

# Logging

echo "======================================================"
echo "E1 Nsight Systems Profiling"
echo "======================================================"
echo "Condition      : ${CONDITION}"
echo "Model          : ${MODEL_PATH}"
echo "GPU tag        : ${GPU_TAG}"
echo "NSYS           : ${NSYS_BIN}"
echo "Python         : ${PYTHON_BIN}"
echo "Output report  : ${OUTPUT_PREFIX}.nsys-rep"
echo "SQLite output  : ${OUTPUT_PREFIX}.sqlite"
echo "Filtered list  : ${FILTERED_KERNELS}"
echo "N_RUNS         : ${N_RUNS}"
echo "======================================================"
echo ""

# Nsight Systems profiling

"${NSYS_BIN}" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  --output "${OUTPUT_PREFIX}" \
  "${PYTHON_BIN}" scripts/experiments/isolation_batch_conditions.py \
    --condition "${CONDITION}" \
    --N "${N}" \
    --K "${K}" \
    --rank "${RANK}" \
    --model "${MODEL_PATH}" \
    --n-runs "${N_RUNS}"

echo ""
echo "======================================================"
echo "Generating kernel statistics..."
echo "======================================================"

# Extract kernel summary from report

"${NSYS_BIN}" stats \
  --report cuda_gpu_kern_sum \
  --force-export=true \
  "${OUTPUT_PREFIX}.nsys-rep" \
  > "${TMP_STATS}"

# Filter only relevant kernels

grep -E 'Kernel|sgmv_shrink|precompute_sgmv_args' \
  "${TMP_STATS}" > "${FILTERED_KERNELS}"

# Cleanup temporary file
rm -f "${TMP_STATS}"

echo ""
echo "======================================================"
echo "Done."
echo "======================================================"
echo "Timeline report  -> ${OUTPUT_PREFIX}.nsys-rep"
echo "SQLite database  -> ${OUTPUT_PREFIX}.sqlite"
echo "Filtered kernels -> ${FILTERED_KERNELS}"
echo ""
echo "Open timeline with:"
echo "  nsys-ui ${OUTPUT_PREFIX}.nsys-rep"
echo ""