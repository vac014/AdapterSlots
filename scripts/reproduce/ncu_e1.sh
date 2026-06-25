#!/usr/bin/env bash
# ncu_e1.sh -- Nsight Compute profiling for E1 isolation experiment.
#
# Usage (single GPU):
#   sudo bash scripts/reproduce/ncu_e1.sh A ./models/llama-7b a6000
#   sudo bash scripts/reproduce/ncu_e1.sh D ./models/llama-7b h100_nvlink
#
# Usage (TP=2, Two A6000 PCIe):
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash scripts/reproduce/ncu_e1.sh A ./models/llama-7b two_a6000_pcie --tp 2
#
# Usage (TP=2, Two H100 NVLink):
#   export CUDA_VISIBLE_DEVICES=0,1
#   bash scripts/reproduce/ncu_e1.sh D ./models/llama-7b h100_nvlink --tp 2
#
# The --tp flag switches the python launch from direct invocation to
# torchrun --nproc_per_node=2, allowing NCU to profile the rank-0 process
# (which runs the SGMV kernels + NCCL all-reduce on cuda:0).
#
# perf_event_paranoid=4 on this system -- run with sudo for hardware counters:
#   sudo -E bash scripts/reproduce/ncu_e1.sh A ./models/llama-7b two_a6000_pcie --tp 2

set -euo pipefail

# Parse arguments (positional + optional --tp flag)

TP=1
SWEEP=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tp)
            TP="${2:?--tp requires a value (e.g. --tp 2)}"
            shift 2
            ;;
        --sweep)
            SWEEP=1
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Sweep mode omits the condition positional (no single condition to name).
# Non-sweep:  <condition> <model_path> <gpu_tag>
# Sweep:                  <model_path> <gpu_tag>
if [[ "${SWEEP}" -eq 1 ]]; then
    CONDITION="ALL"
    MODEL_PATH="${POSITIONAL[0]:-./models/llama-7b}"
    GPU_TAG="${POSITIONAL[1]:-a6000}"
else
    CONDITION="${POSITIONAL[0]:-D}"
    MODEL_PATH="${POSITIONAL[1]:-./models/llama-7b}"
    GPU_TAG="${POSITIONAL[2]:-a6000}"
fi

# Experiment configuration

N=4096
K=4
RANK=16

# Number of timed workload runs inside Python (per condition in sweep mode)
N_RUNS=20

# Nsight launch control
LAUNCH_SKIP=10
# Sweep mode: 4 conditions × N_RUNS × 3 kernels/batch = 240 launches, plus warmup skip
LAUNCH_COUNT=${SWEEP:+240}
LAUNCH_COUNT="${LAUNCH_COUNT:-80}"

# Paths

NCU_BIN=$(which ncu 2>/dev/null || find /usr/local/cuda* -name ncu -type f 2>/dev/null | head -1)
PYTHON_BIN=$(which python3 2>/dev/null || which python)

# Output paths

OUTPUT_DIR="results/e1/ncu/${GPU_TAG}"
mkdir -p "${OUTPUT_DIR}"

if [[ "${SWEEP}" -eq 1 ]]; then
    OUTPUT_FILE="${OUTPUT_DIR}/ncu_sweep_${GPU_TAG}_latest.csv"
    WAR_MAP_FILE="${OUTPUT_DIR}/ncu_sweep_war_map_${GPU_TAG}_latest.csv"
else
    OUTPUT_FILE="${OUTPUT_DIR}/ncu_condition_${CONDITION}_${GPU_TAG}_latest.csv"
fi

# Temporary raw capture file
# Sanitise CONDITION for use in a filename (guard against accidental path chars).
SAFE_COND="$(printf '%s' "${CONDITION}" | tr -cs 'A-Za-z0-9_-' '_')"
TMP_RAW="$(mktemp "${OUTPUT_DIR}/tmp_ncu_${SAFE_COND}_${GPU_TAG}_XXXXXX.csv")"

# Metrics
# Lean metric set to reduce replay overhead

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

# Logging

echo "======================================================"
echo "E1 Nsight Compute Profiling"
echo "======================================================"
echo "Condition      : ${SWEEP:+ALL (sweep)} ${SWEEP:-${CONDITION}}"
echo "Sweep mode     : ${SWEEP}"
echo "Model          : ${MODEL_PATH}"
echo "GPU tag        : ${GPU_TAG}"
echo "TP degree      : ${TP}"
echo "NCU            : ${NCU_BIN}"
echo "Python         : ${PYTHON_BIN}"
echo "Filtered CSV   : ${OUTPUT_FILE}"
echo "Launch skip    : ${LAUNCH_SKIP}"
echo "Launch count   : ${LAUNCH_COUNT}"
echo "N_RUNS         : ${N_RUNS} per condition"
echo "======================================================"
echo ""

# Nsight Compute profiling
# TP=1: profile single process (isolation_batch_conditions.py directly)
# TP=2: use torchrun --nproc_per_node=2 so NCCL/NVLink ops appear in trace.
#        NCU --target-processes all captures all spawned rank processes.
#        The rank-0 process is bound to cuda:0 (first GPU in
#        CUDA_VISIBLE_DEVICES).  Rank-1 runs on cuda:1.

if [[ "${SWEEP}" -eq 1 ]]; then
    # Sweep mode: run all 4 conditions (A B C D) in sequence within one NCU session.
    # isolation_batch_conditions.py --sweep-conditions emits a batch_war_map CSV alongside the NCU CSV
    # so compute_correlations.py can assign per-batch WAR values.
    "${NCU_BIN}" \
      --metrics "${METRICS}" \
      --launch-skip "${LAUNCH_SKIP}" \
      --launch-count "${LAUNCH_COUNT}" \
      --target-processes all \
      --csv \
      --log-file "${TMP_RAW}" \
      "${PYTHON_BIN}" scripts/experiments/isolation_batch_conditions.py \
        --sweep-conditions A,B,C,D \
        --N "${N}" \
        --K "${K}" \
        --rank "${RANK}" \
        --model "${MODEL_PATH}" \
        --n-runs "${N_RUNS}" \
        --war-map-output "${WAR_MAP_FILE}"

elif [[ "${TP}" -gt 1 ]]; then
    # Locate torchrun in the same conda env as python.
    TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/torchrun"
    if [[ ! -x "${TORCHRUN_BIN}" ]]; then
        TORCHRUN_BIN="$(dirname "${PYTHON_BIN}")/python -m torch.distributed.run"
    fi

    echo "[ncu_e1.sh] TP=${TP}: launching via torchrun --nproc_per_node=${TP}"
    echo "[ncu_e1.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}"

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

echo ""
echo "======================================================"
echo "Filtering relevant kernels..."
echo "======================================================"

# Keep:
#   - CSV header
#   - CUTLASS grouped GEMM kernels
#   - FlashInfer routing kernels
#   - precompute helper kernels

grep -E 'Kernel|sgmv_shrink|precompute_sgmv_args|^"ID"' \
  "${TMP_RAW}" > "${OUTPUT_FILE}"

# Remove temporary raw capture
rm -f "${TMP_RAW}"

if [[ "${SWEEP}" -eq 1 ]]; then
    echo "WAR map        -> ${WAR_MAP_FILE}"
fi

echo ""
echo "======================================================"
echo "Done."
echo "======================================================"
echo "Filtered CSV -> ${OUTPUT_FILE}"
echo ""