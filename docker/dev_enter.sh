#!/usr/bin/env bash
# Enter (or create) the persistent local dev container.
# On first run: creates a named container and drops you into bash.
# On subsequent runs: re-attaches to the same container.
#
# One-time alias setup (run once, from the project root):
#   echo "alias ds='bash $(pwd)/docker/dev_enter.sh'" >> ~/.bashrc && source ~/.bashrc
# Then just type: ds
set -e

CONTAINER=adapter_slots-dev
IMAGE=adapter_slots-env:cu124
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    exec docker start -ai "$CONTAINER"
else
    echo "First run — creating container '${CONTAINER}'…"
    exec docker run --gpus all -it --name "$CONTAINER" \
        --ipc=host \
        --shm-size=16g \
        --ulimit memlock=-1:-1 \
        --ulimit stack=-1:-1 \
        --ulimit nofile=1048576:1048576 \
        --ulimit nproc=-1:-1 \
        --cap-add=SYS_ADMIN \
        --cap-add=IPC_LOCK \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v "${PROJECT_DIR}:/workspace" \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e WORKSPACE=/workspace \
        -e HF_HOME=/workspace/models \
        -e TRANSFORMERS_CACHE=/workspace/models \
        -w /workspace \
        "$IMAGE" bash
fi
