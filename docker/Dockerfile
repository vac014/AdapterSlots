# AdapterSlots environment: CUDA 12.4, Python 3.10, the pinned packages of
# requirements.txt, and the vendored Punica and S-LoRA baselines built for the
# RTX A6000 (sm_86). The repository is bind-mounted at /workspace and installed
# editable when the container starts (entrypoint.sh).
#
#   docker build -t adapter_slots-env:cu124 -f docker/Dockerfile .
#   docker compose -f docker/docker-compose.yml run --rm adapter_slots bash
#
# vLLM 0.5.0 and dLoRA use their own environments (envs/setup_vllm050.sh,
# envs/setup_dlora.sh).

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python3.10-venv \
        git cmake ninja-build build-essential \
        curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    pip install --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Punica (CUDA kernels, about 20 to 30 minutes) and S-LoRA from deps/.
COPY deps/punica /opt/deps/punica
COPY deps/slora /opt/deps/slora
ENV TORCH_CUDA_ARCH_LIST="8.6"
RUN cd /opt/deps/punica && pip install --no-cache-dir --no-build-isolation --no-deps -e .
RUN cd /opt/deps/slora && pip install --no-cache-dir --no-build-isolation --no-deps -e .

WORKDIR /workspace
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
