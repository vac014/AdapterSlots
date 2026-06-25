# AdapterSlots environment image
# CUDA 12.4 + Python 3.10 + all pip deps pre-baked.
# adapter_slots is installed editable from /workspace at runtime (see entrypoint.sh).
# punica CUDA kernels are compiled during this build (~20-30 min).
#
# Build:
#   docker build -t adapter_slots-env:cu124 -f docker/Dockerfile .
# Push:
#   docker tag adapter_slots-env:cu124 <registry>/adapter_slots-env:cu124
#   docker push <registry>/adapter_slots-env:cu124

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python3.10-venv \
        git cmake ninja-build build-essential \
        libssl-dev libffi-dev \
        curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    pip install --upgrade pip setuptools wheel

# ── Stage 1: PyTorch (CUDA 12.4) ─────────────────────────────────────────────
RUN pip install --no-cache-dir \
        torch==2.4.0+cu124 \
        torchvision==0.19.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# torchaudio may decouple versioning; fall back to matching torch version if needed
RUN pip install --no-cache-dir torchaudio==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
    || pip install --no-cache-dir torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu124

# ── Stage 2: flash-attention (prebuilt wheel, cu128+torch2.10 ABI) ────────────
RUN pip install --no-cache-dir \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp310-cp310-linux_x86_64.whl"

# ── Stage 3: xformers + flashinfer (CUDA-aware, install after torch) ──────────
RUN pip install --no-cache-dir xformers==0.0.27.post2 flashinfer-python==0.6.11

# ── Stage 4: vLLM + ray (bring in most inference-stack deps) ─────────────────
RUN pip install --no-cache-dir vllm==0.6.3.post1 ray==2.55.1

# ── Stage 5: S-LoRA (CUDA bgmv kernel compiled at build time) ─────────────────
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0" pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/S-LoRA/S-LoRA.git@c1ddf488781ea7f551cd0bb07bfd097124c93411

# ── Stage 6: Punica (CUDA kernel compilation — slow ~20-30 min) ───────────────
# Clone at the pinned commit, patch the bundled flashinfer mma.cuh to fix a
# PTX inline-asm bug (ldmatrix output operands need "=r", not "r"), then build.
ENV PUNICA_PAGE_SIZES=16
ENV PUNICA_GROUP_SIZES=1,2,4,8
ENV PUNICA_HEAD_DIMS=128
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0"
RUN git clone --filter=blob:none https://github.com/punica-ai/punica.git /tmp/punica-src \
    && cd /tmp/punica-src \
    && git checkout 591b59899f0a20760821785d06b331c8a2e5cb86 \
    && git submodule update --init --recursive \
    && sed -i 's/"r"(smem_int_ptr), "r"(R\[0\]), "r"(R\[1\]), "r"(R\[2\]), "r"(R\[3\])/"=r"(R[0]), "=r"(R[1]), "=r"(R[2]), "=r"(R[3]) : "r"(smem_int_ptr)/g' third_party/flashinfer/include/flashinfer/mma.cuh \
    && pip install --no-cache-dir --no-build-isolation .

# ── Stage 7: Remaining packages ───────────────────────────────────────────────
COPY docker/requirements_docker.txt /tmp/requirements_docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements_docker.txt

# ── Runtime setup ─────────────────────────────────────────────────────────────
WORKDIR /workspace

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
