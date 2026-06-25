#!/usr/bin/env bash
# dLoRA (deps/dlora, a vLLM 0.1.x fork) in venvs/dlora, built against
# torch 2.1.2+cu121 with a CUDA 12 nvcc and gcc 12. benchmarks/paper/dlora_engine.py
# finds it there (or at $DLORA_PYTHON).
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
V=$REPO/venvs/dlora
python3.10 -m venv $V
$V/bin/pip install -q --upgrade pip
$V/bin/pip install -q torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
$V/bin/pip install -q "numpy<2" "pydantic<2" "fastapi<0.100" transformers==4.33.3 sentencepiece \
    psutil "ray>=2.5.1,<2.10" pandas "pyarrow<15" pulp aiohttp uvicorn ninja packaging \
    pybind11==2.13.6 accelerate==0.24.1 "setuptools<70"
$V/bin/pip install -q xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121 --no-deps
# torch 2.1.2 ships pybind11 2.10, whose cast.h does not compile under nvcc 12;
# put the 2.13 headers in its place.
SP=$($V/bin/python -c "import site; print(site.getsitepackages()[0])")
rm -rf $SP/torch/include/pybind11
cp -r $SP/pybind11/include/pybind11 $SP/torch/include/pybind11
cd deps/dlora
CC=${CC:-gcc-12} CXX=${CXX:-g++-12} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6} MAX_JOBS=6 \
    $V/bin/pip install -e . --no-build-isolation --no-deps
(cd PEFT-Dist && $V/bin/pip install -q -e . --no-deps)
$V/bin/python -c "import vllm.engine.llm_engine; print('dLoRA engine imports')"
