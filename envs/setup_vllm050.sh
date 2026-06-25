#!/usr/bin/env bash
# vLLM 0.5.0 in venvs/vllm050: the baseline the paper calls "vLLM" for C1 and
# Table II. benchmarks/paper/common.py finds it there (or at $VLLM050_PYTHON).
set -euo pipefail
cd "$(dirname "$0")/.."
V=venvs/vllm050
python3.10 -m venv $V
$V/bin/pip install -q --upgrade pip
$V/bin/pip install -q vllm==0.5.0 transformers==4.41.2 outlines==0.0.43 numpy==1.26.4 \
    huggingface-hub==0.23.4 datasets==2.19.2
# outlines 0.0.43 imports pyairports, whose PyPI release is empty; vLLM never
# uses it, so an empty module satisfies the import.
SP=$($V/bin/python -c "import site; print(site.getsitepackages()[0])")
mkdir -p $SP/pyairports
touch $SP/pyairports/__init__.py
echo "AIRPORT_LIST = []" > $SP/pyairports/airports.py
$V/bin/python -c "import vllm; print('vLLM', vllm.__version__)"
