# Reproducing the SOTA comparison

Results and analysis: [`docs/results.md`](../../docs/results.md) §3.
Pinned versions: [`SOTA_VERSIONS.txt`](./SOTA_VERSIONS.txt).

Hardware: A6000 48 GB, `CUDA_VISIBLE_DEVICES=1`, llama-7b at `models/llama-7b`.
Each system runs in its own venv (period-correct deps).

## 1. vLLM 0.6.3 -- primary baseline

Same env as AdapterSlots. vLLM runs vanilla (`AS_SCHEDULER=0`, C0) in the same
session as the AdapterSlots run to cancel GPU-clock and fragmentation confounds.

```bash
python benchmarks/ablations/bench.py --backend vllm --mode C0 \
    --model models/llama-7b --output results/sota_evaluation/vllm_0_6_3.json
```

## 2. SGLang 0.5.14 -- multi-LoRA + speculative decoding

```bash
python -m venv ~/sglang_venv && source ~/sglang_venv/bin/activate
pip install "sglang[all]"

# Server: 14 rank-32 adapters, all default features on (RadixCache, csgmv).
python -m sglang.launch_server --model-path models/llama-7b \
    --lora-paths lora0=<a0> lora1=<a1> ... lora13=<a13> \
    --max-loras-per-batch 8 --max-running-requests 16 --cuda-graph-max-bs 16 \
    --mem-fraction-static 0.88
# Spec run also passes:
#   --speculative-algorithm NGRAM --speculative-num-draft-tokens 6 --speculative-num-steps 5

python -m sglang.bench_serving --backend sglang --dataset-name random \
    --random-input 256 --random-output 256 --request-rate 16 \
    --lora-request-distribution skewed --lora-zipf-alpha 1.1
```
no-spec 435 decode / 869 total; +ngram spec 638 decode / 1274 total (accept len 2.43).

## 3. vLLM V1 0.24.0 -- spec + LoRA

```bash
python -m venv ~/vllmv1_venv && source ~/vllmv1_venv/bin/activate
pip install vllm

VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
vllm serve models/llama-7b --enable-lora --max-loras 4 --max-lora-rank 32 \
    --lora-modules lora0=<a0> ... lora13=<a13> \
    --gpu-memory-utilization 0.50 --port 30000
# Spec run also passes: --speculative-config '{"method":"ngram","num_speculative_tokens":5}'

python benchmarks/sota/drivers/vllm_v1_throughput_driver.py 14 256
```
no-spec 254 decode / 507 e2e; +ngram spec 333 decode / 666 e2e.
`VLLM_USE_FLASHINFER_SAMPLER=0` is required (the flashinfer sampler's JIT CUB build
fails on nvcc 12).

## 4. punica -- SGMV/BGMV decode microbench

```bash
cd deps/punica && PYTHONPATH=src python bench/bench_textgen_lora.py \
    --rank 32 --maxlen 256 --lora-popularity uniform
```
B16 471 e2e / 607 decode; B32 809 e2e / 1110 decode.

## 5. S-LoRA and HF-PEFT -- server + Poisson trace

```bash
# S-LoRA (adapter_venv, deps/slora built)
cd deps/slora/benchmarks
LD_LIBRARY_PATH=<adapter_venv>/torch/lib PYTHONPATH=../.. \
  python launch_server.py --device debug --backend slora \
  --model-setting S1 --num-adapter 20 --dummy
python run_exp.py --suite debug --debug --mode synthetic       # 2.33 req/s

# HF-PEFT (period-correct peft_venv)
python -m venv ~/peft_venv && source ~/peft_venv/bin/activate
pip install torch==2.1.2 transformers==4.31.0 peft==0.4.0 accelerate==0.21.0
python run_exp_peft.py --suite debug --debug --mode synthetic  # 0.28 req/s
```

## 6. dLoRA (OSDI'24)

Toolchain (the build blocker is nvcc 12 vs pybind11):
```bash
python -m venv ~/dlora_venv && source ~/dlora_venv/bin/activate
pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install pybind11==2.13.6
cp -r $(python -c 'import pybind11,os;print(os.path.dirname(pybind11.get_include())+"/pybind11")') \
      <torch>/include/pybind11        # overlay 2.13.6 into torch headers
CC=gcc-12 CXX=g++-12 TORCH_CUDA_ARCH_LIST=8.6 python setup.py build_ext
pip install -e . --no-build-isolation
# setup.py is unpinned; reinstall pinned --no-deps so pip does not break the .so:
pip install --no-deps torch==2.1.2+cu121 xformers==0.0.23.post1 triton==2.1.0 \
  transformers==4.31.0 tokenizers==0.13.3 ray==2.9.3 setuptools==65.5.0 pyarrow==12.0.1
# + install the bundled PEFT-Dist fork (provides create_lora_model)

# Flagship exec_type=3 server:
python -m dlora.entrypoints.api_server --engine-use-ray --worker-use-ray \
  --exec-type 3 --num-models 32 --max-r 32 --gpu-memory-utilization 0.50 \
  --max-num-seqs 12 --port 8200
python benchmarks/sota/drivers/dlora_throughput_driver.py   # 144 decode / 432 e2e
```
