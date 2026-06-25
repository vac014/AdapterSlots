# Baselines for C1 and Table II

We compare AdapterSlots against vLLM 0.5.0, vLLM 0.6.3, S-LoRA, Punica and dLoRA
on one RTX A6000 at K=10, rank 32. All systems replay the same request trace per
seed: 30 warmup requests, then 400 ShareGPT prompts with Zipf-0.9 adapter ids,
128 output tokens each, sent at rate 300 (saturated). The results are in
`results/runs/c1_k10/`, and `bash benchmarks/paper/run_all.sh c1 table2` runs
them. Versions are in [`SOTA_VERSIONS.txt`](SOTA_VERSIONS.txt).

## vLLM 0.6.3

vLLM 0.6.3 is the engine AdapterSlots runs on, so it uses the AdapterSlots
environment and the same engine flags (`--max-num-seqs 256
--num-scheduler-steps 8`). It is arm `C0` of `benchmarks/paper/cell.py`.

## vLLM 0.5.0

```bash
bash envs/setup_vllm050.sh          # creates venvs/vllm050
```

`benchmarks/paper/k10.py` (arm `v050`) launches it from `venvs/vllm050`, or from
`$VLLM050_PYTHON` if set. vLLM 0.5.0 has no `--num-scheduler-steps`, so it runs
with its default engine flags (its default `--max-num-seqs` is 256). Its streaming responses carry no
token usage and can merge several tokens into one chunk, so the driver sends
these requests non-streaming and counts `usage.completion_tokens`.

## S-LoRA

S-LoRA runs in the AdapterSlots environment with its kernels built in place:

```bash
cd deps/slora && pip install -e . --no-build-isolation --no-deps
```

`backends/backend_slora.py` starts its API server with `--swap` and a KV pool of
`SLORA_MAX_TOTAL_TOKENS` (default 50000) tokens, close to what vLLM allocates on
the same card, and sends each request with the adapter path in `lora_dir`.
S-LoRA loads the attention projections of each adapter (q, k, v, o); its kernels
have no MLP LoRA path.

## Punica

Punica also builds in the AdapterSlots environment:

```bash
cd deps/punica && TORCH_CUDA_ARCH_LIST=8.6 pip install -e . --no-build-isolation --no-deps
```

`benchmarks/paper/punica_engine.py` drives Punica's own batching engine in
process, following `deps/punica/benchmarks/bench_textgen_lora.py`: first come
first served, one new request admitted per step, the batch kept sorted by
adapter, batch cap 256. It uses the prompt lengths and adapter ids of the same
trace and random weights, as Punica's own benchmarks do; every request decodes
exactly 128 tokens, so throughput does not depend on the weight values.

## dLoRA

```bash
bash envs/setup_dlora.sh            # creates venvs/dlora and builds deps/dlora
```

`benchmarks/paper/dlora_engine.py` starts dLoRA's API server from
`venvs/dlora` (or `$DLORA_PYTHON`) with exec type LORA, `--num-models 10
--max-r 32 --max-num-seqs 256` and dummy weights, as in dLoRA's artifact
scripts, and sends the trace from the AdapterSlots environment with
`ignore_eos`, so every request decodes 128 tokens. The build needs a CUDA 12
`nvcc` and gcc 12; the script swaps torch's bundled pybind11 headers for 2.13,
which compile under `nvcc` 12.
