# Reproducing AdapterSlots

Single entry point for reproducing the results in [`docs/results.md`](docs/results.md).
Hardware: A6000 48 GB, llama-7b, fp16.

## 1. Environment

```bash
conda env create -f envs/adapter_env.yml
conda activate adapter_env
pip install -e .                       # installs the adapter_slots package
python -c "import adapter_slots; print(adapter_slots.__version__)"
```

Place the base model at `models/llama-7b` and the rank-32 adapters under `adapters/`.

## 2. AdapterSlots throughput (results.md §1, §2)

The universal harness is `benchmarks/ablations/bench.py`. Mode `C7` is the fused
SGMV AdapterSlots engine; `C0` is vanilla vLLM (the baseline).

```bash
# AdapterSlots, fused SGMV, rank 32, K=32 adapters
python benchmarks/ablations/bench.py --backend adapterslots --mode C7 \
    --model models/llama-7b --adapter-dir adapters --num-adapters 32 --rank 32 \
    --output results/end_to_end_serving/adapterslots_k32.json

# vLLM baseline, same workload
python benchmarks/ablations/bench.py --backend vllm --mode C0 \
    --model models/llama-7b --adapter-dir adapters --num-adapters 32 --rank 32 \
    --output results/end_to_end_serving/vllm_k32.json
```

Sweep `--num-adapters 4 8 16 32` to reproduce the K-scaling table.

## 3. SOTA comparison (results.md §3)

Step-by-step per-system recipes (vLLM, SGLang, vLLM V1, punica, S-LoRA, HF-PEFT,
dLoRA), each in its own venv, are in
[`benchmarks/sota/REPRODUCE_SOTA.md`](benchmarks/sota/REPRODUCE_SOTA.md). Pinned
versions are in [`benchmarks/sota/SOTA_VERSIONS.txt`](benchmarks/sota/SOTA_VERSIONS.txt).

The orchestrator runs the backends that share the AdapterSlots env:

```bash
python scripts/run_sota.py --model models/llama-7b --adapter-dir adapters --rank 32 \
    --output-dir results/sota_evaluation
```

## 4. Full artifact (all experiments)

```bash
GPU=a6000_single bash benchmarks/reproduce/reproduce_all.sh
```

Per-experiment scripts live in `benchmarks/reproduce/` (`reproduce_e4.sh`,
`reproduce_e10.sh`, `reproduce_e11.sh`, `reproduce_ablations.sh`). Outputs land in
`results/<experiment>/`. Figure generation, table generation, and theorem validation
are performed separately from those outputs.

## 5. Layout

| Path | Contents |
|------|----------|
| `adapter_slots/` | the package (kernel, dispatch, control, prefetch, integrations) |
| `benchmarks/` | harness, micro-benchmarks, SOTA recipes, reproduce scripts |
| `backends/` | server wrappers (adapterslots, vllm, punica, slora, dlora) |
| `results/` | measured outputs, named by experiment |
| `docs/results.md` | the results this repo reproduces |
