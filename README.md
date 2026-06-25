# AdapterSlots

Warp-aligned LoRA adapter serving for LLM inference.

AdapterSlots batches decode tokens by adapter into warp-aligned slots and dispatches
them through a fused SGMV kernel, so serving cost stays flat as the number of
concurrently served adapters grows. It ships as a Python package plus a benchmark
harness and the backend wrappers used for comparison against other multi-LoRA systems.

## Install

```bash
conda env create -f envs/adapter_env.yml
conda activate adapter_env
pip install -e .
python -c "import adapter_slots; print(adapter_slots.__version__)"
```

Base model and adapters are expected at `models/llama-7b` and `adapters/`.

## Layout

| Path | Contents |
|------|----------|
| `adapter_slots/` | the package: `kernel/`, `dispatch/`, `control/`, `prefetch/`, `metrics/`, `integrations/` |
| `backends/` | server wrappers (adapterslots, vllm, punica, slora, dlora) |
| `benchmarks/` | universal harness, micro-benchmarks, SOTA recipes, `reproduce/` scripts |
| `workloads/` | trace replay and synthetic request generators |
| `tests/` | unit tests |
| `hw_profiles/` | per-GPU kernel timing profiles |
| `docs/` | design notes and proofs |
| `results/` | measured outputs, named by experiment |

## Reproduce

See [`REPRODUCE.md`](REPRODUCE.md) for the throughput and SOTA-comparison runs, and
[`benchmarks/sota/REPRODUCE_SOTA.md`](benchmarks/sota/REPRODUCE_SOTA.md) for the
per-system recipes.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
