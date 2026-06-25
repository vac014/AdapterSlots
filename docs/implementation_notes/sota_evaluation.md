# SOTA Evaluation

Direct comparison against the multi-LoRA serving field under identical workload,
hardware, and metric -- answering "punica already batches adapters, why is AdapterSlots
better?" with measurements rather than internal gate experiments.

Systems compared: vLLM 0.6.3, vLLM V1, SGLang (spec+LoRA), punica, S-LoRA, HF-PEFT,
dLoRA. Each runs its own upstream harness on the same A6000.

- Backends: `backends/backend_{adapterslots,vllm,punica,slora,dlora}.py`
- Drivers: `benchmarks/sota/drivers/`
- Orchestrator: `scripts/run_sota.py`
- Recipes + versions: `benchmarks/sota/REPRODUCE_SOTA.md`, `SOTA_VERSIONS.txt`

Results and the full table: [docs/results.md](../results.md) §3.
