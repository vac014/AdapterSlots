# Adapter Prefetching

vLLM keeps a warm LoRA pool of size `--max-loras`. A request for a cold adapter blocks
on a CPU→GPU load (τ_load), which shows up as TTFT latency; at high K under zipf,
cold-start fraction is significant.

We predict the next adapters from arrival history (Poisson scoring) and prefetch them
into the pool before requests arrive, hiding τ_load.

- Predictor / cache / policy: `adapter_slots/prefetch/{predictor,cache_manager,policy}.py`
- Microbench: `benchmarks/micro/m3_prefetch_cold_start.py`

Measured on 2×A6000 PCIe (TP=2): prefetching removes the cold-start TTFT spike at high K.
