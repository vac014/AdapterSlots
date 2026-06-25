# Infrastructure

Baseline serving stack and measurement ground truth. No AdapterSlots logic -- this
layer exists so every later result rests on verified baselines.

- Pin and install the serving stack (vLLM 0.6.3, punica) and the llama-7b model +
  rank-32 adapter set under `models/` and `adapters/`.
- Synthetic workload generator (`workloads/pattern_generator.py`): Poisson arrivals,
  zipf adapter selection, configurable rank/K/decode-length.
- Benchmark harness (`benchmarks/ablations/bench.py`) and Nsight/NCU profiling
  wrappers (`hw_profiles/`), validated against known-good baselines before use.

Baselines established here (vanilla vLLM, punica decode microbench) are the reference
points all later phases measure against.
