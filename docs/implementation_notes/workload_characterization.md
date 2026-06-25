# Workload Characterization (E9)

Synthetic Poisson/zipf workloads understate AdapterSlots. Real LLM serving traces are
bursty and positively autocorrelated, and under autocorrelated arrivals
E[WAR] > E[WAR]_iid -- consecutive requests from one adapter fill its queue together
(Theorem 8.9).

- Trace analysis (autocorrelation, burst length): `analysis/workload_autocorrelation.py`,
  `analysis/burst_distribution.py`, `analysis/traffic_pattern_stability.py`
- Replay harness for real traces: `benchmarks/ablations/bench_real_traces.py`

Traces: BurstGPT, LMSYS-Chat-1M. The measured burstiness confirms AdapterSlots
performs better in practice than the i.i.d. lower bound predicts.
