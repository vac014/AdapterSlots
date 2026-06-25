# End-to-End Serving (E2–E5)

The full system -- alignment buffer + Erlang/PI/Whittle dispatch + fused SGMV -- run
against SOTA baselines across hardware and realistic workloads. This phase produces
the headline numbers; see [docs/results.md](../results.md).

- E2: WAR variability baseline (the problem)
- E3: WAR control (AdapterSlots controls WAR, monotone in T_max)
- E4: end-to-end throughput vs vLLM and the SOTA field
- E5: latency / SLO behaviour

Harness: `benchmarks/ablations/bench.py`. Results under `results/end_to_end_serving/`.
