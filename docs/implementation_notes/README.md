# Implementation notes

Condensed notes on each component of AdapterSlots, in build order. Results are in
[../results.md](../results.md); reproduction in [../../REPRODUCE.md](../../REPRODUCE.md).

1. [infrastructure](infrastructure.md) -- baselines, workload generator, profiling
2. [isolation_experiment](isolation_experiment.md) -- adapter mixing degrades SGMV intensity
3. [instrumentation](instrumentation.md) -- WAR, WARτ, H_align metrics
4. [kernel_decomposition](kernel_decomposition.md) -- O(N)→O(K) batch decomposition
5. [alignment_buffer](alignment_buffer.md) -- per-adapter queues, warp-aligned dispatch
6. [erlang_scheduler](erlang_scheduler.md) -- per-adapter Erlang timeouts + fairness
7. [pi_controller](pi_controller.md) -- online T_max control under drift
8. [whittle_scheduler](whittle_scheduler.md) -- Whittle-index dispatch
9. [workload_characterization](workload_characterization.md) -- real bursty traces
10. [end_to_end_serving](end_to_end_serving.md) -- full system vs SOTA
11. [multi_gpu_correctness](multi_gpu_correctness.md) -- TP/PP/preemption correctness
12. [flashinfer_composition](flashinfer_composition.md) -- composes with FlashInfer
13. [adapter_prefetching](adapter_prefetching.md) -- predictive cold-start prefetch
14. [kernel_promotion](kernel_promotion.md) -- WAR-gated merged-GEMM promotion
15. [sota_evaluation](sota_evaluation.md) -- direct SOTA comparison framework
