# SGMV Decomposition (E11)

A standalone result, independent of the E1 warp narrative.

Given an unsorted batch of N tokens drawn from K adapters, SGMV must scan the batch to
decompose it into K per-adapter sub-batches before CTA assignment -- O(N) work on the
critical path. Pre-sorting tokens by adapter reduces this to O(K). For N=512, K=4 the
decomposition cost drops ~128×. Under TP this scan is replicated per rank, so the
saving scales with parallelism.

Analysis: `analysis/decomposition_slope_regression.py`; microbench
`benchmarks/micro/m1_sgmv_throughput*.py`.
