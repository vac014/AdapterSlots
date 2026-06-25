# Isolation Experiment (E1)

Controlled test of whether mixing adapters in a batch degrades GPU execution.

Four-condition isolation (single adapter vs mixed, aligned vs unaligned) on A6000.
The dominant effect is **SGMV operational-intensity collapse**, not raw warp
divergence: an unsorted multi-adapter batch forces the kernel into low-intensity,
memory-bound execution. This is the hardware premise AdapterSlots exploits -- aligning
batches by adapter restores intensity.

Harness: `benchmarks/isolation/benchmark_e1.py`, `benchmark_e1_scale.py`.
