# FlashInfer Composition (E10)

AdapterSlots optimizes LoRA routing in time (warp alignment); FlashInfer optimizes
attention in space (load-balanced scheduling). They operate on orthogonal axes and
compose: FlashInfer runs the attention kernel while AdapterSlots routes LoRA, and the
combined throughput gain exceeds the sum of the individual gains.

Experiment: `scripts/experiments/flashinfer_composability.py`. Results under
`results/flashinfer_composition/`. This phase also packages the reproducibility
artifact (see [REPRODUCE.md](../../REPRODUCE.md)).
