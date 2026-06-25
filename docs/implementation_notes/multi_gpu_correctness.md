# Multi-GPU Correctness

The alignment buffer must stay correct under tensor parallelism, pipeline parallelism,
and preemption -- not just single-GPU.

- Tensor parallelism: dispatch is identical across ranks, so alignment is transparent
  to TP. Verified by `tests/test_tp_transparency.py`, `scripts/test_tp_correctness.py`.
- Pipeline parallelism: `scripts/test_pp_correctness.py`.
- Preemption safety: buffered tokens survive vLLM preempt/swap without loss
  (Theorem 8.11).
- KV-cache stress: `scripts/kv_stress_test.py`.

Outputs match the single-GPU reference bit-for-bit (greedy).
