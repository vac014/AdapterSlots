# Alignment Buffer

The core mechanism. Incoming tokens accumulate into per-adapter queues; a batch is
dispatched when a queue reaches a warp-sized aligned group or its timeout expires.

This phase uses a threshold policy with a single global T_max (per-adapter Erlang
timeouts, PI control, and Whittle dispatch are added in later phases).

- Buffer and dispatch: `adapter_slots/buffer.py`
- vLLM integration via scheduler subclass: `adapter_slots/integrations/vllm_scheduler.py`

Throughput is monotone in T_max and the buffer is loss-free (Theorems 8.10, 11.1, 11.2).
