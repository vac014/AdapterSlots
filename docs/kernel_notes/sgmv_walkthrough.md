# SGMV Kernel Walkthrough

## 1. What is SGMV?

**SGMV = Segmented General Matrix-Vector** multiplication.

In a LoRA serving system, when a batch of N tokens arrives from K different adapters,
each token `i` requires a low-rank matmul:

```
output[i] = base_weight @ x[i] + lora_B[adapter(i)] @ lora_A[adapter(i)] @ x[i]
```

Naively running this per-token is expensive. SGMV amortizes this by grouping tokens
per adapter (a "segment") and running batched GEMV for each segment.

**Key insight:** If the batch is *unsorted* (adapter IDs randomly interleaved),
the kernel must first scan all N tokens to build K segment lists. This is the O(N)
decomposition that E11 addresses.

---

## 2. Punica SGMV Kernel: File Map

```
punica/ops/sgmv_cutlass.py          Python wrapper / dispatch
punica/ops/csrc/sgmv_cutlass.cu     CUDA kernel (Cutlass-based)
punica/ops/csrc/sgmv_cutlass.h      Kernel header / templates
```

Key Python entry points:
- `sgmv_shrink()` -- applies LoRA A matrix (shrinks hidden dim to rank)
- `sgmv_expand()` -- applies LoRA B matrix (expands rank back to hidden dim)
- `segment_gemv_*` -- the actual tiled GEMV kernel launcher

---

## 3. CTA Assignment: How Tokens Map to Thread Blocks

In SGMV, **CTAs (thread blocks)** are assigned per-adapter-segment:
- For adapter k with n_k tokens, the kernel launches `ceil(n_k / TILE_SIZE)` CTAs
- Each CTA handles one tile of tokens for one adapter
- `TILE_SIZE` is typically 8–16 tokens depending on the Cutlass config

**CTA launch overhead:**
- Total CTAs = Σ_k ceil(n_k / TILE_SIZE)
- For fixed N and K=4: total CTAs ≈ N / TILE_SIZE regardless of mixing
- The overhead is in *building the segment lists* (O(N)), not the CTA count itself

**Instrumentation point:** `ncu --metrics launch__grid_size` reports total CTAs launched.

---

## 4. The O(N) Decomposition: Where It Happens

### Unsorted path (current Punica SGMV)

```python
# sgmv_cutlass.py (approximate pseudocode)
def sgmv_cutlass(inputs, lora_weights, batch_lora_ids):
    # Step 1: O(N) scan -- build segment list
    segments = {}                           # dict: adapter_id -> [token_indices]
    for i, aid in enumerate(batch_lora_ids):   # O(N) scan
        if aid not in segments:
            segments[aid] = []
        segments[aid].append(i)             # O(1) append

    # Step 2: for each adapter, launch GEMV on its segment
    for aid, indices in segments.items():   # O(K) iterations
        x_k = inputs[indices]              # gather (O(|segment|))
        result_k = lora_B[aid] @ lora_A[aid] @ x_k
        output[indices] += result_k        # scatter (O(|segment|))
```

**Complexity breakdown:**
| Step | Time | Space |
|------|------|-------|
| Scan to build segments | O(N) | O(N) |
| K GEMV kernel launches | O(K) dispatches | O(K) |
| Total GEMV compute | O(N × r) | O(N) |

### AdapterSlots pre-sorted path

When tokens are pre-sorted by adapter_id (AdapterSlots AlignmentBuffer guarantees this):

```python
# Pre-sorted: adapter IDs are [0,0,...,0, 1,1,...,1, ...]
# Boundaries known at O(K):
def build_segments_from_sorted(sorted_ids, K):
    boundaries = [0]                        # O(1) start
    for k in range(1, K):                   # O(K) scan
        # Binary search for boundary of adapter k
        # With a sorted list: O(log N) per boundary
        # With pointer scan: O(1) if boundaries[k-1] is known
        pass
    boundaries.append(len(sorted_ids))
    return boundaries
```

**Reduction:** O(N) → O(K). For N=512, K=4: 128× reduction in scan work.

---

## 5. Warp Mapping: How 32-Thread Warps Handle LoRA Tiles

Each SM warp (32 CUDA threads) processes one tile of the GEMV:
- Thread `t` in warp `j` handles row `j*TILE_ROWS + t/THREADS_PER_ROW`
- All threads in a warp access the **same LoRA weight tile**
- If two threads in a warp belong to different adapters: they access different weight
  tiles → **warp divergence** (different code paths or stall while the other tile loads)

**Key counter:** `sm__warps_active.avg.pct_of_peak_sustained_active`
- If this drops A→D: warp-level divergence is present
- If flat: SGMV operational intensity (not raw warp divergence) is the bottleneck

---

## 6. S-LoRA MBGMV vs. Punica SGMV

| Feature | Punica SGMV | S-LoRA MBGMV |
|---------|-------------|--------------|
| Kernel type | Segmented (per-adapter contiguous segments) | Multi-batch (arbitrary index lists) |
| Adapter count | Fits in VRAM | Can swap adapters (handles 1000+) |
| CTA assignment | Per-segment | Per-request (uses index buffer) |
| Decomposition | O(N) scan to build segments | Uses pre-built index buffer (O(N) fill) |
| Pre-sort benefit | O(N)→O(K) for segment building | O(N)→O(K) for index buffer fill |

---

## 7. FlashInfer Batching Path

FlashInfer differs from SGMV in the attention path, not the LoRA path:
- FlashInfer handles attention with load-balanced CTA assignment
- LoRA computation still goes through SGMV (unless FlashInfer also integrates LoRA)
- Insertion point for AdapterSlots: before FlashInfer's prefill scheduler

---

## 8. vLLM Decode Scheduler: Key Code Points

```
vllm/core/scheduler.py

Scheduler.schedule()
    └─ _schedule_running()          # which seqs continue this iter
    └─ _schedule_prefills()         # which prefill seqs to add
    └─ SchedulerOutputs()           # the batch formation struct

vllm/worker/model_runner.py
    ModelRunner.execute_model()
        └─ model.forward()
            └─ LoRA layer: calls sgmv_cutlass() or bgmv()
```

**AdapterSlots insertion point:** Override `Scheduler` via `--scheduler-class` to return
pre-sorted `scheduled_seq_groups`. This is implemented in alignment_buffer.

---

## 9. PagedAttention Lifecycle (Request Path)

```
1. HTTP request → AsyncLLMEngine.add_request()
2. Scheduler assigns KV blocks → BlockAllocator.allocate()
3. Sequence added to scheduler.waiting queue
4. scheduler.schedule() moves sequence to running
5. ModelRunner.execute_model() runs forward pass
6. SGMV kernel computes LoRA deltas for the batch
7. Tokens returned via AsyncLLMEngine.generate()
```

**Memory layout:** KV cache uses PagedAttention blocks (default 16 tokens/block).
Adapter weights are separate tensors, not paged. Swap-out (S-LoRA) moves adapter
weights to CPU RAM when VRAM is full.

---

## 10. Key Metrics and What They Mean

| Counter | What it measures | E1 interpretation |
|---------|-----------------|------------------|
| `sm__warps_active` | % of peak warps active | Drop A→D = warp divergence |
| `l2tex__t_sector_hit_rate` | L2 cache hit rate | Drop = weight cache thrashing |
| `gpu__dram_throughput` | DRAM BW utilization | Rise = more cache misses |
| `sm__cycles_active` | SM active cycle fraction | Drop = SM stalls |
| `launch__grid_size` | Total CTAs launched | Should be ~N/TILE_SIZE |

