"""tiled_segment_gemm.py -- grouped LoRA GEMM with tile scheduling.

vLLM's SGMV kernel indexes segments on grid axis 2, so a static (capturable)
launch must size that axis to the worst-case segment count and every
(tile, segment) pair that does not overlap early-exits. At K=10 that is ~11x
more programs than do useful work.

Here each program takes ONE work tile and looks up which segment it belongs to
via a per-step tile->segment map. The number of tiles is bounded by
cdiv(M, BLOCK_M) + n_seg (each segment contributes at most one partial tile),
so the grid is still static and graph-capturable, but ~5x smaller.

Math is identical to vLLM's sgmv_shrink / sgmv_expand_slice; only the index
derivation changes.
"""
from __future__ import annotations
import torch, triton, triton.language as tl


@triton.jit
def _tiled_shrink_kernel(
    input_ptr, lora_ptr, out_ptr, N, K,
    seg_start, seg_len, seg_lora, tile_seg, tile_m,
    scaling,
    xm_stride, xk_stride, l0_stride, lora_k_stride, lora_n_stride,
    cm_stride, cn_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid_t = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_sk = tl.program_id(axis=2)
    s = tl.load(tile_seg + pid_t)
    if s < 0:
        return
    lora_index = tl.load(seg_lora + s)
    if lora_index == -1:
        return
    M = tl.load(seg_len + s)
    pid_m = tl.load(tile_m + pid_t)
    if pid_m * BLOCK_M >= M:
        return
    cur_seq_start = tl.load(seg_start + s)

    offset_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    ram = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride
             + offset_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + l0_stride * lora_index + rbn[None, :] * lora_k_stride
             + offset_k[:, None] * lora_n_stride)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(a_ptr); b = tl.load(b_ptr)
        else:
            krem = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(a_ptr, mask=offset_k[None, :] < krem, other=0.0)
            b = tl.load(b_ptr, mask=offset_k[:, None] < krem, other=0.0)
        acc += tl.dot(a, b)
        a_ptr += BLOCK_K * SPLIT_K * xk_stride
        b_ptr += BLOCK_K * SPLIT_K * lora_n_stride
    offset_cm = cur_seq_start + tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    c_ptr = out_ptr + offset_cm[:, None] * cm_stride + offset_cn[None, :] * cn_stride
    c_mask = (offset_cm[:, None] < (cur_seq_start + M)) & (offset_cn[None, :] < N)
    acc *= scaling
    if SPLIT_K == 1:
        tl.store(c_ptr, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, acc, mask=c_mask)


@triton.jit
def _tiled_expand_slice_kernel(
    input_ptr, lora_ptr, out_ptr, N, K,
    seg_start, seg_len, seg_lora, tile_seg, tile_m,
    xm_stride, xk_stride, l0_stride, lora_k_stride, lora_n_stride,
    cm_stride, cn_stride, slice_offset,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr, ADD_INPUTS: tl.constexpr, CAST_TYPE: tl.constexpr,
):
    pid_t = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    s = tl.load(tile_seg + pid_t)
    if s < 0:
        return
    lora_index = tl.load(seg_lora + s)
    if lora_index == -1:
        return
    M = tl.load(seg_len + s)
    pid_m = tl.load(tile_m + pid_t)
    if pid_m * BLOCK_M >= M:
        return
    cur_seq_start = tl.load(seg_start + s)

    offset_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K)
    ram = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride
             + offset_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + l0_stride * lora_index
             + offset_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptr); b = tl.load(b_ptr)
        else:
            krem = K - k * BLOCK_K
            a = tl.load(a_ptr, mask=offset_k[None, :] < krem, other=0.0)
            b = tl.load(b_ptr, mask=offset_k[:, None] < krem, other=0.0)
        if CAST_TYPE:
            a = a.to(lora_ptr.dtype.element_ty)
        acc += tl.dot(a, b)
        a_ptr += BLOCK_K * xk_stride
        b_ptr += BLOCK_K * lora_n_stride
    tiled_c = acc.to(lora_ptr.dtype.element_ty)
    offset_cm = cur_seq_start + tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + slice_offset
    c_ptr = out_ptr + offset_cm[:, None] * cm_stride + offset_cn[None, :] * cn_stride
    c_mask = ((offset_cm[:, None] < (cur_seq_start + M))
              & (offset_cn[None, :] < (slice_offset + N)))
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask, other=0.0)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


def build_tile_map(seg_len: torch.Tensor, block_m: int, max_tiles: int,
                   tile_seg: torch.Tensor, tile_m: torch.Tensor) -> None:
    """Fill tile_seg/tile_m so each entry names one (segment, m-tile) work item.

    Fully device-side: no .item()/int() read anywhere. An earlier version called
    int(ntiles.sum()) to size the work list, which forced a GPU->CPU sync on every
    decode step and stalled the pipeline -- in-system that made the tiled path
    SLOWER than the static-grid one despite winning the kernel microbenchmark.

    Positions beyond the real tile count get tile_seg = -1 and are early-exited
    by the kernel, so the launch grid stays static and graph-capturable.
    """
    n_seg = seg_len.numel()
    dev = seg_len.device
    ntiles = (seg_len + block_m - 1) // block_m          # tiles per segment
    cum = torch.cumsum(ntiles, 0)                        # exclusive-end offsets
    starts = cum - ntiles
    pos = torch.arange(max_tiles, device=dev, dtype=torch.long)
    # segment owning each tile slot; searchsorted keeps this on-device
    seg_of = torch.searchsorted(cum, pos, right=True)
    seg_clamped = seg_of.clamp_(max=n_seg - 1)
    local = pos - starts[seg_clamped]
    valid = pos < cum[n_seg - 1]
    tile_seg.copy_(torch.where(valid, seg_clamped, torch.full_like(seg_clamped, -1)))
    tile_m.copy_(torch.where(valid, local, torch.zeros_like(local)))
