"""tiled_multislice.py -- tile-scheduled grouped LoRA GEMM over packed slices.

vLLM applies LoRA to a packed projection (QKV: 3 slices, gate_up: 2) one slice
at a time: a zeroed buffer, a shrink and an expand per slice, so a decode step
of a 40-layer model issues ~840 small kernels for LoRA alone. Under CUDA graphs
each still costs its own launch latency and tail, which at decode batch sizes
is most of its runtime.

Here one shrink launch covers every slice (grid axis 2 = slice x split-K) and
one expand launch writes every output slice, using the same per-step
tile->segment work list as tiled_segment_gemm.py. The slices' weight tensors
are separate allocations; the kernels take the first one as the base pointer
and the others as element offsets from it. vLLM's stacked LoRA buffers are
allocated once and never move, so the offsets are fixed scalars and safe to
bake into a captured graph.
"""
from __future__ import annotations
import triton
import triton.language as tl


@triton.jit
def _tiled_shrink_ms_kernel(
    input_ptr, lora_ptr, out_ptr, N, K,
    seg_start, seg_len, seg_lora, tile_seg, tile_m,
    scaling, woff1, woff2,
    xm_stride, xk_stride, l0_stride, lora_k_stride, lora_n_stride,
    cm_stride, cn_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid_t = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_2 = tl.program_id(axis=2)
    sid = pid_2 // SPLIT_K
    pid_sk = pid_2 % SPLIT_K
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
    woff = tl.where(sid == 1, woff1, tl.where(sid == 2, woff2, 0))

    offset_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    ram = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride
             + offset_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + woff + l0_stride * lora_index + rbn[None, :] * lora_k_stride
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
    c_ptr = (out_ptr + offset_cm[:, None] * cm_stride
             + (offset_cn[None, :] + sid * N) * cn_stride)
    c_mask = (offset_cm[:, None] < (cur_seq_start + M)) & (offset_cn[None, :] < N)
    acc *= scaling
    if SPLIT_K == 1:
        tl.store(c_ptr, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, acc, mask=c_mask)


@triton.jit
def _tiled_expand_ms_kernel(
    input_ptr, lora_ptr, out_ptr, N, K,
    seg_start, seg_len, seg_lora, tile_seg, tile_m,
    woff1, woff2, yoff0, yoff1, yoff2,
    xm_stride, xk_stride, l0_stride, lora_k_stride, lora_n_stride,
    cm_stride, cn_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr, ADD_INPUTS: tl.constexpr, CAST_TYPE: tl.constexpr,
):
    pid_t = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    sid = tl.program_id(axis=2)
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
    woff = tl.where(sid == 1, woff1, tl.where(sid == 2, woff2, 0))
    yoff = tl.where(sid == 1, yoff1, tl.where(sid == 2, yoff2, yoff0))

    offset_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K)
    ram = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride
             + (offset_k[None, :] + sid * K) * xk_stride)
    b_ptr = (lora_ptr + woff + l0_stride * lora_index
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
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    c_ptr = (out_ptr + offset_cm[:, None] * cm_stride
             + (offset_cn[None, :] + yoff) * cn_stride)
    c_mask = (offset_cm[:, None] < (cur_seq_start + M)) & (offset_cn[None, :] < N)
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask, other=0.0)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


def slice_offsets(tensors) -> list:
    """Element offsets of each slice's weight tensor from the first one's."""
    base = tensors[0].data_ptr()
    esz = tensors[0].element_size()
    offs = []
    for t in tensors:
        d = t.data_ptr() - base
        assert d % esz == 0
        offs.append(d // esz)
    return offs + [0] * (3 - len(offs))
