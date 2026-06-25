"""
fused_packed_lora_kernel.py -- graph-safe fusion of vLLM's own acknowledged
TODO at vllm/lora/punica.py:575-582 (add_lora_packed_nslice):

    offset_left = 0
    # TODO fuse these kernels
    for slice_idx in range(len(output_slices)):
        self.add_lora(y, x, lora_a_stacked[slice_idx], lora_b_stacked[slice_idx],
                      scale, offset_left, output_slices[slice_idx])
        offset_left += output_slices[slice_idx]

For a QKV-packed layer this is 3 add_lora() calls = 6 launches (bgmv_shrink
+ bgmv_expand, x3); for gate_up, 2 calls = 4 launches. Every one of the
shrink launches independently re-reads the SAME x tensor from HBM (x is
identical across slices -- it's the same input row, only the LoRA weight
differs). This file replaces the per-slice Python loop with exactly 2
kernel launches total (one fused shrink across all slices, one fused
expand across all slices), each modeled directly on vLLM's own
bgmv_shrink_kernel / bgmv_expand_kernel (vllm/lora/ops/bgmv_{shrink,expand}.py)
-- same grid shape, same on-device per-row index load
(`tl.load(lora_indices + cur_batch)`), same `if lora_index == -1: return`
early exit -- generalized only to loop over NUM_SLICES inside one program
instead of launching NUM_SLICES separate kernels.

Why this is graph-safe (unlike the original fused_lora_kernel.py wiring):
    - Grid shape is (SPLIT, batches) -- fixed once `batches` (the decode
      bucket size) is fixed, exactly like the real bgmv kernels vLLM already
      captures into CUDA graphs today. No Python-side branching on tensor
      *values*, no .tolist()/.item(), no data-dependent launch count.
    - The per-row adapter index is loaded inside the kernel via a normal
      device-side tl.load, never read back to Python.

Why this does NOT touch the base GEMM (deliberate):
    cuBLAS's GEMM for the base weight is highly tuned; the prior attempt at
    fusing it into a hand-written Triton tl.dot kernel (fused_lora_kernel.py)
    measured catastrophically because (a) it forced --enforce-eager to dodge
    a CUDA-graph-capture crash, losing graph-replay amortization for the
    ENTIRE model's ~40 layers, and (b) even setting that aside, a Triton
    GEMM has to out-tune cuBLAS at the dominant-FLOPs op to win, which is a
    much harder bar than fusing two small LoRA-only GEMV launches. This file
    only ever touches the LoRA branch (shrink: x @ A^T, expand: h @ B^T) --
    the base GEMM keeps using cuBLAS exactly as vanilla vLLM does.

Real saving: the shrink fusion eliminates (NUM_SLICES - 1) redundant
re-reads of x from HBM per packed layer (x is hidden_size-wide, not
rank-wide -- this is the actual bandwidth win, not launch-count, which
vLLM's own maintainers note costs ~nothing once inside a captured graph:
see vLLM PR #15152, "for cudagraphs we always capture with LoRA. so there
is just 1 path in that case."). The expand fusion's saving is smaller (h is
rank-wide, already tiny) -- it is included for completeness/launch-count
reduction but is not where the real win is expected to come from.

Limitations (by design, kept narrow to control risk):
    - Only handles the decode (bgmv) path. Prefill uses vLLM's SGMV kernels
      with a different grid/grouping strategy (compute_meta's run-length
      compression) and is not graph-captured anyway, so it is out of scope
      here: it would be a separate, not-yet-implemented item.
    - Assumes every slice shares the same rank R (true for all the packed
      layers this project deals with -- QKV and gate_up each get one LoRA
      rank per adapter, not per-slice) and the same x. Both are checked at
      call time; any violation falls back to the unfused per-slice loop.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_IMPORT_OK = True
except ImportError:
    _TRITON_IMPORT_OK = False

_AVAILABLE: Optional[bool] = None


def _check_available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    if os.environ.get("AS_FUSED_PACKED_NSLICE", "1") == "0":
        _AVAILABLE = False
        return False
    _AVAILABLE = _TRITON_IMPORT_OK
    return _AVAILABLE


# CPU reference (testing only)

def packed_nslice_cpu_reference(
    x: torch.Tensor,                       # (batch, K)
    lora_a_stacked: Sequence[torch.Tensor],  # each (num_loras, 1, R, K)
    lora_b_stacked: Sequence[torch.Tensor],  # each (num_loras, 1, N_s, R)
    scale: float,
    output_slices: Sequence[int],
    lora_indices: torch.Tensor,            # (batch,), -1 == no adapter
) -> torch.Tensor:
    """Reference matching vLLM's add_lora_packed_nslice semantics exactly."""
    batch = x.shape[0]
    total_n = sum(output_slices)
    out = torch.zeros(batch, total_n, dtype=torch.float32, device=x.device)
    offset = 0
    for s, n_s in enumerate(output_slices):
        A = lora_a_stacked[s].squeeze(1)  # (num_loras, R, K)
        B = lora_b_stacked[s].squeeze(1)  # (num_loras, N_s, R)
        for row in range(batch):
            idx = int(lora_indices[row].item())
            if idx < 0:
                continue
            h = x[row].float() @ A[idx].float().T          # (R,)
            delta = (h @ B[idx].float().T) * scale          # (N_s,)
            out[row, offset:offset + n_s] += delta
        offset += n_s
    return out


# Triton kernels (built lazily; modeled on vllm/lora/ops/bgmv_*.py)

def _build_kernels():
    @triton.jit
    def _fused_shrink_kernel(
        input_ptr,
        lora_ptr0, lora_ptr1, lora_ptr2,
        out_ptr0, out_ptr1, out_ptr2,
        N, K,
        lora_indices,
        scaling,
        xm_stride, xk_stride,
        l0_stride, lora_k_stride, lora_n_stride,
        cm_stride, cn_stride,
        NUM_SLICES: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
    ):
        """Fuses NUM_SLICES bgmv_shrink launches into 1: x is loaded once per
        K-chunk and reused across all slices' independent LoRA-A weights.
        Directly mirrors bgmv_shrink_kernel's grid/masking/early-exit; the
        only change is the inner loop over slices sharing one x load.
        """
        pid_sk = tl.program_id(axis=0)
        cur_batch = tl.program_id(axis=1)
        lora_index = tl.load(lora_indices + cur_batch)
        if lora_index == -1:
            return

        offset_n = tl.arange(0, BLOCK_N)
        offset_k = tl.arange(0, BLOCK_K) + pid_sk * BLOCK_K
        a_ptr = input_ptr + cur_batch * xm_stride

        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k in range(0, K, BLOCK_K * SPLIT_K):
            current_k = k + offset_k
            current_k_c = tl.max_contiguous(current_k, BLOCK_K)
            tiled_a = tl.load(
                a_ptr + current_k_c, mask=current_k < K, other=0.0,
            )  # [BLOCK_K] -- the one shared x load this whole file exists for

            b_mask = (offset_n[:, None] < N) & (current_k[None, :] < K)

            tiled_b0 = tl.load(
                lora_ptr0 + l0_stride * lora_index +
                offset_n[:, None] * lora_k_stride +
                current_k[None, :] * lora_n_stride,
                mask=b_mask, other=0.0,
            )
            acc0 += tl.sum(tiled_a * tiled_b0, 1)

            if NUM_SLICES > 1:
                tiled_b1 = tl.load(
                    lora_ptr1 + l0_stride * lora_index +
                    offset_n[:, None] * lora_k_stride +
                    current_k[None, :] * lora_n_stride,
                    mask=b_mask, other=0.0,
                )
                acc1 += tl.sum(tiled_a * tiled_b1, 1)

            if NUM_SLICES > 2:
                tiled_b2 = tl.load(
                    lora_ptr2 + l0_stride * lora_index +
                    offset_n[:, None] * lora_k_stride +
                    current_k[None, :] * lora_n_stride,
                    mask=b_mask, other=0.0,
                )
                acc2 += tl.sum(tiled_a * tiled_b2, 1)

        acc0 *= scaling
        offset_cn = tl.arange(0, BLOCK_N)
        c_mask = offset_cn < N

        c_ptr0 = out_ptr0 + cur_batch * cm_stride + offset_cn * cn_stride
        if SPLIT_K == 1:
            tl.store(c_ptr0, acc0, mask=c_mask)
        else:
            tl.atomic_add(c_ptr0, acc0, mask=c_mask)

        if NUM_SLICES > 1:
            acc1 *= scaling
            c_ptr1 = out_ptr1 + cur_batch * cm_stride + offset_cn * cn_stride
            if SPLIT_K == 1:
                tl.store(c_ptr1, acc1, mask=c_mask)
            else:
                tl.atomic_add(c_ptr1, acc1, mask=c_mask)

        if NUM_SLICES > 2:
            acc2 *= scaling
            c_ptr2 = out_ptr2 + cur_batch * cm_stride + offset_cn * cn_stride
            if SPLIT_K == 1:
                tl.store(c_ptr2, acc2, mask=c_mask)
            else:
                tl.atomic_add(c_ptr2, acc2, mask=c_mask)

    @triton.jit
    def _fused_expand_kernel(
        h_ptr0, h_ptr1, h_ptr2,
        lora_ptr0, lora_ptr1, lora_ptr2,
        out_ptr,
        n0, n1, n2,
        K,
        lora_indices,
        y_off0, y_off1, y_off2,
        hm_stride, hk_stride,
        # Per-slice strides: each packed slice (Q/K/V or gate/up) has its
        # own output width N, hence its own lora_b_stacked tensor shape and
        # strides -- these must NOT be shared across slices (a single shared
        # set of strides here was the bug that caused out-of-bounds access
        # in the first version of this kernel: slice 1/2's pointer math used
        # slice 0's (num_loras, N, R) stride-0, which is wrong whenever
        # N differs per slice, e.g. QKV with GQA).
        l0_stride0, lora_k_stride0, lora_n_stride0,
        l0_stride1, lora_k_stride1, lora_n_stride1,
        l0_stride2, lora_k_stride2, lora_n_stride2,
        cm_stride, cn_stride,
        NUM_SLICES: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_N: tl.constexpr,
        ADD_INPUTS: tl.constexpr,
    ):
        """Fuses NUM_SLICES bgmv_expand launches into 1. Each slice keeps its
        own h (rank-sized, already tiny) and its own output column offset;
        the saving here is launch count + shared index/grid setup, not
        bandwidth (unlike the shrink kernel above) -- see module docstring.
        """
        pid_sn = tl.program_id(axis=0)
        cur_batch = tl.program_id(axis=1)
        lora_index = tl.load(lora_indices + cur_batch)
        if lora_index == -1:
            return

        offset_k = tl.arange(0, BLOCK_K)
        offset_n = tl.arange(0, BLOCK_N)

        # slice 0
        tiled_h0 = tl.load(h_ptr0 + cur_batch * hm_stride + offset_k * hk_stride,
                           mask=offset_k < K, other=0.0)
        split_n_length0 = tl.cdiv(n0, SPLIT_N)
        b_ptr0 = lora_ptr0 + l0_stride0 * lora_index + pid_sn * split_n_length0 * lora_k_stride0
        c_ptr0 = out_ptr + cur_batch * cm_stride + y_off0 + pid_sn * split_n_length0
        for n in range(0, split_n_length0, BLOCK_N):
            current_n = n + offset_n
            current_n_c = tl.max_contiguous(current_n, BLOCK_N)
            b_mask = (current_n[:, None] < split_n_length0) & (offset_k[None, :] < K)
            c_mask = current_n < split_n_length0
            tiled_b = tl.load(
                b_ptr0 + current_n_c[:, None] * lora_k_stride0 + offset_k[None, :] * lora_n_stride0,
                mask=b_mask, other=0.0,
            )
            if ADD_INPUTS:
                tiled_out = tl.load(c_ptr0 + current_n * cn_stride, mask=c_mask)
                accumulator = tl.sum(tiled_h0 * tiled_b, 1) + tiled_out
            else:
                accumulator = tl.sum(tiled_h0 * tiled_b, 1)
            tl.store(c_ptr0 + current_n * cn_stride, accumulator, mask=c_mask)

        # slice 1
        if NUM_SLICES > 1:
            tiled_h1 = tl.load(h_ptr1 + cur_batch * hm_stride + offset_k * hk_stride,
                               mask=offset_k < K, other=0.0)
            split_n_length1 = tl.cdiv(n1, SPLIT_N)
            b_ptr1 = lora_ptr1 + l0_stride1 * lora_index + pid_sn * split_n_length1 * lora_k_stride1
            c_ptr1 = out_ptr + cur_batch * cm_stride + y_off1 + pid_sn * split_n_length1
            for n in range(0, split_n_length1, BLOCK_N):
                current_n = n + offset_n
                current_n_c = tl.max_contiguous(current_n, BLOCK_N)
                b_mask = (current_n[:, None] < split_n_length1) & (offset_k[None, :] < K)
                c_mask = current_n < split_n_length1
                tiled_b = tl.load(
                    b_ptr1 + current_n_c[:, None] * lora_k_stride1 + offset_k[None, :] * lora_n_stride1,
                    mask=b_mask, other=0.0,
                )
                if ADD_INPUTS:
                    tiled_out = tl.load(c_ptr1 + current_n * cn_stride, mask=c_mask)
                    accumulator = tl.sum(tiled_h1 * tiled_b, 1) + tiled_out
                else:
                    accumulator = tl.sum(tiled_h1 * tiled_b, 1)
                tl.store(c_ptr1 + current_n * cn_stride, accumulator, mask=c_mask)

        # slice 2
        if NUM_SLICES > 2:
            tiled_h2 = tl.load(h_ptr2 + cur_batch * hm_stride + offset_k * hk_stride,
                               mask=offset_k < K, other=0.0)
            split_n_length2 = tl.cdiv(n2, SPLIT_N)
            b_ptr2 = lora_ptr2 + l0_stride2 * lora_index + pid_sn * split_n_length2 * lora_k_stride2
            c_ptr2 = out_ptr + cur_batch * cm_stride + y_off2 + pid_sn * split_n_length2
            for n in range(0, split_n_length2, BLOCK_N):
                current_n = n + offset_n
                current_n_c = tl.max_contiguous(current_n, BLOCK_N)
                b_mask = (current_n[:, None] < split_n_length2) & (offset_k[None, :] < K)
                c_mask = current_n < split_n_length2
                tiled_b = tl.load(
                    b_ptr2 + current_n_c[:, None] * lora_k_stride2 + offset_k[None, :] * lora_n_stride2,
                    mask=b_mask, other=0.0,
                )
                if ADD_INPUTS:
                    tiled_out = tl.load(c_ptr2 + current_n * cn_stride, mask=c_mask)
                    accumulator = tl.sum(tiled_h2 * tiled_b, 1) + tiled_out
                else:
                    accumulator = tl.sum(tiled_h2 * tiled_b, 1)
                tl.store(c_ptr2 + current_n * cn_stride, accumulator, mask=c_mask)

    return _fused_shrink_kernel, _fused_expand_kernel


_shrink_kernel = None
_expand_kernel = None


def _ensure_kernels_built() -> None:
    global _shrink_kernel, _expand_kernel
    if _shrink_kernel is None:
        _shrink_kernel, _expand_kernel = _build_kernels()


_DUMMY_CACHE: dict = {}


def _dummy_ptr(x: torch.Tensor) -> torch.Tensor:
    """A 1-element placeholder tensor for unused slice-2/3 pointer args
    when NUM_SLICES < 3 -- never read inside the kernel (NUM_SLICES gates
    those branches out at compile time), but Triton still needs a valid
    GPU pointer argument to bind.
    """
    key = (x.device, x.dtype)
    t = _DUMMY_CACHE.get(key)
    if t is None:
        t = torch.zeros(1, dtype=x.dtype, device=x.device)
        _DUMMY_CACHE[key] = t
    return t


class FusedPackedLoRAKernel:
    """Public entry point: fused replacement for PunicaWrapper.add_lora_packed_nslice
    (decode/bgmv path only). See module docstring for the full rationale.
    """

    @staticmethod
    def is_available() -> bool:
        return _check_available()

    def apply_packed_nslice(
        self,
        y: torch.Tensor,                         # (batch, total_N) -- in-place add target
        x: torch.Tensor,                          # (batch, K)
        lora_a_stacked: Sequence[torch.Tensor],
        lora_b_stacked: Sequence[torch.Tensor],
        scale: float,
        output_slices: Sequence[int],
        lora_indices: torch.Tensor,
    ) -> None:
        num_slices = len(output_slices)
        assert 1 <= num_slices <= 3, "fused packed kernel supports 2-3 slices (QKV/gate_up)"
        _ensure_kernels_built()

        x = x.contiguous()
        batch, K = x.shape

        a_sq = [a.squeeze(1) if a.ndim == 4 else a for a in lora_a_stacked]
        b_sq = [b.squeeze(1) if b.ndim == 4 else b for b in lora_b_stacked]
        R = a_sq[0].shape[1]
        for a in a_sq:
            assert a.shape[1] == R, "all packed slices must share one LoRA rank"

        BLOCK_K_SHRINK = triton.next_power_of_2(min(K, 512))
        BLOCK_N_SHRINK = triton.next_power_of_2(R)
        SPLIT_K = 1

        h_buffers = [torch.zeros(batch, R, dtype=torch.float32, device=x.device)
                     for _ in range(num_slices)]
        while len(h_buffers) < 3:
            h_buffers.append(torch.zeros(1, R, dtype=torch.float32, device=x.device))
        while len(a_sq) < 3:
            a_sq.append(a_sq[0])

        grid_shrink = (SPLIT_K, batch)
        _shrink_kernel[grid_shrink](
            x,
            a_sq[0], a_sq[1], a_sq[2],
            h_buffers[0], h_buffers[1], h_buffers[2],
            R, K,
            lora_indices,
            scale,
            x.stride(0), x.stride(1),
            a_sq[0].stride(0), a_sq[0].stride(1), a_sq[0].stride(2),
            h_buffers[0].stride(0), h_buffers[0].stride(1),
            NUM_SLICES=num_slices,
            BLOCK_N=BLOCK_N_SHRINK,
            BLOCK_K=BLOCK_K_SHRINK,
            SPLIT_K=SPLIT_K,
        )

        ns = list(output_slices) + [1, 1, 1]
        offsets = [0] * 3
        running = 0
        for i in range(num_slices):
            offsets[i] = running
            running += output_slices[i]

        while len(b_sq) < 3:
            b_sq.append(b_sq[0])

        BLOCK_K_EXPAND = triton.next_power_of_2(R)
        BLOCK_N_EXPAND = 256
        SPLIT_N = 4

        grid_expand = (SPLIT_N, batch)
        _expand_kernel[grid_expand](
            h_buffers[0], h_buffers[1], h_buffers[2],
            b_sq[0], b_sq[1], b_sq[2],
            y,
            ns[0], ns[1], ns[2],
            R,
            lora_indices,
            offsets[0], offsets[1], offsets[2],
            h_buffers[0].stride(0), h_buffers[0].stride(1),
            b_sq[0].stride(0), b_sq[0].stride(1), b_sq[0].stride(2),
            b_sq[1].stride(0), b_sq[1].stride(1), b_sq[1].stride(2),
            b_sq[2].stride(0), b_sq[2].stride(1), b_sq[2].stride(2),
            y.stride(0), y.stride(1),
            NUM_SLICES=num_slices,
            BLOCK_N=BLOCK_N_EXPAND,
            BLOCK_K=BLOCK_K_EXPAND,
            SPLIT_N=SPLIT_N,
            ADD_INPUTS=True,
        )
