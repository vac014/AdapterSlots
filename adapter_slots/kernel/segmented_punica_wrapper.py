"""segmented_punica_wrapper.py -- WGKP Level-2, implemented as a grouped GEMM.

The AlignmentBuffer already emits an adapter-contiguous decode batch
(vllm_scheduler.py sorts scheduled_seq_groups by adapter id). vLLM's own SGMV
kernel is a GroupGEMM+SPLIT-K using tl.dot (tensor cores) but is only ever used
in prefill, because a vanilla decode batch is not adapter-sorted. This wrapper
routes the decode path through that grouped kernel instead of the per-token BGMV
gather, so the sorted batch determines which kernel runs.

Graph safety: the SGMV kernels read segment length and adapter index from DEVICE
tensors (`M = tl.load(seq_lens + cur_batch)`, `if lora_index == -1: return`) and
mask their stores, so the grid can be sized to a static worst case
(MAXSEG = num_loras, MAXM = max_batches) and captured. Segment metadata is
recomputed eagerly in update_metadata() into persistent buffers, exactly the way
vLLM already refreshes token_lora_indices, so graph replay picks up new values.

Promotion gate (GWAR): grouped GEMM only wins above a token-count crossover
(~200 on A6000); below it the static-grid overhead dominates and we fall back to
stock BGMV. AS_SEGGEMM_MIN_TOKENS sets the gate.
"""
from __future__ import annotations
import os
import torch
import triton

from vllm.lora.ops.sgmv_shrink import _sgmv_shrink_kernel
from vllm.lora.ops.sgmv_expand_slice import _sgmv_expand_slice_kernel
from vllm.lora.punica import PunicaWrapper

_VLLM_PUNICA_AVAILABLE = True

_MIN_TOKENS = int(os.environ.get("AS_SEGGEMM_MIN_TOKENS", "192"))
# 1 = vLLM's stock SGMV on a static (tile x segment) grid
# 2 = tile-scheduled grouped GEMM: grid sized to the useful tiles only
_MODE = int(os.environ.get("AS_SEGGEMM", "1") or "1")
_BM_SHRINK, _BM_EXPAND = 32, 32


class SegmentedPunicaWrapper(PunicaWrapper):
    """PunicaWrapper whose decode path uses grouped SGMV over adapter segments."""

    def update_metadata(self, mapping, lora_index_to_id, max_loras, *args, **kwargs):
        """Recompute segments ONCE per step, eagerly, here -- never inside a
        captured graph. The SGMV kernels then only READ these persistent buffers,
        so graph replay picks up each step's values without capturing any of the
        metadata math (same contract vLLM uses for token_lora_indices)."""
        out = super().update_metadata(mapping, lora_index_to_id, max_loras,
                                      *args, **kwargs)
        if not mapping.is_prefill:
            self._refresh_segments(max_loras)
            if _MODE == 2:
                self._refresh_tiles(max_loras)
        return out

    def _seg_bufs(self, n_loras: int, device):
        if getattr(self, "_seg_cap", -1) != n_loras:
            self._seg_cap = n_loras
            self._seg_start = torch.zeros(n_loras, dtype=torch.long, device=device)
            self._seg_len = torch.zeros(n_loras, dtype=torch.long, device=device)
            self._seg_lora = torch.full((n_loras,), -1, dtype=torch.long, device=device)
            self._seg_dirty = True
        return self._seg_start, self._seg_len, self._seg_lora

    def _refresh_segments(self, n_loras: int) -> int:
        """Recompute adapter-run segments from token_lora_indices. Sync-free."""
        idx = self.token_lora_indices
        n = idx.numel()
        start, length, lora = self._seg_bufs(n_loras, idx.device)
        start.zero_(); length.zero_(); lora.fill_(-1)
        if n == 0:
            return 0
        is_start = torch.ones(n, dtype=torch.bool, device=idx.device)
        is_start[1:] = idx[1:] != idx[:-1]
        seg_id = torch.cumsum(is_start.long(), 0) - 1        # [n] segment per token
        seg_id = seg_id.clamp_(max=n_loras - 1)
        pos = torch.arange(n, device=idx.device, dtype=torch.long)
        length.scatter_add_(0, seg_id, torch.ones_like(pos))
        start.scatter_reduce_(0, seg_id, pos, reduce="amin", include_self=False)
        lora.scatter_(0, seg_id, idx.long())
        lora.masked_fill_(length == 0, -1)
        return n

    def _refresh_tiles(self, n_loras: int) -> None:
        """Build the tile->segment work list once per step (eager, never captured)."""
        from adapter_slots.kernel.tiled_segment_gemm import build_tile_map
        n_tok = self.token_lora_indices.numel()
        cap = (n_tok + _BM_SHRINK - 1) // _BM_SHRINK + n_loras + 1
        dev = self._seg_len.device
        if getattr(self, "_tile_cap", -1) != cap:
            self._tile_cap = cap
            self._tile_seg = torch.full((cap,), -1, dtype=torch.long, device=dev)
            self._tile_m = torch.zeros(cap, dtype=torch.long, device=dev)
        build_tile_map(self._seg_len, _BM_SHRINK, cap, self._tile_seg, self._tile_m)

    def _use_grouped(self, x: torch.Tensor) -> bool:
        return (not self.is_prefill) and x.shape[0] >= _MIN_TOKENS

    # -- decode overrides -------------------------------------------------
    def shrink_decode(self, y, x, w_t_all, scale):
        if not self._use_grouped(x):
            return super().shrink_decode(y, x, w_t_all, scale)
        # Grid depth is the SEGMENT-SLOT count (buffer size), not w_t_all.shape[0]:
        # vLLM passes lora_slots+1 as max_loras, so the two differ by one. Padding
        # slots carry lora_index=-1 and are early-exited by the kernel.
        n_seg = getattr(self, "_seg_cap", -1)
        if n_seg <= 0:
            return super().shrink_decode(y, x, w_t_all, scale)
        start, length, lora = self._seg_start, self._seg_len, self._seg_lora
        M, K, N = x.shape[0], x.shape[1], w_t_all.shape[-2]
        # Block sizes are rank-dependent. vLLM's defaults were tuned for rank 16;
        # at rank 64 a swept configuration is 15.6% faster on the LoRA op
        # (44.44us -> 38.43us, 6.17x -> 7.14x vs BGMV, h=4096 N=256 K=10).
        if N >= 128:      # rank 128: swept 53.40us vs 61.90us default (13.74x vs BGMV)
            BM, BN, BK, SK = 16, 32, 64, 4
        elif N >= 64:     # rank 64: swept 38.43us vs 44.44us default (7.14x)
            BM, BN, BK, SK = 16, 16, 64, 8
        else:
            BM, BN, BK, SK = 32, 16, 32, 8
        if _MODE == 2:
            from adapter_slots.kernel.tiled_segment_gemm import _tiled_shrink_kernel
            g = (self._tile_cap, triton.cdiv(N, BN), SK)
            _tiled_shrink_kernel[g](
                x, w_t_all, y, N, K, start, length, lora,
                self._tile_seg, self._tile_m, scale,
                x.stride(0), x.stride(1),
                w_t_all.stride(0), w_t_all.stride(2), w_t_all.stride(3),
                y.stride(0), y.stride(1), BM, BN, BK, K % (BK * SK) == 0, SK)
            return
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), SK, n_seg)
        _sgmv_shrink_kernel[grid](
            x, w_t_all, y, N, K, start, length, lora, scale,
            x.stride(0), x.stride(1),
            w_t_all.stride(0), w_t_all.stride(2), w_t_all.stride(3),
            y.stride(0), y.stride(1), BM, BN, BK, K % (BK * SK) == 0, SK)

    def expand_decode(self, y, x, w_t_all, add_input):
        return self.expand_slice_decode(y, x, w_t_all, 0, w_t_all.shape[-2], add_input)

    def expand_slice_decode(self, y, x, w_t_all, y_offset, y_slice_size, add_input):
        if not self._use_grouped(x):
            return super().expand_slice_decode(y, x, w_t_all, y_offset,
                                               y_slice_size, add_input)
        n_seg = getattr(self, "_seg_cap", -1)
        if n_seg <= 0:
            return super().expand_slice_decode(y, x, w_t_all, y_offset,
                                               y_slice_size, add_input)
        start, length, lora = self._seg_start, self._seg_len, self._seg_lora
        M, K, N = x.shape[0], x.shape[1], y_slice_size
        if K >= 128:
            BM, BN, BK = 32, 64, 32
        elif K >= 64:
            BM, BN, BK = 32, 128, 32
        else:
            BM, BN, BK = 32, 32, 16
        if _MODE == 2:
            from adapter_slots.kernel.tiled_segment_gemm import _tiled_expand_slice_kernel
            g = (self._tile_cap, triton.cdiv(N, BN))
            _tiled_expand_slice_kernel[g](
                x, w_t_all, y, N, K, start, length, lora,
                self._tile_seg, self._tile_m,
                x.stride(0), x.stride(1),
                w_t_all.stride(0), w_t_all.stride(2), w_t_all.stride(3),
                y.stride(0), y.stride(1), y_offset,
                BM, BN, BK, K % BK == 0, add_input,
                x.dtype == torch.float32 and w_t_all.dtype in (torch.float16, torch.bfloat16))
            return
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), n_seg)
        _sgmv_expand_slice_kernel[grid](
            x, w_t_all, y, N, K, start, length, lora,
            x.stride(0), x.stride(1),
            w_t_all.stride(0), w_t_all.stride(2), w_t_all.stride(3),
            y.stride(0), y.stride(1), y_offset,
            BM, BN, BK, K % BK == 0, add_input,
            x.dtype == torch.float32 and w_t_all.dtype in (torch.float16, torch.bfloat16))


def install_segmented_punica_wrapper(lora_manager) -> bool:
    """Reassign the shared PunicaWrapper's __class__ in place (same technique
    fused_punica_wrapper.py uses; preserves all per-step state).

    The object passed in is the WORKER-level manager; the single shared
    PunicaWrapper lives on its _adapter_manager (LoRAModelManager), which is
    what every LoRA layer holds a reference to.
    """
    if not _VLLM_PUNICA_AVAILABLE:
        return False
    adapter_manager = getattr(lora_manager, "_adapter_manager", None)
    if adapter_manager is None:
        return False
    pw = getattr(adapter_manager, "punica_wrapper", None)
    if pw is None or isinstance(pw, SegmentedPunicaWrapper):
        return False
    pw.__class__ = SegmentedPunicaWrapper
    pw._seg_cap = -1
    pw._seg_dirty = True
    return True
