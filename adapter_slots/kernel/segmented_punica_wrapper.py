"""segmented_punica_wrapper.py -- WGKP L2 as a grouped GEMM over the adapter segments.

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

Token-count gate: grouped GEMM only wins above a token-count crossover; below
it the launch overhead dominates and the stock BGMV path runs.
AS_SEGGEMM_MIN_TOKENS sets the gate. With the static grid of mode 1 the
crossover is ~200 tokens on an A6000; the tile-scheduled modes win from about
64 (LLaMA-13B, rank 32, K=16: 8.7 ms of LoRA per decode step at 100 tokens
against 21.2 ms for BGMV, 9.8 ms against 32.9 ms at 160 tokens).
"""
from __future__ import annotations
import os
import torch
import triton

from vllm.lora.ops.sgmv_shrink import _sgmv_shrink_kernel
from vllm.lora.ops.sgmv_expand_slice import _sgmv_expand_slice_kernel
from adapter_slots.kernel.prefill_lora import DirectPrefillMixin
from vllm.lora.punica import PunicaWrapper

_VLLM_PUNICA_AVAILABLE = True

_MIN_TOKENS = int(os.environ.get("AS_SEGGEMM_MIN_TOKENS", "192"))
# 1 = vLLM's stock SGMV on a static (tile x segment) grid
# 2 = tile-scheduled grouped GEMM: grid sized to the useful tiles only
# 3 = mode 2, plus one shrink and one expand launch per packed projection
#     (QKV, gate_up) instead of one pair per slice (tiled_multislice.py)
_MODE = int(os.environ.get("AS_SEGGEMM", "1") or "1")
# Tile height of the mode-2/3 work list; both kernels must use it.
_BM_SHRINK = int(os.environ.get("AS_TILE_BM", "16"))
# Upper bound on tokens in one step (well above vLLM's largest captured
# batch and default prefill budget), used to size the tile work list once.
_TILE_MAP_TOKENS = 8192
# Mode 3 also serves prefill steps whose adapters form contiguous runs: one
# launch per packed projection and no zeroed split-K buffers. A short prefill
# with LoRA is CPU-bound on per-slice launches and buffer fills.
_PREFILL_GROUPED = os.environ.get("AS_SEGGEMM_PREFILL", "1") == "1"


class SegmentedPunicaWrapper(DirectPrefillMixin, PunicaWrapper):
    """PunicaWrapper whose decode path uses grouped SGMV over adapter segments."""

    def update_metadata(self, mapping, lora_index_to_id, max_loras, *args, **kwargs):
        """Recompute segments once per step, eagerly, here and never inside a
        captured graph. The SGMV kernels then only READ these persistent buffers,
        so graph replay picks up each step's values without capturing any of the
        metadata math (same contract vLLM uses for token_lora_indices)."""
        out = super().update_metadata(mapping, lora_index_to_id, max_loras,
                                      *args, **kwargs)
        self._prefill_ok = False
        if not mapping.is_prefill:
            self._refresh_segments(max_loras)
            if _MODE >= 2:
                self._refresh_tiles(max_loras)
        elif _MODE == 3 and _PREFILL_GROUPED:
            # Count adapter runs on the host mapping (no GPU sync). More runs
            # than segment slots means the batch's adapters are not contiguous:
            # keep vLLM's prefill path.
            ids = mapping.index_mapping
            n = len(ids)
            if 0 < n <= _TILE_MAP_TOKENS:
                runs = 1 + sum(1 for a, b in zip(ids, ids[1:]) if a != b)
                if runs <= max_loras:
                    self._refresh_segments(max_loras)
                    self._refresh_tiles(max_loras)
                    self._prefill_ok = True
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
        # Grid size for this step: at most one partial tile per segment on top
        # of the full tiles. Each captured graph replays with the batch size it
        # was captured at, so this matches the grid baked into that graph.
        self._tile_cap = (n_tok + _BM_SHRINK - 1) // _BM_SHRINK + n_loras + 1
        # The work-list buffers are allocated once, at a capacity no batch can
        # exceed: every captured graph holds their addresses, so reallocating
        # them per batch size would leave earlier graphs reading a stale map.
        need = (_TILE_MAP_TOKENS + _BM_SHRINK - 1) // _BM_SHRINK + n_loras + 1
        if getattr(self, "_tile_seg", None) is None or self._tile_seg.numel() < need:
            dev = self._seg_len.device
            self._tile_seg = torch.full((need,), -1, dtype=torch.long, device=dev)
            self._tile_m = torch.zeros(need, dtype=torch.long, device=dev)
        build_tile_map(self._seg_len, _BM_SHRINK, self._tile_seg.numel(),
                       self._tile_seg, self._tile_m)

    def _use_grouped(self, x: torch.Tensor) -> bool:
        return (not self.is_prefill) and x.shape[0] >= _MIN_TOKENS

    def _use_multislice(self, x: torch.Tensor) -> bool:
        if self.is_prefill:
            return getattr(self, "_prefill_ok", False)
        return _MODE == 3 and self._use_grouped(x)

    # Decode overrides
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
        # Block sizes are rank-dependent (vLLM's defaults suit rank 16); the
        # rank-64 and rank-128 settings come from a block-size sweep.
        if N >= 128:
            BM, BN, BK, SK = 16, 32, 64, 4
        elif N >= 64:
            BM, BN, BK, SK = 16, 16, 64, 8
        else:
            BM, BN, BK, SK = 32, 16, 32, 8
        if _MODE >= 2:
            from adapter_slots.kernel.tiled_segment_gemm import _tiled_shrink_kernel
            # The tile map is built in _BM_SHRINK-row tiles, so BM is fixed.
            BN, BK, SK, nw = _tiled_shrink_cfg(N)
            g = (self._tile_cap, triton.cdiv(N, BN), SK)
            _tiled_shrink_kernel[g](
                x, w_t_all, y, N, K, start, length, lora,
                self._tile_seg, self._tile_m, scale,
                x.stride(0), x.stride(1),
                w_t_all.stride(0), w_t_all.stride(2), w_t_all.stride(3),
                y.stride(0), y.stride(1), _BM_SHRINK, BN, BK, K % (BK * SK) == 0, SK,
                num_warps=nw, num_stages=1)
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
        if _MODE >= 2:
            from adapter_slots.kernel.tiled_segment_gemm import _tiled_expand_slice_kernel
            BN, BK, nw = _tiled_expand_cfg(K)
            g = (self._tile_cap, triton.cdiv(N, BN))
            _tiled_expand_slice_kernel[g](
                x, w_t_all, y, N, K, start, length, lora,
                self._tile_seg, self._tile_m,
                x.stride(0), x.stride(1),
                w_t_all.stride(0), w_t_all.stride(2), w_t_all.stride(3),
                y.stride(0), y.stride(1), y_offset,
                _BM_SHRINK, BN, BK, K % BK == 0, add_input,
                x.dtype == torch.float32 and w_t_all.dtype in (torch.float16, torch.bfloat16),
                num_warps=nw, num_stages=1)
            return
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), n_seg)
        _sgmv_expand_slice_kernel[grid](
            x, w_t_all, y, N, K, start, length, lora,
            x.stride(0), x.stride(1),
            w_t_all.stride(0), w_t_all.stride(2), w_t_all.stride(3),
            y.stride(0), y.stride(1), y_offset,
            BM, BN, BK, K % BK == 0, add_input,
            x.dtype == torch.float32 and w_t_all.dtype in (torch.float16, torch.bfloat16))


    def add_lora_packed_nslice(self, y, x, lora_a_stacked, lora_b_stacked,
                               scale, output_slices):
        x2 = x.view(-1, x.shape[-1])
        a0, b0 = lora_a_stacked[0], lora_b_stacked[0]
        if (not self._use_multislice(x2)
                or getattr(self, "_seg_cap", -1) <= 0
                or not 1 <= len(output_slices) <= 3
                or len(set(output_slices)) != 1
                or any(t.stride() != a0.stride() for t in lora_a_stacked)
                or any(t.stride() != b0.stride() for t in lora_b_stacked)):
            return super().add_lora_packed_nslice(y, x, lora_a_stacked, lora_b_stacked,
                                                  scale, output_slices)
        from adapter_slots.kernel.tiled_multislice import (
            _tiled_shrink_ms_kernel, _tiled_expand_ms_kernel, slice_offsets)
        y2 = y.view(-1, y.shape[-1])
        ns, r, n_out = len(output_slices), b0.shape[-1], output_slices[0]
        M, K = x2.shape
        start, length, lora = self._seg_start, self._seg_len, self._seg_lora
        wa = slice_offsets(lora_a_stacked)
        wb = slice_offsets(lora_b_stacked)
        BN, BK, SK, nw = _tiled_shrink_cfg(r)
        if self.is_prefill:
            SK = 1          # long segments give enough parallelism; no zeroed buffer
        buf = (torch.zeros if SK > 1 else torch.empty)(
            (M, ns * r), dtype=torch.float32, device=x2.device)
        _tiled_shrink_ms_kernel[(self._tile_cap, triton.cdiv(r, BN), ns * SK)](
            x2, a0, buf, r, K, start, length, lora,
            self._tile_seg, self._tile_m, scale, wa[1], wa[2],
            x2.stride(0), x2.stride(1), a0.stride(0), a0.stride(2), a0.stride(3),
            buf.stride(0), buf.stride(1), _BM_SHRINK, BN, BK, K % (BK * SK) == 0, SK,
            num_warps=nw, num_stages=1)
        BN2, BK2, nw2 = _tiled_expand_cfg(r)
        yo = [i * n_out for i in range(ns)] + [0] * (3 - ns)
        _tiled_expand_ms_kernel[(self._tile_cap, triton.cdiv(n_out, BN2), ns)](
            buf, b0, y2, n_out, r, start, length, lora,
            self._tile_seg, self._tile_m, wb[1], wb[2], yo[0], yo[1], yo[2],
            buf.stride(0), buf.stride(1), b0.stride(0), b0.stride(2), b0.stride(3),
            y2.stride(0), y2.stride(1), _BM_SHRINK, BN2, BK2, r % BK2 == 0, True, True,
            num_warps=nw2, num_stages=1)

    def add_lora(self, y, x, wa_t_all, wb_t_all, scale, y_offset=None,
                 y_slice_size=None, *, buffer=None):
        if (y_offset in (None, 0) and buffer is None
                and (y_slice_size is None or y_slice_size == wb_t_all.shape[-2])
                and self._use_multislice(x.view(-1, x.shape[-1]))):
            return self.add_lora_packed_nslice(y, x, (wa_t_all,), (wb_t_all,), scale,
                                               (wb_t_all.shape[-2],))
        return super().add_lora(y, x, wa_t_all, wb_t_all, scale, y_offset,
                                y_slice_size, buffer=buffer)


def _tiled_shrink_cfg(rank):
    """(BLOCK_N, BLOCK_K, SPLIT_K, num_warps) of the tile-scheduled shrink.

    From the block-size sweep in benchmarks/paper/kernel_tiles.py at LLaMA-13B
    decode shapes (K=16, 100-160 tokens, 16-row tiles).
    """
    if rank >= 128:
        return 32, 64, 4, 4
    if rank >= 64:
        return 16, 64, 8, 4
    return 16, 64, 16, 4


def _tiled_expand_cfg(rank):
    """(BLOCK_N, BLOCK_K, num_warps) of the tile-scheduled expand, from the same
    sweep."""
    if rank >= 128:
        return 64, 32, 4
    return 128, min(rank, 32), 4


def install_segmented_punica_wrapper(lora_manager) -> bool:
    """Reassign the shared PunicaWrapper's __class__ in place, which keeps all
    of its per-step state.

    vLLM's LoRAModelManager builds one PunicaWrapper that every LoRA layer
    references, so the instance itself is the integration point. The object
    passed in is the worker-level manager; the shared wrapper lives on its
    _adapter_manager.
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
