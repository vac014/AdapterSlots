"""
fused_lora_layers.py -- real wiring of the Level-2 fused Triton kernel
(fused_lora_kernel.py) into vLLM's actual LoRA forward computation.

STATUS: kept for reference, NOT installed by default, and model_runner.py's
load_model() does not call install_fused_lora_layers(). The dispatch loop here is
data-dependent Python, so it is only CUDA-graph-legal under --enforce-eager, and that
disables graph capture for the ENTIRE model: measured end-to-end at K=2,4,8, it costs
3.5x-10x more than any LoRA-only fusion can save. The path that is installed
(adapter_slots/kernel/fused_punica_wrapper.py + fused_packed_lora_kernel.py) fuses
only the LoRA shrink/expand launches across packed slices on a fixed,
CUDA-graph-safe grid: never the base GEMM, never a data-dependent launch count.

Why this file exists:

FusedLoRAKernel has existed since kernel_promotion with its own standalone
correctness/speed validation (scripts/experiments/e13_crossover_benchmark.py, gate
psi_fuse >= 1.25), but AlignmentAwareModelRunner.execute_model()
(model_runner.py) never called it -- it was a pure passthrough to
super().execute_model(). This file gives it a real call site for the first
time: vLLM's LoRA-enabled linear layers (vllm/lora/layers.py) compute
Y = X @ W^T + alpha * (X @ A^T) @ B^T inside apply() via two separate
kernel launches (punica_wrapper.add_lora, the SGMV/BGMV path). These
subclasses override apply() to call FusedLoRAKernel.forward() instead,
one launch, X loaded once, the LoRA intermediate kept in registers.

Why grouping by contiguous run is (usually) free here:

AlignmentAwareScheduler sorts decode-phase scheduled_seq_groups by adapter
id when AS_MODE=wgkp (adapter_slots/integrations/vllm_scheduler.py,
post_alignment_groups.sort(key=lambda s: self._adapter_id_of(s.seq_group))).
vLLM's input-tensor builder consumes scheduled_seq_groups in that order, so
by the time X reaches each linear layer's apply(), same-adapter rows are
already contiguous under WGKP -- no extra gather/scatter needed to find
runs, just a linear scan over punica_wrapper.token_lora_indices. Under
non-WGKP modes this still works (falls back to one run per index change,
i.e. effectively per-token), just without the free contiguity win.

Installation: AlignmentAwareModelRunner.load_model() reassigns each
installed LoRA layer instance's __class__ to the matching Fused* subclass
below, in place, after vLLM's own LoRAModelManager has already wrapped the
model's linear layers (see model_runner.py). Reassigning __class__ in place
(not reconstructing via Fused*(base_layer)) is deliberate: these subclasses
add no new __init__ state and don't override create_lora_weights, so
reconstruction would needlessly re-run create_lora_weights and zero out
already-loaded lora_a_stacked/lora_b_stacked tensors. This is "subclass,
don't monkeypatch" applied to an existing instance rather than a fresh one --
no vLLM module attribute, registry, or method is ever reassigned.

Gated by the existing AS_FUSED_KERNEL env var (FusedLoRAKernel.is_available()
already reads it) -- no new env var. Falls back to the stock SGMV apply()
whenever Triton is unavailable, AS_FUSED_KERNEL=0, or the batch shape looks
unexpected (CUDA-graph padding mismatch) -- always correct, just not fused.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from adapter_slots.kernel.fused_lora_kernel import FusedLoRAKernel

try:
    from vllm.lora.layers import (
        ColumnParallelLinearWithLoRA,
        MergedColumnParallelLinearWithLoRA,
        RowParallelLinearWithLoRA,
    )
    _VLLM_LORA_AVAILABLE = True
except ImportError:
    _VLLM_LORA_AVAILABLE = False
    ColumnParallelLinearWithLoRA = object  # type: ignore[assignment,misc]
    MergedColumnParallelLinearWithLoRA = object  # type: ignore[assignment,misc]
    RowParallelLinearWithLoRA = object  # type: ignore[assignment,misc]

_kernel = FusedLoRAKernel()


def _fused_path_usable(x: torch.Tensor) -> bool:
    """Per-run Python dispatch (variable launch count, data-dependent trip
    count) cannot be captured into a CUDA graph -- torch.cuda.graph() requires
    a static kernel-launch sequence. vLLM captures one graph per decode
    batch-size bucket and replays it afterward; trying to branch on tensor
    *values* (token_lora_indices) during capture raises "operation not
    permitted when stream is capturing". Servers running the fused path must
    pass --enforce-eager (see backend_adapterslots.py) so capture never happens in
    the first place; this check is a defensive fallback in case it's enabled
    anyway, so the worst case is silently losing the fusion benefit for that
    call, never a crash.
    """
    return x.is_cuda and not torch.cuda.is_current_stream_capturing()


def _contiguous_runs(indices: torch.Tensor) -> List[Tuple[int, int, int]]:
    """Group per-token LoRA slot indices into contiguous (start, end, idx) runs.

    O(M) Python loop over a 1D index tensor -- fine at decode-batch scale
    (tens to low hundreds of tokens); not used in any code path where M is
    large enough for this to matter relative to the GEMM it precedes.
    """
    idx = indices.tolist()
    runs: List[Tuple[int, int, int]] = []
    n = len(idx)
    i = 0
    while i < n:
        j = i + 1
        while j < n and idx[j] == idx[i]:
            j += 1
        runs.append((i, j, idx[i]))
        i = j
    return runs


def _dispatch_run(
    x_run: torch.Tensor,
    w_slice: torch.Tensor,
    bias_slice: Optional[torch.Tensor],
    lora_idx: int,
    lora_a_stacked: torch.Tensor,
    lora_b_stacked: torch.Tensor,
) -> torch.Tensor:
    """Compute one contiguous run's output: base GEMM, +LoRA delta if lora_idx>=0.

    lora_idx < 0 means "no adapter for these tokens" (vLLM convention,
    PunicaWrapper.token_lora_indices) -- must not index lora_a_stacked/
    lora_b_stacked with it (negative index would silently wrap to the last
    slot instead of skipping LoRA).
    """
    if lora_idx < 0:
        return F.linear(x_run, w_slice, bias_slice)
    A = lora_a_stacked[lora_idx, 0]
    B = lora_b_stacked[lora_idx, 0]
    y = _kernel.forward(x_run, w_slice, A, B, 1.0).to(x_run.dtype)
    if bias_slice is not None:
        y = y + bias_slice
    return y


class FusedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    """Level-2 apply() for q_proj/k_proj/v_proj/o_proj-shaped LoRA layers."""

    def apply(self, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        if not _kernel.is_available() or not _fused_path_usable(x):
            return super().apply(x, bias)
        indices = self.punica_wrapper.token_lora_indices
        if indices.shape[0] != x.shape[0]:
            return super().apply(x, bias)
        W = self.base_layer.weight
        out = torch.empty(x.shape[0], self.output_size,
                          dtype=x.dtype, device=x.device)
        for i, j, lora_idx in _contiguous_runs(indices):
            out[i:j] = _dispatch_run(x[i:j], W, bias, lora_idx,
                                     self.lora_a_stacked, self.lora_b_stacked)
        return out


class FusedRowParallelLinearWithLoRA(RowParallelLinearWithLoRA):
    """Level-2 apply() for o_proj/down_proj-shaped (row-parallel) LoRA layers.

    RowParallelLinearWithLoRA.apply() takes no bias argument -- vLLM adds
    bias after the tensor-parallel all-reduce in forward(), not inside
    apply() (see vllm/lora/layers.py).
    """

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if not _kernel.is_available() or not _fused_path_usable(x):
            return super().apply(x)
        indices = self.punica_wrapper.token_lora_indices
        if indices.shape[0] != x.shape[0]:
            return super().apply(x)
        W = self.base_layer.weight
        out = torch.empty(x.shape[0], self.output_size,
                          dtype=x.dtype, device=x.device)
        for i, j, lora_idx in _contiguous_runs(indices):
            out[i:j] = _dispatch_run(x[i:j], W, None, lora_idx,
                                     self.lora_a_stacked, self.lora_b_stacked)
        return out


class FusedMergedColumnParallelLinearWithLoRA(MergedColumnParallelLinearWithLoRA):
    """Level-2 apply() for gate_up_proj-shaped (2-slice packed) LoRA layers.

    Each slice's output columns are disjoint (first half / second half), so
    dispatching the two slices as two separate FusedLoRAKernel calls over
    disjoint W column-ranges does the same total GEMM work as one full-width
    call would -- no redundant FLOPs, just two launches instead of one.
    """

    def apply(self, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        if not _kernel.is_available() or not _fused_path_usable(x):
            return super().apply(x, bias)
        indices = self.punica_wrapper.token_lora_indices
        if indices.shape[0] != x.shape[0]:
            return super().apply(x, bias)
        W = self.base_layer.weight
        half = self.output_dim
        out = torch.empty(x.shape[0], self.output_size,
                          dtype=x.dtype, device=x.device)
        bias0 = bias[:half] if bias is not None else None
        bias1 = bias[half:] if bias is not None else None
        for i, j, lora_idx in _contiguous_runs(indices):
            out[i:j, :half] = _dispatch_run(
                x[i:j], W[:half], bias0, lora_idx,
                self.lora_a_stacked[0], self.lora_b_stacked[0])
            out[i:j, half:] = _dispatch_run(
                x[i:j], W[half:], bias1, lora_idx,
                self.lora_a_stacked[1], self.lora_b_stacked[1])
        return out


# (stock class) -> (fused subclass) -- used by model_runner.py's
# AlignmentAwareModelRunner.load_model() to reassign installed layer
# instances' __class__ in place.
FUSED_CLASS_MAP = {
    ColumnParallelLinearWithLoRA: FusedColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA: FusedRowParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA: FusedMergedColumnParallelLinearWithLoRA,
}


def install_fused_lora_layers(model: torch.nn.Module) -> int:
    """Reassign every installed stock LoRA layer's __class__ to its Fused*
    counterpart, in place. Must run after vLLM's LoRAModelManager has already
    wrapped the model's linear layers (i.e. after ModelRunner.load_model()'s
    super() call) -- relies on lora_a_stacked/lora_b_stacked already existing
    as instance attributes (set by create_lora_weights(), not touched here).

    Returns the number of layers swapped (0 if vLLM/LoRA unavailable or no
    matching layers found -- both are valid, non-error states).
    """
    if not _VLLM_LORA_AVAILABLE:
        return 0
    n = 0
    for module in model.modules():
        # Exact type match only (module.__class__ is already exactly one of
        # the stock classes here, never a third-party subclass of them) --
        # isinstance would also match a layer we already swapped on a
        # previous call, which is harmless but wasteful to re-check.
        fused_cls = FUSED_CLASS_MAP.get(type(module))
        if fused_cls is not None:
            module.__class__ = fused_cls
            n += 1
    return n
