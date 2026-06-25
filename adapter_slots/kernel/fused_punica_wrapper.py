"""
fused_punica_wrapper.py -- installs FusedPackedLoRAKernel (fused_packed_lora_kernel.py)
at vLLM's PunicaWrapper.add_lora_packed_nslice call site.

Why a PunicaWrapper subclass (not a fresh instance, not the per-layer
__class__ swap pattern fused_lora_layers.py uses):

LoRAModelManager.__init__ constructs exactly one PunicaWrapper instance
(vllm/lora/models.py:323) and every LoRA layer holds a reference to that
SAME instance (models.py:495's comment: "All lora layers share the same
punica_wrapper based on reference."). There is nothing to walk/replace at
the layer level for this fix -- the integration point is the wrapper
instance itself. Reassigning its __class__ in place (the same technique
fused_lora_layers.py already uses for layer instances) preserves all of
its already-initialized state (token_lora_indices, prefill_metadata,
no_lora, etc., set per-step by update_metadata()) with zero copying and
adds no new __init__-time state of its own -- FusedPunicaWrapper only
overrides one method.

Why overriding add_lora_packed_nslice is graph-safe even though it branches
on self.is_prefill and len(output_slices):

Neither of those is a per-token data value. self.is_prefill is constant
for the whole duration of a captured decode graph (vLLM only captures
decode graphs; is_prefill is always False during capture and replay) and
len(output_slices) is determined by which static layer is calling (QKV
always passes 3, gate_up always passes 2, never data-dependent) -- so for a
given installed layer in a given captured graph, exactly one branch is ever
taken, every single call. This is different in kind from the OLD
fused_lora_layers.py wiring's branch, which depended on
punica_wrapper.token_lora_indices' *contents* (a per-step tensor value) to
decide how many kernel launches to issue -- that's the actual thing CUDA
graph capture forbids, not branching per se.
"""

from __future__ import annotations

from typing import Tuple

import torch

from adapter_slots.kernel.fused_packed_lora_kernel import FusedPackedLoRAKernel

try:
    from vllm.lora.punica import PunicaWrapper
    _VLLM_PUNICA_AVAILABLE = True
except ImportError:
    _VLLM_PUNICA_AVAILABLE = False
    PunicaWrapper = object  # type: ignore[assignment,misc]

_kernel = FusedPackedLoRAKernel()


class FusedPunicaWrapper(PunicaWrapper):
    """Drop-in replacement for PunicaWrapper that fuses add_lora_packed_nslice's
    per-slice loop into 2 kernel launches (decode only). See module docstring.
    """

    def add_lora_packed_nslice(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: Tuple[torch.Tensor, ...],
        lora_b_stacked: Tuple[torch.Tensor, ...],
        scale: float,
        output_slices: Tuple[int, ...],
    ) -> None:
        if (self.is_prefill
                or not (2 <= len(output_slices) <= 3)
                or not _kernel.is_available()
                or not x.is_cuda):
            return super().add_lora_packed_nslice(
                y, x, lora_a_stacked, lora_b_stacked, scale, output_slices)

        y_org = y
        x_flat = x.view(-1, x.shape[-1])
        y_flat = y.view(-1, y.shape[-1])
        try:
            _kernel.apply_packed_nslice(
                y_flat, x_flat, lora_a_stacked, lora_b_stacked, scale,
                output_slices, self.token_lora_indices,
            )
        except AssertionError:
            # Defensive only: an unmet precondition (e.g. mismatched ranks
            # across slices) falls back to the always-correct stock path
            # rather than risk a silent wrong-output bug.
            super().add_lora_packed_nslice(
                y_org, x, lora_a_stacked, lora_b_stacked, scale, output_slices)
            return
        y = y_flat.view_as(y_org)  # noqa: F841 (view_as for parity with stock method; y_flat already aliases y_org's storage)


def install_fused_punica_wrapper(lora_manager) -> bool:
    """Reassign the shared PunicaWrapper instance's __class__ to FusedPunicaWrapper.

    `lora_manager` is the ModelRunner's self.lora_manager
    (LRUCacheWorkerLoRAManager). Returns False (no-op, not an error) if vLLM
    LoRA support isn't available, lora_manager is None, or the wrapper was
    already swapped by a previous call.
    """
    if not _VLLM_PUNICA_AVAILABLE or lora_manager is None:
        return False
    adapter_manager = getattr(lora_manager, "_adapter_manager", None)
    if adapter_manager is None:
        return False
    wrapper = getattr(adapter_manager, "punica_wrapper", None)
    if wrapper is None or isinstance(wrapper, FusedPunicaWrapper):
        return False
    wrapper.__class__ = FusedPunicaWrapper
    return True
