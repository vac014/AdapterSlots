"""prefill_lora.py -- prefill LoRA through vLLM's SGMV kernels without the custom-op wrapper.

vLLM 0.6.3 registers sgmv_shrink / sgmv_expand / sgmv_expand_slice as
torch.library custom ops. Prefill runs eagerly (only decode is captured in CUDA
graphs), so every prefill call pays the custom-op dispatch: about 140 us per
call against a 6 us kernel on an A6000, of which ~107 us is the wrapper. A
LLaMA-7B prefill step makes 14 such calls per layer, 448 per step, roughly 60 ms
of host time before the first token of every LoRA request.

The mixin below calls the same Python functions the custom ops wrap, with the
same arguments and kernels, so results are identical; only the dispatch layer
is skipped. AdapterSlots' PunicaWrapper subclasses inherit it.
"""
from __future__ import annotations

from vllm.lora.ops.sgmv_expand import _sgmv_expand
from vllm.lora.ops.sgmv_expand_slice import _sgmv_expand_slice
from vllm.lora.ops.sgmv_shrink import _sgmv_shrink


class DirectPrefillMixin:
    """PunicaWrapper prefill methods that bypass the custom-op dispatch."""

    def shrink_prefill(self, y, x, w_t_all, scale):
        if self.no_lora:
            return
        _sgmv_shrink(x, w_t_all, y, *self.prefill_metadata, scale)

    def expand_prefill(self, y, x, w_t_all, add_input):
        if self.no_lora:
            return
        _sgmv_expand(x, w_t_all, y, *self.prefill_metadata, add_input)

    def expand_slice_prefill(self, y, x, w_t_all, y_offset, y_slice_size, add_input):
        if self.no_lora:
            return
        _sgmv_expand_slice(x, w_t_all, y, *self.prefill_metadata, y_offset, y_slice_size,
                           add_input)
