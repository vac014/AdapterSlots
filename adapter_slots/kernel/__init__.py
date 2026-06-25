"""
adapter_slots.kernel -- WGKP kernel dispatch stack (kernel_promotion).

Modules:
    fused_lora_kernel        FusedLoRAKernel -- Triton fused base+LoRA (Level 2,
                             base-GEMM-fusing; kept for reference, NOT installed
                             by default -- see fused_lora_layers.py's docstring
                             ).
    fused_packed_lora_kernel FusedPackedLoRAKernel -- graph-safe fusion of
                             vLLM's add_lora_packed_nslice (QKV/gate-up).
                             This is the kernel actually installed by default
                             (via fused_punica_wrapper.py). See kernel.md §2.1/§10.
    fused_punica_wrapper     FusedPunicaWrapper -- PunicaWrapper subclass that
                             installs the above at the real serving call site.
    merged_weight_cache      MergedWeightCache -- BA product cache + eviction (Level 3)
    wgkp_dispatcher          WGKPDispatcher + SegmentDescriptor
    apt                      AdaptivePromoThreshold -- runtime n* selection
    model_runner             AlignmentAwareModelRunner -- vLLM ModelRunner subclass
"""

from adapter_slots.kernel.wgkp_dispatcher import WGKPDispatcher, SegmentDescriptor
from adapter_slots.kernel.apt import AdaptivePromoThreshold
from adapter_slots.kernel.merged_weight_cache import MergedWeightCache
from adapter_slots.kernel.fused_lora_kernel import FusedLoRAKernel
from adapter_slots.kernel.fused_packed_lora_kernel import FusedPackedLoRAKernel
from adapter_slots.kernel.fused_punica_wrapper import FusedPunicaWrapper

__all__ = [
    "WGKPDispatcher",
    "SegmentDescriptor",
    "AdaptivePromoThreshold",
    "MergedWeightCache",
    "FusedLoRAKernel",
    "FusedPackedLoRAKernel",
    "FusedPunicaWrapper",
]
