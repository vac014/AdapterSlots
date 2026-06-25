"""
adapter_slots.kernel -- WGKP kernels and their vLLM integration.

    segmented_punica_wrapper  grouped LoRA kernels over the adapter segments
                              (AS_SEGGEMM); kernels in tiled_segment_gemm.py and
                              tiled_multislice.py, prefill in prefill_lora.py
    fused_lora_kernel         fused base+LoRA Triton kernel (L2, psi_fuse)
    l3                        merged-weight cache and promoted attention projections (L3)
    wgkp_dispatcher           segments and promotion flags of a dispatched batch
    apt                       run-time choice of n* from the crossover profile
    model_runner              ModelRunner subclass that installs the wrapper
"""

from adapter_slots.kernel.wgkp_dispatcher import WGKPDispatcher, SegmentDescriptor
from adapter_slots.kernel.apt import AdaptivePromoThreshold
from adapter_slots.kernel.l3 import MergedWeightCache
from adapter_slots.kernel.fused_lora_kernel import FusedLoRAKernel

__all__ = [
    "WGKPDispatcher",
    "SegmentDescriptor",
    "AdaptivePromoThreshold",
    "MergedWeightCache",
    "FusedLoRAKernel",
]
