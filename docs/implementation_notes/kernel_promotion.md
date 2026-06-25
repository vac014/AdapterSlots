# WAR-Gated Kernel Promotion (WGKP)

When the alignment buffer produces WAR=1 segments (all N tokens one adapter), the
default SGMV path runs two extra kernels per layer (shrink X·Aᵀ, expand H·Bᵀ) plus the
base GEMM. For a uniform segment these can be promoted to a single merged-weight GEMM
(base + α·B·A folded), removing the LoRA launches.

Promotion is gated on WAR so it fires only when alignment makes it correct and
profitable, and it compounds with the dispatch stack (Theorems 13.1, 13.2).

- Promotion dispatcher: `adapter_slots/kernel/wgkp_dispatcher.py`
- Fused / merged kernels: `adapter_slots/kernel/{fused_lora_kernel,merged_weight_cache}.py`
- Microbench: `benchmarks/micro/m2_kernel_promotion.py`
