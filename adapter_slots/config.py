"""
config.py -- Central configuration for AdapterSlots.

All hyperparameters referenced across implementation phases live here.
Values are the paper's defaults; override via CLI or environment variables.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdapterSlotsConfig:
    # Hardware
    warp_size: int = 32                    # GPU warp width (32 for all NVIDIA)
    gpu_arch: str = "ampere"               # "ampere" | "hopper"

    # Serving
    max_adapters: int = 8                  # Maximum concurrent adapters (K)
    max_batch_tokens: int = 512            # Maximum tokens per dispatch (N)

    # Alignment buffer
    t_max_ms: float = 5.0                  # Global timeout in milliseconds (alignment_buffer)
    war_target: float = 0.85              # WAR* target (erlang_scheduler+)
    ttft_slo_ms: float = 200.0            # TTFT SLO cap for fairness (erlang_scheduler)

    # PI Controller (pi_controller)
    pi_kp: float = 0.01                   # Proportional gain
    pi_ki: float = 0.001                  # Integral gain
    pi_tmax_min_ms: float = 1.0           # Hard lower bound on T_max
    pi_tmax_max_ms: float = 5000.0        # Hard upper bound on T_max (5 s)

    # EWMA rate estimator (erlang_scheduler/6)
    ewma_alpha: float = 0.1               # Smoothing factor for λ_k estimates

    # Workload / experiment
    n_tokens: int = 512                   # Batch size for isolation experiments
    n_runs: int = 100                     # Number of benchmark repetitions
    warmup_runs: int = 10                 # Warmup iterations before measurement
    random_seed: int = 42                 # Global RNG seed

    # Paths
    adapter_dir: str = "./adapters"
    results_dir: str = "./results"
    figures_dir: str = "./figures"

    # Logging
    log_level: str = "INFO"
    prometheus_port: int = 8001
    batch_log_path: Optional[str] = None   # If set, write JSONL batch logs here

    # kernel_promotion -- WGKP + compound stack
    wgkp_threshold: int = 8              # GWAR threshold n* (tokens per segment)
    wgkp_apt_enabled: bool = False       # Enable AdaptivePromoThreshold
    wgkp_hw_profile: str = ""            # Path to E13.6 hardware profile JSON
    mwc_k_hot: int = 5                   # Max adapters in MergedWeightCache
    mwc_memory_budget_gb: float = 10.0  # GPU VRAM budget for merged weights
    mwc_projections: str = "q_proj,k_proj,v_proj,o_proj"  # Comma-sep layer names
    fused_kernel_enabled: bool = False   # Replace SGMV with fused Triton kernel
    macro_n_accum: int = 1               # Iterations to accumulate (1 = disabled)
    adapter_rank: int = 16               # LoRA rank for generated adapters
    apis_enabled: bool = False           # Enable adapter-partitioned routing
    apis_n_gpus: int = 2                 # Number of independent GPU shards
    apis_upstream_urls: str = ""         # Comma-sep vLLM server URLs for APIS
    apis_rebalance_interval_s: float = 30.0  # Seconds between APIS rebalancing

    def as_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
