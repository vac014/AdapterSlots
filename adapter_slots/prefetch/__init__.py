"""
adapter_slots/prefetch/ -- Predictive adapter prefetching (adapter_prefetching).

Uses the per-adapter EWMA rate estimates already maintained by
ArrivalRateEstimator (erlang_scheduler/6) to predict which adapters will be requested
within the next τ_load seconds and pre-warm them in the GPU LoRA cache.

Public API:
    PredictivePrefetcher   -- Poisson-scored prefetch priority queue
    WarmCacheManager       -- warm/cold cache state + hit/miss accounting
    PrefetchPolicy         -- enum of available policies
    make_prefetcher        -- factory for policy selection

Mathematical grounding (§2, adapter_prefetching.md):
    score(k) = 1 - exp(-λ̂_k × τ_load)   [Poisson prefetch utility]
    evict    = argmin_k score(k)           [LFU with Poisson weights]
"""

from adapter_slots.prefetch.predictor import PredictivePrefetcher
from adapter_slots.prefetch.cache_manager import WarmCacheManager
from adapter_slots.prefetch.policy import PrefetchPolicy, make_prefetcher

__all__ = [
    "PredictivePrefetcher",
    "WarmCacheManager",
    "PrefetchPolicy",
    "make_prefetcher",
]
