"""
policy.py -- Prefetch policy registry and factory (adapter_prefetching).

Policies:
    NONE        -- No prefetching; load on demand only (vLLM default baseline)
    TOPK        -- Keep top-K adapters by EWMA rate warm at all times
    LRU         -- LRU eviction (same as vLLM default, explicit baseline)
    PREDICTIVE  -- PredictiveLFU: evict by Poisson score (our policy)

Use make_prefetcher(policy, ...) to instantiate from a string name.
"""

from enum import Enum
from typing import Optional

from adapter_slots.prefetch.cache_manager import WarmCacheManager
from adapter_slots.prefetch.predictor import PredictivePrefetcher


class PrefetchPolicy(str, Enum):
    NONE = "none"
    TOPK = "topk"
    LRU = "lru"
    PREDICTIVE = "predictive"


def make_prefetcher(
    policy: str,
    k_warm_max: int,
    tau_load_ms: float = 200.0,
    p_thresh: float = 0.3,
) -> tuple:
    """Factory: return (WarmCacheManager, PredictivePrefetcher | None).

    Args:
        policy:      One of "none", "topk", "lru", "predictive".
        k_warm_max:  Maximum warm adapters (--max-loras value).
        tau_load_ms: Measured cold-start latency.
        p_thresh:    Prefetch probability threshold (predictive only).

    Returns:
        (cache_manager, predictor_or_none)
    """
    pol = PrefetchPolicy(policy.lower())

    # Map policy → WarmCacheManager eviction policy string
    if pol == PrefetchPolicy.NONE:
        eviction_policy = "lru"  # doesn't matter -- no prefetch triggers
    elif pol == PrefetchPolicy.TOPK:
        eviction_policy = "topk"
    elif pol == PrefetchPolicy.LRU:
        eviction_policy = "lru"
    elif pol == PrefetchPolicy.PREDICTIVE:
        eviction_policy = "predictive"
    else:
        eviction_policy = "lru"

    cache = WarmCacheManager(
        k_warm_max=k_warm_max,
        tau_load_ms=tau_load_ms,
        policy=eviction_policy,
    )

    predictor: Optional[PredictivePrefetcher] = None
    if pol == PrefetchPolicy.PREDICTIVE:
        predictor = PredictivePrefetcher(
            tau_load_ms=tau_load_ms,
            p_thresh=p_thresh,
        )

    return cache, predictor
