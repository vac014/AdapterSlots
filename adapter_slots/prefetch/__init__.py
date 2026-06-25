"""
adapter_slots.prefetch -- PredictiveLFU.

Each adapter is scored by the probability that it is requested again within
the load time tau_load, from the EWMA rate the scheduler already tracks:

    score(k) = 1 - exp(-lambda_k * tau_load)

and the warm adapter with the lowest score is evicted.

    PredictivePrefetcher   score, prefetch candidates, eviction target
    WarmCacheManager       warm-set state and hit/miss accounting
    PrefetchPolicy         available policies
    make_prefetcher        policy factory
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
