"""
entropy.py -- H_align (alignment entropy) re-export + convenience helpers.

H_align is computed in war.py alongside WAR and WARτ. This module provides
a focused import surface for callers that only need the entropy metric.
"""

import math
from adapter_slots.metrics.war import compute_halign, WARP_SIZE

__all__ = ["compute_halign", "WARP_SIZE", "halign_upper_bound"]


def halign_upper_bound(K: int) -> float:
    """
    Theoretical upper bound: H_align ≤ log2(K).

    Achieved when every warp has a perfectly uniform distribution across K adapters.

    Args:
        K: Number of distinct adapters in the batch.

    Returns:
        log2(K) in bits. Returns 0.0 for K <= 1.
    """
    if K <= 1:
        return 0.0
    return math.log2(K)
