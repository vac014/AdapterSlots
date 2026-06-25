"""
war.py -- Warp Alignment Ratio (Definition 1).

    WAR(B) = (1/M) * #{ j < M : warp j of B is adapter-homogeneous },  M = floor(N / W)

A warp is the j-th run of W consecutive tokens of the batch; it is homogeneous
when all W tokens belong to one adapter. The partial tail warp is ignored.
"""

from typing import Dict, List

import numpy as np

WARP_SIZE = 32  # Default GPU warp width


def compute_war(batch_tokens: List[Dict], warp_size: int = WARP_SIZE) -> float:
    """
    WAR(B): fraction of complete warps that are adapter-homogeneous.

    Args:
        batch_tokens: List of token dicts, each with key "adapter_id".
        warp_size:    GPU warp width (32 by default).

    Returns:
        WAR in [0.0, 1.0]. Returns 0.0 for empty batches.

    Example (all aligned, W=4):
        >>> tokens = [{"adapter_id": 0}]*4 + [{"adapter_id": 1}]*4
        >>> compute_war(tokens, warp_size=4)
        1.0

    Example (worst case, W=4):
        >>> tokens = [{"adapter_id": i%2} for i in range(8)]
        >>> compute_war(tokens, warp_size=4)
        0.0
    """
    n = len(batch_tokens)
    if n == 0:
        return 0.0

    m = n // warp_size  # Number of complete warps (ignore partial tail)
    if m == 0:
        return 0.0

    aligned = 0
    for j in range(m):
        warp = batch_tokens[j * warp_size: (j + 1) * warp_size]
        adapters = {t["adapter_id"] for t in warp}
        if len(adapters) == 1:
            aligned += 1

    return aligned / m


def compute_war_from_ids(adapter_ids: List[int],
                         warp_size: int = WARP_SIZE) -> float:
    """
    WAR from a flat list of integer adapter IDs (faster than dict version).

    Args:
        adapter_ids: List of adapter IDs, one per token.
        warp_size:   GPU warp width.

    Returns:
        WAR in [0.0, 1.0].
    """
    arr = np.array(adapter_ids, dtype=np.int32)
    n = len(arr)
    m = n // warp_size
    if m == 0:
        return 0.0

    warps = arr[: m * warp_size].reshape(m, warp_size)
    # A warp is aligned iff min == max (all same adapter)
    aligned = int(np.sum(warps.min(axis=1) == warps.max(axis=1)))
    return aligned / m
