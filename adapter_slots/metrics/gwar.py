"""
gwar.py -- Generalized Warp Alignment Ratio GWAR(n*) (Definition 1).

Partition a dispatched batch into maximal runs of consecutive same-adapter
tokens. GWAR(n*) is the fraction of tokens in runs of length >= n*:

    GWAR(n*) = (tokens in same-adapter runs of length >= n*) / N

It is non-increasing in n*, equals 1 at n* = 1, and is what the WGKP gate
consumes. WAR counts complete homogeneous warps instead, so the two use
different denominators and are not directly comparable.
"""

from typing import Dict, List


def compute_gwar(adapter_ids: List[int], threshold: int) -> float:
    """GWAR(threshold): fraction of tokens in same-adapter runs of length >= threshold.

    One O(N) scan; correct for any token order.

    Args:
        adapter_ids: List of integer adapter IDs, one per token.
        threshold:   Minimum contiguous segment size n* for promotion eligibility.

    Returns:
        GWAR in [0.0, 1.0]. Returns 0.0 for empty batch.
    """
    n = len(adapter_ids)
    if n == 0 or threshold <= 0:
        return 0.0
    if threshold == 1:
        return 1.0

    promoted_tokens = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and adapter_ids[j] == adapter_ids[i]:
            j += 1
        seg_len = j - i
        if seg_len >= threshold:
            promoted_tokens += seg_len
        i = j

    return promoted_tokens / n


def compute_gwar_curve(
    adapter_ids: List[int],
    thresholds: List[int],
) -> Dict[int, float]:
    """GWAR at several thresholds from one run-length encoding of the batch.

    Args:
        adapter_ids: List of integer adapter IDs, one per token.
        thresholds:  List of threshold values to evaluate.

    Returns:
        Dict mapping threshold -> GWAR value.
    """
    n = len(adapter_ids)
    if n == 0:
        return {t: 0.0 for t in thresholds}

    # Build run-length encoding in a single O(N) pass.
    seg_lengths: List[int] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and adapter_ids[j] == adapter_ids[i]:
            j += 1
        seg_lengths.append(j - i)
        i = j

    result: Dict[int, float] = {}
    for thr in thresholds:
        if thr <= 0:
            result[thr] = 1.0
        elif thr == 1:
            result[thr] = 1.0
        else:
            promoted = sum(s for s in seg_lengths if s >= thr)
            result[thr] = promoted / n

    return result
