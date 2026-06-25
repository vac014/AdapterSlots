"""
gwar.py -- GWAR: Generalized Warp Alignment Ratio at variable threshold n*.

GWAR(n*) is the fraction of tokens in contiguous same-adapter segments of size
>= n*. Monotonically decreasing in n*; reduces to WAR at n*=W=32.

Formal definition (kernel_promotion §2.3):
    GWAR(n*) = (# tokens in contiguous same-adapter segments of size >= n*) / N

Properties (Theorem 13.2):
    - GWAR(n*) is monotonically decreasing in n*
    - GWAR(n*) is maximised by adapter-sorted batches at any n*
    - GWAR(W) = WAR  (definitional consistency at warp boundary)
    - GWAR(1) = 1.0  (every token is trivially in a segment of size >= 1)

The theoretical machinery of isolation_experiment–8 (Erlang T_max, Whittle index, PI controller)
controls WAR = GWAR(32). GWAR(n* < 32) is the *implementation metric* that determines
promotion eligibility in the WGKP kernel dispatch path.
"""

from typing import Dict, List

WARP_SIZE = 32


def compute_gwar(adapter_ids: List[int], threshold: int) -> float:
    """GWAR(threshold): fraction of tokens in contiguous same-adapter segments >= threshold.

    O(N) via single linear scan. Assumes the batch is adapter-sorted (AlignmentBuffer
    invariant from kernel_decomposition/4), but is correct for any ordering.

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
    """Compute GWAR at multiple thresholds in one pass for profiling.

    Computes the run-length encoding once, then evaluates each threshold against
    the segment lengths. More efficient than calling compute_gwar() repeatedly.

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
