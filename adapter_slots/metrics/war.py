"""
war.py -- WAR, WARτ, and H_align metric computation.

Definitions (from the paper):

    WAR(B) = (1/M) * Σ_j  𝟙[warp j is adapter-homogeneous]

        M = ⌊N / W⌋   (number of complete warps)
        W = 32         (warp size)
        A warp is homogeneous iff all W tokens in it have the same adapter_id.

    WARτ(B) = mean misalignment age across all adapter queues
        (average time tokens have waited in an adapter queue before dispatch)

    H_align(B) = -(1/M) * Σ_j Σ_k p_{j,k} * log2(p_{j,k})
        where p_{j,k} = fraction of tokens in warp j belonging to adapter k
        H_align = 0 iff WAR = 1 (perfect alignment)
        H_align = log2(K) iff tokens are perfectly uniform across K adapters

All functions accept a list of token dicts:
    {"adapter_id": int, "arrival_time_ms": float (optional), ...}
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

WARP_SIZE = 32  # Default GPU warp width


# WAR

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


# WARτ

def compute_wartau(batch_tokens: List[Dict],
                   dispatch_time_ms: float) -> float:
    """
    WARτ(B): mean misalignment age across all tokens in the batch.

    Misalignment age of token i = dispatch_time_ms - arrival_time_ms[i].
    This measures how long each token waited in the alignment buffer.

    Args:
        batch_tokens:     List of token dicts with "arrival_time_ms" key.
        dispatch_time_ms: Wall-clock time (ms) when batch was dispatched.

    Returns:
        Mean wait time in milliseconds. Returns 0.0 if no arrival times present.
    """
    ages = []
    for t in batch_tokens:
        if "arrival_time_ms" in t and t["arrival_time_ms"] is not None:
            age = dispatch_time_ms - t["arrival_time_ms"]
            ages.append(max(0.0, age))

    if not ages:
        return 0.0
    return float(np.mean(ages))


def compute_wartau_per_adapter(batch_tokens: List[Dict],
                               dispatch_time_ms: float) -> Dict[int, float]:
    """
    WARτ broken down per adapter_id.

    Returns:
        Dict mapping adapter_id -> mean wait time (ms) for tokens of that adapter.
    """
    from collections import defaultdict
    per_adapter: Dict[int, List[float]] = defaultdict(list)

    for t in batch_tokens:
        aid = t["adapter_id"]
        if "arrival_time_ms" in t and t["arrival_time_ms"] is not None:
            age = max(0.0, dispatch_time_ms - t["arrival_time_ms"])
            per_adapter[aid].append(age)

    return {aid: float(np.mean(ages)) for aid, ages in per_adapter.items()}


# H_align

def compute_halign(batch_tokens: List[Dict],
                   warp_size: int = WARP_SIZE) -> float:
    """
    H_align(B): mean Shannon entropy of adapter distributions across warps.

    For a perfectly aligned batch: H_align = 0.
    For a uniformly mixed batch with K adapters: H_align → log2(K).

    Args:
        batch_tokens: List of token dicts with "adapter_id" key.
        warp_size:    GPU warp width.

    Returns:
        Mean per-warp entropy in bits. Returns 0.0 for empty / single-warp batches.
    """
    n = len(batch_tokens)
    m = n // warp_size
    if m == 0:
        return 0.0

    total_entropy = 0.0
    for j in range(m):
        warp = batch_tokens[j * warp_size: (j + 1) * warp_size]
        counts: Dict[int, int] = {}
        for t in warp:
            aid = t["adapter_id"]
            counts[aid] = counts.get(aid, 0) + 1

        entropy = 0.0
        for cnt in counts.values():
            p = cnt / warp_size
            if p > 0:
                entropy -= p * math.log2(p)

        total_entropy += entropy

    return total_entropy / m


# Combined metric dict

def compute_all_metrics(batch_tokens: List[Dict],
                        dispatch_time_ms: Optional[float] = None,
                        warp_size: int = WARP_SIZE) -> Dict[str, float]:
    """
    Compute WAR, WARτ, and H_align in one pass.

    Args:
        batch_tokens:     Token list with "adapter_id" (and optionally "arrival_time_ms").
        dispatch_time_ms: Wall-clock dispatch time (ms). Required for WARτ; pass None
                          to skip WARτ computation.
        warp_size:        GPU warp width.

    Returns:
        Dict with keys: "war", "wartau_ms", "halign", "n_tokens", "n_warps", "n_adapters".
    """
    n = len(batch_tokens)
    m = n // warp_size
    adapter_ids = list({t["adapter_id"] for t in batch_tokens})

    war = compute_war(batch_tokens, warp_size)
    halign = compute_halign(batch_tokens, warp_size)

    if dispatch_time_ms is not None:
        wartau = compute_wartau(batch_tokens, dispatch_time_ms)
    else:
        wartau = float("nan")

    return {
        "war": war,
        "wartau_ms": wartau,
        "halign": halign,
        "n_tokens": n,
        "n_warps": m,
        "n_adapters": len(adapter_ids),
    }


# Warp-level breakdown

def warp_alignment_breakdown(batch_tokens: List[Dict],
                              warp_size: int = WARP_SIZE) -> List[Dict]:
    """
    Return per-warp alignment details for debugging / visualization.

    Returns:
        List of dicts (one per complete warp) with keys:
            warp_idx, adapter_counts, is_aligned, entropy
    """
    n = len(batch_tokens)
    m = n // warp_size
    result = []

    for j in range(m):
        warp = batch_tokens[j * warp_size: (j + 1) * warp_size]
        counts: Dict[int, int] = {}
        for t in warp:
            aid = t["adapter_id"]
            counts[aid] = counts.get(aid, 0) + 1

        is_aligned = len(counts) == 1
        entropy = 0.0
        for cnt in counts.values():
            p = cnt / warp_size
            if p > 0:
                entropy -= p * math.log2(p)

        result.append({
            "warp_idx": j,
            "adapter_counts": dict(counts),
            "is_aligned": is_aligned,
            "entropy": entropy,
        })

    return result


# Theoretical WAR under random mixing

def theoretical_war_random(n: int, k: int, warp_size: int = WARP_SIZE,
                            adapter_probs: Optional[List[float]] = None) -> float:
    """
    Theoretical expected WAR under independent random token-to-adapter assignment.

    Under uniform assignment (adapter_probs = None):
        P(warp is aligned) = K * (1/K)^W = K^(1-W)

    For non-uniform assignment:
        P(warp is aligned) = Σ_k p_k^W

    This is the floor WAR with no alignment buffer.

    Args:
        n:             Batch size (number of tokens).
        k:             Number of adapters.
        warp_size:     GPU warp width.
        adapter_probs: Per-adapter arrival probabilities (must sum to 1).
                       If None, uniform is assumed.

    Returns:
        Expected WAR in [0.0, 1.0].
    """
    if adapter_probs is None:
        adapter_probs = [1.0 / k] * k

    p_aligned = sum(p ** warp_size for p in adapter_probs)
    return p_aligned
