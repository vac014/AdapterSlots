"""
erlang.py -- Per-adapter optimal T_max computation via Erlang CDF inversion.

Implements Theorem 5.3:

    T_max^(k)* = F_k^{-1}(WAR*)
               = Erlang_CDF_inv(W=32, λ_k, quantile=WAR*)

The fill time T_k for adapter k (time to collect W tokens from a Poisson
arrival stream at rate λ_k) follows Erlang(W, λ_k) distribution.

Setting T_max^(k)* to the WAR*-quantile of this distribution guarantees:
    P(adapter k fills a complete warp within T_max^(k)*) = WAR*

Fairness constraint (Theorem 5.2):
    T_max^(k)* = min(F_k^{-1}(WAR*), TTFT_SLO)

Without this cap, rare adapters (small λ_k) accumulate astronomically large
T_max values, causing TTFT SLO violations.

TP-transparency: this module is pure CPU-side Python. The computed T_max^(k)*
values are passed to AlignmentBuffer.form_batch_erlang(), which applies them
at the scheduling layer before TP sharding. No GPU interaction in this module.

References:
    - erlang_scheduler.md §3.1, §8.1
    - Theorem 5.3, Theorem 5.2, Corollary 5.4
"""

from functools import lru_cache

import numpy as np
from scipy.special import gammainc, gammaincinv
from scipy.stats import erlang


@lru_cache(maxsize=256)
def _gamma_ppf_unit_scale(warp_size: int, war_target: float) -> float:
    """Cache gammaincinv(warp_size, war_target) -- the part of Erlang.ppf()
    that does NOT depend on lambda_k.

    Erlang(a=warp_size, scale=1/lambda_k).ppf(q) == gammaincinv(warp_size, q) / lambda_k
    (Gamma scaling property: X ~ Gamma(a, theta) => X = theta * Y, Y ~ Gamma(a, 1),
    so ppf(q) = theta * Gamma(a, 1).ppf(q) = theta * gammaincinv(a, q)).

    warp_size and war_target are fixed for the lifetime of a scheduler instance,
    so this collapses what would be one scipy.stats.erlang.ppf() call per adapter
    per tick (~0.1ms each due to rv_continuous generic-dispatch overhead) into a
    single cached scalar plus a division. Verified bit-exact against
    scipy.stats.erlang.ppf for a range of lambda_k.
    """
    return float(gammaincinv(warp_size, war_target))


def compute_tmax_erlang(
    warp_size: int,
    lambda_k: float,
    war_target: float,
    ttft_slo_ms: float = 2000.0,
) -> float:
    """Compute optimal per-adapter T_max via Erlang CDF inversion (Theorem 5.3).

    T_max^(k)* = Erlang_CDF_inv(W, lambda_k, war_target)

    The fill time T_k ~ Erlang(k=warp_size, scale=1/lambda_k) under Poisson
    arrivals. Setting T_max to the WAR*-quantile of this distribution means
    adapter k fills a complete warp within T_max with probability WAR*.

    The fairness cap (Theorem 5.2) prevents rare adapters from waiting forever:
    if the unconstrained T_max exceeds the TTFT SLO, it is clamped to TTFT_SLO.

    Args:
        warp_size:    GPU warp width W (32 for all NVIDIA hardware).
        lambda_k:     Estimated arrival rate for adapter k (tokens/sec).
                      Must be non-negative. Zero or negative → return TTFT SLO.
        war_target:   Target WAR* in (0, 1), e.g. 0.8 means 80% of dispatches
                      should be full warp-aligned.
        ttft_slo_ms:  TTFT SLO in milliseconds used as the fairness cap.
                      Default: 2000 ms (2 seconds).

    Returns:
        T_max^(k)* in seconds, in [0, ttft_slo_ms/1000].
    """
    if lambda_k <= 0.0:
        # No traffic → use fairness cap so the adapter is not starved
        return ttft_slo_ms / 1000.0

    # Erlang(k=warp_size, scale=1/lambda_k): sum of warp_size i.i.d. Exp(lambda_k)
    # ppf = percent-point function = inverse CDF. The (warp_size, war_target) part
    # is cached since it's invariant across adapters/ticks; only the division by
    # lambda_k varies.
    t_optimal = _gamma_ppf_unit_scale(warp_size, war_target) / lambda_k

    # Fairness constraint (Theorem 5.2): cap at TTFT SLO
    t_max = min(float(t_optimal), ttft_slo_ms / 1000.0)

    return t_max


def compute_tmax_erlang_batch(
    warp_size: int,
    lambda_k_dict: dict,
    war_target: float,
    ttft_slo_ms: float = 2000.0,
) -> dict:
    """Compute T_max^(k)* for all adapters in a single call.

    Vectorised over adapters but serialised per-adapter (each adapter has its
    own λ_k, so no common parameterisation is possible).

    Args:
        warp_size:      GPU warp width W.
        lambda_k_dict:  {adapter_id: lambda_k} mapping.
        war_target:     WAR* target in (0, 1).
        ttft_slo_ms:    Fairness cap in milliseconds.

    Returns:
        {adapter_id: T_max^(k)*_seconds} dict.
    """
    return {
        adapter_id: compute_tmax_erlang(warp_size, lam, war_target, ttft_slo_ms)
        for adapter_id, lam in lambda_k_dict.items()
    }


def erlang_cdf(t: float, warp_size: int, lambda_k: float) -> float:
    """Evaluate the Erlang CDF F_k(t) = P(T_fill ≤ t).

    Used for numerical validation of Theorem 5.3:
    if T_max^(k)* was computed at quantile WAR*, then F_k(T_max^(k)*) == WAR*.

    Args:
        t:          Time in seconds.
        warp_size:  W.
        lambda_k:   Arrival rate (tokens/sec).

    Returns:
        Probability P(T_fill ≤ t).
    """
    if lambda_k <= 0.0 or t <= 0.0:
        return 0.0
    return float(erlang.cdf(t, a=warp_size, scale=1.0 / lambda_k))


def erlang_pdf(t: float, warp_size: int, lambda_k: float) -> float:
    """Erlang PDF f_k(t) -- used in quantization over-delivery bound (Prop 5.5).

    The over-delivery magnitude is bounded by:
        WAR_quantized - WAR* ≤ λ_k × τ_iter × f_k(T_max^(k)*)

    Args:
        t:          Time in seconds (evaluated at T_max^(k)*).
        warp_size:  W.
        lambda_k:   Arrival rate.

    Returns:
        f_k(t).
    """
    if lambda_k <= 0.0 or t <= 0.0:
        return 0.0
    return float(erlang.pdf(t, a=warp_size, scale=1.0 / lambda_k))


def quantization_conservatism_bound(
    t_max_s: float,
    tau_iter_s: float,
    warp_size: int,
    lambda_k: float,
) -> float:
    """Upper bound on WAR over-delivery due to iteration quantization (Prop 5.5).

    Under iteration-quantized dispatch (tokens only dispatched at iteration
    boundaries {0, τ_iter, 2τ_iter, ...}), the effective timeout is:
        T_max_quantized = ceil(T_max / τ_iter) × τ_iter  ≥  T_max

    By monotonicity of the Erlang CDF:
        WAR_quantized = F_k(T_max_quantized) ≥ F_k(T_max) = WAR*

    The over-delivery bound:
        WAR_quantized - WAR* ≤ F_k(T_max + τ_iter) - F_k(T_max)

    Args:
        t_max_s:     T_max^(k)* in seconds.
        tau_iter_s:  Decode iteration wall-clock time in seconds.
        warp_size:   W.
        lambda_k:    Arrival rate.

    Returns:
        Upper bound on (WAR_quantized - WAR*).
    """
    f_upper = erlang_cdf(t_max_s + tau_iter_s, warp_size, lambda_k)
    f_lower = erlang_cdf(t_max_s, warp_size, lambda_k)
    return max(0.0, f_upper - f_lower)


def fairness_constrained_war(
    warp_size: int,
    lambda_k_list: list,
    p_k_list: list,
    war_target: float,
    ttft_slo_ms: float = 2000.0,
) -> dict:
    """Compute system-wide WAR cost of the fairness constraint (§4.2).

    Reports:
        WAR_nofair: weighted sum using unconstrained T_max* (should equal WAR*)
        WAR_fair:   weighted sum using min(T_max*, TTFT_SLO) T_max
        WAR_cost:   WAR_nofair - WAR_fair

    Args:
        warp_size:      W.
        lambda_k_list:  List of per-adapter arrival rates.
        p_k_list:       List of per-adapter traffic fractions (sum to 1).
        war_target:     WAR* in (0, 1).
        ttft_slo_ms:    Fairness cap in milliseconds.

    Returns:
        dict with keys: war_nofair, war_fair, war_cost, constrained_adapters
    """
    ttft_slo_s = ttft_slo_ms / 1000.0
    war_nofair = 0.0
    war_fair = 0.0
    constrained = []

    for i, (lam, p_k) in enumerate(zip(lambda_k_list, p_k_list)):
        if lam <= 0.0:
            # No traffic: fairness cap applied
            constrained.append(i)
            war_fair += p_k * erlang_cdf(ttft_slo_s, warp_size, 1e-9)
            war_nofair += p_k * war_target
            continue

        t_unconstrained = erlang.ppf(war_target, a=warp_size, scale=1.0 / lam)
        t_fair = min(float(t_unconstrained), ttft_slo_s)

        war_nofair += p_k * erlang_cdf(float(t_unconstrained), warp_size, lam)
        war_fair += p_k * erlang_cdf(t_fair, warp_size, lam)

        if t_fair < float(t_unconstrained):
            constrained.append(i)

    return {
        "war_nofair": war_nofair,
        "war_fair": war_fair,
        "war_cost": max(0.0, war_nofair - war_fair),
        "constrained_adapters": constrained,
        "n_constrained": len(constrained),
    }
