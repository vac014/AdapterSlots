"""
erlang.py -- Erlang fill-probability deadlines T*(k) (Theorem 1).

Under Poisson token arrivals at rate lambda_k, the time for adapter k's queue to
collect W tokens from empty is Erlang(W, lambda_k). Inverting its CDF F gives
the deadline at which the queue fills a warp with probability WAR*:

    T*(k) = min{ F^{-1}_{W, lambda_k}(WAR*), T_SLO }

With the cap inactive, P(T_fill <= T*(k)) = WAR*. When the cap binds the
attained fill probability is F(T_SLO) < WAR*. The deadlines are computed on the
CPU in the scheduler, before tensor-parallel sharding.
"""

from functools import lru_cache

from scipy.special import gammaincinv
from scipy.stats import erlang


@lru_cache(maxsize=256)
def _gamma_ppf_unit_scale(warp_size: int, war_target: float) -> float:
    """Erlang(W, 1) quantile at WAR*.

    Erlang(W, lambda).ppf(q) = gammaincinv(W, q) / lambda, so the quantile for
    any rate is this cached value divided by lambda_k. It replaces one
    scipy.stats.erlang.ppf call per adapter per tick (about 0.1 ms each) with a
    division.
    """
    return float(gammaincinv(warp_size, war_target))


def compute_tmax_erlang(
    warp_size: int,
    lambda_k: float,
    war_target: float,
    ttft_slo_ms: float = 200.0,
) -> float:
    """T*(k) = min{F^{-1}_{W, lambda_k}(WAR*), T_SLO} in seconds.

    Args:
        warp_size:    Warp width W.
        lambda_k:     Token arrival rate of adapter k (tokens/s). A rate of zero
                      or below returns T_SLO.
        war_target:   Target fill probability WAR* in (0, 1).
        ttft_slo_ms:  T_SLO in milliseconds.

    Returns:
        T*(k) in seconds, in [0, ttft_slo_ms / 1000].
    """
    if lambda_k <= 0.0:
        return ttft_slo_ms / 1000.0
    t_fill = _gamma_ppf_unit_scale(warp_size, war_target) / lambda_k
    return min(float(t_fill), ttft_slo_ms / 1000.0)


def compute_tmax_erlang_batch(
    warp_size: int,
    lambda_k_dict: dict,
    war_target: float,
    ttft_slo_ms: float = 200.0,
) -> dict:
    """T*(k) for every adapter in {adapter_id: lambda_k}; returns {adapter_id: seconds}."""
    return {
        adapter_id: compute_tmax_erlang(warp_size, lam, war_target, ttft_slo_ms)
        for adapter_id, lam in lambda_k_dict.items()
    }


def erlang_cdf(t: float, warp_size: int, lambda_k: float) -> float:
    """F_{W, lambda_k}(t) = P(T_fill <= t): probability a queue fills W tokens
    within t seconds of becoming non-empty."""
    if lambda_k <= 0.0 or t <= 0.0:
        return 0.0
    return float(erlang.cdf(t, a=warp_size, scale=1.0 / lambda_k))
