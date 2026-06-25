"""
estimator.py -- Online EWMA estimator for per-adapter arrival rate λ_k.

Used by the Erlang timeout system (erlang_scheduler) to feed live λ_k estimates into
compute_tmax_erlang() for per-adapter T_max^(k)* computation.

Design (erlang_scheduler.md §3.2):
    Uses exponential smoothing over inter-arrival times:
        rate_estimate_new = (1 - α) * rate_estimate_old  +  α * (1 / IAT)

    where IAT is the inter-arrival time between consecutive tokens for the
    same adapter.  This is an online EWMA over the instantaneous rate.

TP=2 integration note:
    The ArrivalRateEstimator MUST run only in the scheduler process (rank 0
    orchestrator). Each arriving token is logged once at the scheduling layer,
    before TP sharding. If both TP workers tracked arrivals independently, the
    EWMA would receive 2× the updates and overestimate λ_k by ~2×.

    The constructor enforces this via a LOCAL_RANK environment check.

EWMA alpha selection (§6.4 experiment):
    α too small → slow adaptation to rate changes (bad for drift/burst)
    α too large → noisy estimates → Erlang T_max^(k)* oscillates
    Recommended: α = 0.1 (tested in §6.4 against {0.05, 0.1, 0.2})

References:
    - erlang_scheduler.md §3.2, §6.4
"""

import os
import time
from typing import Dict, Optional

import numpy as np


class ArrivalRateEstimator:
    """Online EWMA estimator for per-adapter arrival rate λ_k (tokens/sec).

    Maintains a running estimate of λ_k for each adapter using exponential
    smoothing on inter-arrival times (IAT).  The first arrival for an adapter
    seeds the estimate at the specified default_rate (avoids cold-start delay).

    Args:
        alpha:        EWMA smoothing factor ∈ (0, 1).  α=0.1 recommended.
                      Smaller α → smoother, slower convergence.
                      Larger α → noisier, faster adaptation.
        default_rate: Initial λ_k estimate before sufficient data is seen
                      (tokens/sec).  Default: 1.0 req/s (neutral starting point).
        enforce_rank0: If True, assert that LOCAL_RANK == 0 to prevent
                      accidental double-counting under TP=2.  Set to False
                      only in unit-test environments where LOCAL_RANK is unset.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        default_rate: float = 1.0,
        enforce_rank0: bool = True,
    ) -> None:
        if enforce_rank0:
            tp_rank = int(os.environ.get("LOCAL_RANK", 0))
            assert tp_rank == 0, (
                "ArrivalRateEstimator must run in scheduler (rank 0) only. "
                f"Got LOCAL_RANK={tp_rank}. Under TP=2 each arrival must be "
                "logged once, not once per GPU worker."
            )

        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.default_rate = default_rate

        # Per-adapter state
        self._last_arrival: Dict[str, float] = {}     # wall-clock time of last token
        self._rate_estimate: Dict[str, float] = {}    # current λ_k estimate
        self._arrival_count: Dict[str, int] = {}      # total arrivals seen per adapter

    # Public API

    def update(self, adapter_id: str, t_now: Optional[float] = None) -> float:
        """Record a new token arrival and update the EWMA λ_k estimate.

        Call this once per arriving token in the scheduler process (rank 0).
        Do NOT call from GPU worker processes.

        Args:
            adapter_id: String identifier of the adapter this token belongs to.
            t_now:      Wall-clock time of arrival (seconds, monotonic).
                        Defaults to time.monotonic() if not provided.

        Returns:
            Updated λ_k estimate (tokens/sec) for this adapter.
        """
        if t_now is None:
            t_now = time.monotonic()

        self._arrival_count[adapter_id] = self._arrival_count.get(adapter_id, 0) + 1

        if adapter_id in self._last_arrival:
            iat = t_now - self._last_arrival[adapter_id]
            # Guard against zero or negative IAT (clock resolution, burst arrivals)
            iat = max(iat, 1e-9)

            old_rate = self._rate_estimate.get(adapter_id, self.default_rate)
            instantaneous_rate = 1.0 / iat
            new_rate = (1.0 - self.alpha) * old_rate + self.alpha * instantaneous_rate
            self._rate_estimate[adapter_id] = new_rate
        else:
            # First arrival: seed with default rate
            self._rate_estimate.setdefault(adapter_id, self.default_rate)

        self._last_arrival[adapter_id] = t_now
        return self._rate_estimate[adapter_id]

    def get_rate(self, adapter_id: str) -> float:
        """Return current λ_k estimate for an adapter (tokens/sec).

        Returns default_rate if adapter has not been seen yet.
        """
        return self._rate_estimate.get(adapter_id, self.default_rate)

    def get_all_rates(self) -> Dict[str, float]:
        """Return {adapter_id: lambda_k} for all seen adapters."""
        return dict(self._rate_estimate)

    def get_arrival_count(self, adapter_id: str) -> int:
        """Return total number of arrivals seen for an adapter."""
        return self._arrival_count.get(adapter_id, 0)

    def reset(self, adapter_id: Optional[str] = None) -> None:
        """Reset EWMA state.

        Args:
            adapter_id: If given, reset only this adapter.
                        If None, reset all adapters.
        """
        if adapter_id is not None:
            self._last_arrival.pop(adapter_id, None)
            self._rate_estimate.pop(adapter_id, None)
            self._arrival_count.pop(adapter_id, None)
        else:
            self._last_arrival.clear()
            self._rate_estimate.clear()
            self._arrival_count.clear()

    def convergence_check(
        self,
        adapter_id: str,
        true_rate: float,
        tolerance: float = 0.20,
    ) -> bool:
        """Check if the EWMA estimate has converged within tolerance of the true rate.

        Used in §6.4 EWMA α sensitivity experiment and EC 11.1.5 validation.

        Args:
            adapter_id:  Adapter to check.
            true_rate:   Ground-truth λ_k (tokens/sec).
            tolerance:   Fractional tolerance, default 0.20 (±20%).

        Returns:
            True if |estimate - true_rate| / true_rate ≤ tolerance.
        """
        estimate = self.get_rate(adapter_id)
        if true_rate <= 0:
            return False
        relative_error = abs(estimate - true_rate) / true_rate
        return relative_error <= tolerance

    def summary(self) -> dict:
        """Return a summary dict for logging/debugging."""
        return {
            "n_adapters_seen": len(self._rate_estimate),
            "alpha": self.alpha,
            "rates": dict(self._rate_estimate),
            "arrival_counts": dict(self._arrival_count),
        }


# Lipschitz constant estimation (pi_controller, Proposition 6.7)


def estimate_lipschitz(
    lambda_k_dict: dict,
    warp_size: int,
    tmax_range: tuple = None,
    n_points: int = 50,
) -> float:
    """
    Estimate the Lipschitz constant L of WAR(T_max) from the Erlang model.

    L = max |ΔWAR / ΔT_max| over a fine grid of T_max values.

    Uses the theoretical Erlang CDF formula (arrival-rate based) rather than
    empirical hardware observations.  This avoids the staircase quantization
    bias present in PCIe hardware observations (Corollary to Proposition 6.7):
    on PCIe, τ_iter >> T_max* makes the empirical WAR(T_max) piecewise-constant,
    causing empirical finite differences to underestimate L.

    L is hardware-independent (depends only on λ_k, p_k, W -- not on τ_iter or
    GPU architecture).  Gains K_p and K_i calibrated from this L are portable
    across Single A6000, Two A6000 PCIe, and Two H100 NVLink without re-tuning
    (Proposition 6.7).

    Args:
        lambda_k_dict:  {lambda_k: p_k} -- arrival rate → mixture weight mapping.
                        Both lambda_k and p_k must be positive; p_k need not sum
                        to 1 (they are normalised internally).
                        NOTE: duplicate lambda_k keys will be collapsed by dict.
                        For uniform distributions, use distinct per-adapter lambdas
                        (e.g., add small epsilon offsets) or pass a list via the
                        lambda_k_list / p_k_list keyword alternative.
        warp_size:      Erlang shape parameter W (number of tokens per warp).
        tmax_range:     (T_min, T_max) grid bounds in seconds.  If None (default),
                        auto-computed from the lambda values: the range spans from
                        0.1× to 5× the Erlang mean W/max(lambda_k), centred on the
                        region where the WAR curve transitions from ~0 to ~1.
        n_points:       Number of grid points for finite-difference estimation.
                        Increase for higher accuracy at the cost of runtime.

    Returns:
        L -- estimated Lipschitz constant (max absolute finite difference of WAR
        over the T_max grid).  Always positive.

    Example:
        # K=4, Zipf α=0.9, λ_total=7 (in any consistent unit with tmax)
        from adapter_slots.control.estimator import estimate_lipschitz
        lambdas = [3.19, 1.71, 1.19, 0.92]
        probs   = [0.42, 0.22, 0.16, 0.12]
        rates = {lam: p for lam, p in zip(lambdas, probs)}
        L = estimate_lipschitz(rates, warp_size=32)
        print(f"L = {L:.6f},  K_p_max = {2/L:.4f}")

    References:
        - pi_controller.md §3.2, §7.6 (Proposition 6.7 and Corollary)
    """
    from scipy.stats import erlang as _erlang

    # Normalise mixture weights
    raw_weights = list(lambda_k_dict.values())
    total_weight = sum(raw_weights)
    if total_weight <= 0:
        raise ValueError("lambda_k_dict mixture weights must sum to a positive value.")

    normalised = {
        lam: p / total_weight for lam, p in lambda_k_dict.items()
    }

    # Auto-compute tmax_range from lambda values if not provided.
    # The Erlang mean is W/lambda_k; the WAR curve transitions around this range.
    # We cover [0.01 * mean_max, 5 * mean_max] to capture the full transition.
    if tmax_range is None:
        lam_max = max(normalised.keys())
        lam_min_nonzero = min(lam for lam in normalised.keys() if lam > 0)
        erlang_mean_fastest = warp_size / max(lam_max, 1e-12)
        erlang_mean_slowest = warp_size / max(lam_min_nonzero, 1e-12)
        tmax_lo = max(1e-6, erlang_mean_fastest * 0.01)
        tmax_hi = erlang_mean_slowest * 3.0
        tmax_range = (tmax_lo, tmax_hi)

    tmax_values = np.linspace(tmax_range[0], tmax_range[1], n_points)
    wars = []
    for tmax in tmax_values:
        war = sum(
            p * _erlang.cdf(tmax, a=warp_size, scale=1.0 / max(lam, 1e-12))
            for lam, p in normalised.items()
        )
        wars.append(war)

    wars = np.array(wars)
    deltas = np.diff(wars) / np.diff(tmax_values)
    L = float(np.max(np.abs(deltas)))
    return max(L, 1e-9)   # guard: never return exactly 0


def compute_war_curve(
    lambda_k_dict: dict,
    warp_size: int,
    tmax_range: tuple = None,
    n_points: int = 50,
) -> tuple:
    """
    Compute the theoretical WAR(T_max) curve from the Erlang model.

    Returns (tmax_values, war_values) arrays of length n_points.

    Useful for visualising the WAR curve and cross-hardware Lipschitz comparison
    (§5.5b / §5.6b in pi_controller.md).
    """
    from scipy.stats import erlang as _erlang

    raw_weights = list(lambda_k_dict.values())
    total_weight = sum(raw_weights)
    normalised = {lam: p / total_weight for lam, p in lambda_k_dict.items()}

    if tmax_range is None:
        lam_max = max(normalised.keys())
        lam_min_nonzero = min(lam for lam in normalised.keys() if lam > 0)
        tmax_lo = max(1e-6, warp_size / max(lam_max, 1e-12) * 0.01)
        tmax_hi = warp_size / max(lam_min_nonzero, 1e-12) * 3.0
        tmax_range = (tmax_lo, tmax_hi)

    tmax_values = np.linspace(tmax_range[0], tmax_range[1], n_points)
    wars = []
    for tmax in tmax_values:
        war = sum(
            p * _erlang.cdf(tmax, a=warp_size, scale=1.0 / max(lam, 1e-12))
            for lam, p in normalised.items()
        )
        wars.append(war)

    return tmax_values, np.array(wars)
