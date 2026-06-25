"""
estimator.py -- EWMA estimate of each adapter's token arrival rate lambda_k.

On every new arrival for adapter k:

    lambda_k <- (1 - alpha) * lambda_k + alpha / IAT

where IAT is the time since adapter k's previous arrival. The Erlang deadlines,
the CASH scores and PredictiveLFU all read these estimates. The estimator runs
once, in the scheduler process; under tensor parallelism a per-worker copy
would count every arrival once per rank.
"""

import os
import time
from typing import Dict, Optional


class ArrivalRateEstimator:
    """Per-adapter EWMA of the instantaneous arrival rate (tokens/s).

    Args:
        alpha:         Smoothing factor in (0, 1); 0.1 in our runs.
        default_rate:  Estimate before an adapter's second arrival.
        enforce_rank0: Refuse to run outside LOCAL_RANK 0.
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
                "ArrivalRateEstimator must run in the scheduler process (rank 0), "
                f"got LOCAL_RANK={tp_rank}."
            )
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.default_rate = default_rate
        self._last_arrival: Dict[str, float] = {}
        self._rate_estimate: Dict[str, float] = {}
        self._arrival_count: Dict[str, int] = {}

    def update(self, adapter_id: str, t_now: Optional[float] = None) -> float:
        """Record one arrival for adapter_id at t_now (monotonic seconds).

        Returns the updated lambda_k.
        """
        if t_now is None:
            t_now = time.monotonic()

        self._arrival_count[adapter_id] = self._arrival_count.get(adapter_id, 0) + 1

        if adapter_id in self._last_arrival:
            iat = max(t_now - self._last_arrival[adapter_id], 1e-9)
            old_rate = self._rate_estimate.get(adapter_id, self.default_rate)
            self._rate_estimate[adapter_id] = (
                (1.0 - self.alpha) * old_rate + self.alpha / iat
            )
        else:
            self._rate_estimate.setdefault(adapter_id, self.default_rate)

        self._last_arrival[adapter_id] = t_now
        return self._rate_estimate[adapter_id]

    def set_rates(self, rates: Dict[str, float]) -> None:
        """Start from given estimates, e.g. rates measured earlier."""
        self._rate_estimate.update(rates)

    def get_rate(self, adapter_id: str) -> float:
        """lambda_k, or default_rate for an adapter not seen yet."""
        return self._rate_estimate.get(adapter_id, self.default_rate)

    def get_all_rates(self) -> Dict[str, float]:
        """{adapter_id: lambda_k} for every adapter seen."""
        return dict(self._rate_estimate)

    def get_arrival_count(self, adapter_id: str) -> int:
        return self._arrival_count.get(adapter_id, 0)

    def reset(self, adapter_id: Optional[str] = None) -> None:
        """Forget one adapter's state, or every adapter's when adapter_id is None."""
        if adapter_id is not None:
            self._last_arrival.pop(adapter_id, None)
            self._rate_estimate.pop(adapter_id, None)
            self._arrival_count.pop(adapter_id, None)
        else:
            self._last_arrival.clear()
            self._rate_estimate.clear()
            self._arrival_count.clear()

    def summary(self) -> dict:
        return {
            "n_adapters_seen": len(self._rate_estimate),
            "alpha": self.alpha,
            "rates": dict(self._rate_estimate),
            "arrival_counts": dict(self._arrival_count),
        }
