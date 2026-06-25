"""
apis_router.py -- adapter-to-shard assignment for APIS (Sec. IV).

Adapters are ordered by their EWMA request rate lambda_k, highest first, and the
adapter at position i goes to shard i mod n (rate-sorted round-robin), so the
shards share the traffic rather than an equal adapter count. Every request goes
to the shard that holds its adapter. The router updates the rate estimates on
every request and recomputes the assignment every rebalance_s seconds.
"""

from typing import Dict, Optional

from adapter_slots.control.estimator import ArrivalRateEstimator


class APISRouter:
    """Rate-sorted round-robin assignment of adapters to independent shards.

    Args:
        n_shards:       Number of TP=1 shards.
        rebalance_s:    Seconds between assignment refreshes (30 in our runs).
        alpha:          EWMA weight of the rate estimator.
        initial_rates:  Starting estimates {adapter_id: requests/s}; the initial
                        assignment is computed from them. An adapter without an
                        estimate joins the next shard in round-robin order when
                        its first request arrives.
    """

    def __init__(
        self,
        n_shards: int = 2,
        rebalance_s: float = 30.0,
        alpha: float = 0.1,
        initial_rates: Optional[Dict[int, float]] = None,
    ) -> None:
        if n_shards < 1:
            raise ValueError(f"n_shards must be at least 1, got {n_shards}")
        self.n_shards = n_shards
        self.rebalance_s = rebalance_s
        self.estimator = ArrivalRateEstimator(alpha=alpha, enforce_rank0=False)
        self.assignment: Dict[int, int] = {}
        self._last_rebalance: Optional[float] = None
        self.n_rebalances = 0
        if initial_rates:
            self.estimator.set_rates(initial_rates)
            self.rebalance()

    def rebalance(self) -> None:
        """Rate-sorted round-robin over the current rate estimates."""
        rates = self.estimator.get_all_rates()
        order = sorted(rates, key=lambda a: (-rates[a], str(a)))
        self.assignment = {a: i % self.n_shards for i, a in enumerate(order)}
        self.n_rebalances += 1

    def route(self, adapter_id: int, t_now: float) -> int:
        """Shard for a request of adapter_id arriving at t_now (seconds)."""
        self.estimator.update(adapter_id, t_now)
        if self._last_rebalance is None:
            self._last_rebalance = t_now
        elif t_now - self._last_rebalance >= self.rebalance_s:
            self.rebalance()
            self._last_rebalance = t_now
        if adapter_id not in self.assignment:
            self.assignment[adapter_id] = len(self.assignment) % self.n_shards
        return self.assignment[adapter_id]
