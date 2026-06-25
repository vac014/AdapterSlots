"""
predictor.py -- PredictivePrefetcher: Poisson scores for prefetch and eviction.

    score(k) = 1 - exp(-lambda_k * tau_load)
             = P(adapter k receives at least one request within tau_load)

lambda_k is the EWMA rate from ArrivalRateEstimator, under the same Poisson
assumption as the Erlang deadlines.
"""

import math
from typing import Dict, List, Optional, Tuple


class PredictivePrefetcher:
    """Priority queue of adapters to prefetch, scored by Poisson arrival probability.

    Args:
        tau_load_ms:  Adapter load time in milliseconds.
        p_thresh:     An adapter is a prefetch candidate when score(k) > p_thresh.
                      At K = 50 and 7 req/s, per-adapter rates are 0.1 to 0.7 req/s;
                      0.1 admits rates above -ln(0.9) / tau_load (0.53 req/s at
                      200 ms), the top few adapters.
    """

    def __init__(self, tau_load_ms: float = 200.0, p_thresh: float = 0.1) -> None:
        if tau_load_ms <= 0:
            raise ValueError(f"tau_load_ms must be positive, got {tau_load_ms}")
        if not (0.0 < p_thresh < 1.0):
            raise ValueError(f"p_thresh must be in (0,1), got {p_thresh}")
        self.tau_load_s = tau_load_ms / 1000.0
        self.p_thresh = p_thresh
        # 1 - exp(-lambda * tau) > p  <=>  lambda > -ln(1 - p) / tau
        self._lambda_min = -math.log(1.0 - p_thresh) / self.tau_load_s

    def score(self, lambda_k: float) -> float:
        """Poisson prefetch utility score for adapter with arrival rate λ_k.

        Returns P(>=1 request within τ_load seconds) in [0, 1].
        Higher score -> higher priority to prefetch.
        """
        if lambda_k <= 0.0:
            return 0.0
        return 1.0 - math.exp(-lambda_k * self.tau_load_s)

    def eviction_score(self, lambda_k: float, tau_evict_s: float) -> float:
        """P(at least one request within tau_evict_s); the lowest is evicted."""
        if lambda_k <= 0.0 or tau_evict_s <= 0.0:
            return 0.0
        return 1.0 - math.exp(-lambda_k * tau_evict_s)

    def should_prefetch(self, lambda_k: float) -> bool:
        """Return True if adapter with rate λ_k should be pre-warmed."""
        return lambda_k > self._lambda_min

    def get_prefetch_priorities(
        self,
        rate_estimates: Dict[str, float],
        warm_set: Optional[set] = None,
        top_n: Optional[int] = None,
    ) -> List[Tuple[float, str]]:
        """Return adapters sorted by prefetch priority (highest first).

        Args:
            rate_estimates: {adapter_id: λ̂_k} from ArrivalRateEstimator.
            warm_set:       Set of adapter IDs currently warm (skip these).
            top_n:          Return at most top_n candidates.

        Returns:
            List of (score, adapter_id) tuples, sorted descending by score.
            Only includes adapters where score > p_thresh.
        """
        warm_set = warm_set or set()
        candidates = []
        for adapter_id, lam in rate_estimates.items():
            if adapter_id in warm_set:
                continue
            s = self.score(lam)
            if s > self.p_thresh:
                candidates.append((s, adapter_id))
        candidates.sort(reverse=True)
        if top_n is not None:
            candidates = candidates[:top_n]
        return candidates

    def get_eviction_target(
        self,
        rate_estimates: Dict[str, float],
        warm_set: set,
        tau_evict_s: Optional[float] = None,
    ) -> Optional[str]:
        """PredictiveLFU eviction: argmin over the warm set of
        P(request for k within tau_evict).

        Args:
            rate_estimates: {adapter_id: λ̂_k} from ArrivalRateEstimator.
            warm_set:       Current set of warm adapter IDs.
            tau_evict_s:    Eviction window in seconds. If None, uses τ_load_s.

        Returns:
            adapter_id to evict, or None if warm_set is empty.
        """
        if not warm_set:
            return None
        tau = tau_evict_s if tau_evict_s is not None else self.tau_load_s
        scores = [
            (self.eviction_score(rate_estimates.get(a, 0.0), tau), a)
            for a in warm_set
        ]
        scores.sort()  # ascending: lowest score = lowest future use = evict first
        return scores[0][1]

    def prefetch_schedule(
        self,
        rate_estimates: Dict[str, float],
        warm_set: set,
        k_warm_max: int,
    ) -> Tuple[List[str], List[str]]:
        """Adapters to load and to evict, by score, for a warm set of k_warm_max.

        Returns:
            (to_prefetch, to_evict): lists of adapter IDs to load/unload.
            Caller must apply these changes to the GPU LoRA cache.
        """
        n_free_slots = k_warm_max - len(warm_set)
        candidates = self.get_prefetch_priorities(rate_estimates, warm_set)

        to_prefetch: List[str] = []
        to_evict: List[str] = []

        for _score, adapter_id in candidates:
            if n_free_slots > 0:
                to_prefetch.append(adapter_id)
                n_free_slots -= 1
            else:
                # Need to evict to make room
                evict_target = self.get_eviction_target(
                    rate_estimates,
                    warm_set - set(to_prefetch) | set(to_evict),
                )
                if evict_target is None:
                    break
                # Only evict if the gain is worth it: new score > evict score
                new_score = self.score(rate_estimates.get(adapter_id, 0.0))
                old_score = self.score(rate_estimates.get(evict_target, 0.0))
                if new_score > old_score:
                    to_evict.append(evict_target)
                    to_prefetch.append(adapter_id)
                else:
                    break  # Remaining candidates also have lower scores; stop

        return to_prefetch, to_evict

    @property
    def lambda_threshold(self) -> float:
        """Minimum λ_k for an adapter to be considered for prefetching."""
        return self._lambda_min
