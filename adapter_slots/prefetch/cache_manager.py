"""
cache_manager.py -- WarmCacheManager: warm/cold adapter state tracking.

Mirrors the vLLM LoRA pool state and tracks cold-start metrics. Used by the
AlignmentAwareScheduler background prefetch thread during serving.

Cold-start throughput loss model (adapter_prefetching.md §1.2):
    loss_fraction ≈ f_cold × τ_load × λ_total / (1 + f_cold × τ_load × λ_total)
"""

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


class WarmCacheManager:
    """Tracks warm/cold adapter state and cache hit/miss statistics.

    Models the vLLM LoRA pool as a fixed-capacity cache. When a request
    arrives for a cold adapter, it records a cache miss and the τ_load
    cold-start penalty.

    Eviction policies available:
        "lru"        -- Least Recently Used (vLLM default)
        "predictive" -- Least Likely Future Use (Poisson LFU, adapter_prefetching)
        "topk"       -- Keep top-K by EWMA rate, evict lowest rate

    Args:
        k_warm_max:  Maximum simultaneously warm adapters (= --max-loras).
        tau_load_ms: Measured adapter cold-start latency in milliseconds.
        policy:      Eviction policy name ("lru", "predictive", "topk").
    """

    def __init__(
        self,
        k_warm_max: int,
        tau_load_ms: float = 200.0,
        policy: str = "lru",
    ) -> None:
        if k_warm_max <= 0:
            raise ValueError(f"k_warm_max must be positive, got {k_warm_max}")
        self.k_warm_max = k_warm_max
        self.tau_load_ms = tau_load_ms
        self.policy = policy

        # LRU cache: OrderedDict preserves insertion/access order
        # Key: adapter_id, Value: last_access_time
        self._warm: OrderedDict = OrderedDict()

        # Per-adapter stats
        self._hits: Dict[str, int] = {}
        self._misses: Dict[str, int] = {}
        self._cold_start_ms_total: Dict[str, float] = {}

        # Global counters
        self._total_requests: int = 0
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._total_evictions: int = 0

    # Public API

    def request(
        self,
        adapter_id: str,
        rate_estimates: Optional[Dict[str, float]] = None,
        t_now: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """Record a request for adapter_id.

        Args:
            adapter_id:     The requested adapter.
            rate_estimates: Current λ̂_k estimates (required for "predictive" / "topk").
            t_now:          Current time (perf_counter seconds). Auto-set if None.

        Returns:
            (was_warm, cold_start_ms):
                was_warm:      True if adapter was already in the warm set.
                cold_start_ms: 0.0 if warm; τ_load_ms if cold.
        """
        t = t_now if t_now is not None else time.perf_counter()
        self._total_requests += 1
        self._hits.setdefault(adapter_id, 0)
        self._misses.setdefault(adapter_id, 0)
        self._cold_start_ms_total.setdefault(adapter_id, 0.0)

        if adapter_id in self._warm:
            # Cache hit -- move to most-recently-used end
            self._warm.move_to_end(adapter_id)
            self._warm[adapter_id] = t
            self._hits[adapter_id] += 1
            self._total_hits += 1
            return True, 0.0
        else:
            # Cache miss -- cold start
            self._misses[adapter_id] += 1
            self._total_misses += 1
            self._cold_start_ms_total[adapter_id] += self.tau_load_ms
            self._load(adapter_id, t, rate_estimates)
            return False, self.tau_load_ms

    def prefetch(
        self,
        adapter_id: str,
        rate_estimates: Optional[Dict[str, float]] = None,
        t_now: Optional[float] = None,
    ) -> bool:
        """Pre-warm adapter_id during idle time (no request needed).

        Returns True if a new load was triggered (adapter was cold), False if already warm.
        """
        t = t_now if t_now is not None else time.perf_counter()
        if adapter_id in self._warm:
            return False
        self._load(adapter_id, t, rate_estimates)
        return True

    def is_warm(self, adapter_id: str) -> bool:
        return adapter_id in self._warm

    @property
    def warm_set(self) -> set:
        return set(self._warm.keys())

    @property
    def n_warm(self) -> int:
        return len(self._warm)

    @property
    def hit_rate(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return self._total_hits / self._total_requests

    @property
    def cold_fraction(self) -> float:
        return 1.0 - self.hit_rate

    def throughput_loss_estimate(self, lambda_total: float) -> float:
        """Analytical estimate of throughput loss fraction from cold starts.

        Uses the adapter_prefetching.md §1.2 model:
            loss ≈ f_cold × τ_load_s × λ_total / (1 + f_cold × τ_load_s × λ_total)
        """
        f = self.cold_fraction
        tau = self.tau_load_ms / 1000.0
        num = f * tau * lambda_total
        return num / (1.0 + num)

    def per_adapter_stats(self) -> Dict[str, dict]:
        """Return per-adapter hit/miss/cold-start stats."""
        result = {}
        for aid in set(list(self._hits.keys()) + list(self._misses.keys())):
            h = self._hits.get(aid, 0)
            m = self._misses.get(aid, 0)
            total = h + m
            result[aid] = {
                "hits": h,
                "misses": m,
                "hit_rate": h / total if total else 0.0,
                "cold_start_ms_total": self._cold_start_ms_total.get(aid, 0.0),
                "cold_start_ms_per_miss": (
                    self._cold_start_ms_total.get(aid, 0.0) / m if m else 0.0
                ),
            }
        return result

    def stats(self) -> dict:
        return {
            "policy": self.policy,
            "k_warm_max": self.k_warm_max,
            "n_warm_current": self.n_warm,
            "tau_load_ms": self.tau_load_ms,
            "total_requests": self._total_requests,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "total_evictions": self._total_evictions,
            "hit_rate": round(self.hit_rate, 4),
            "cold_fraction": round(self.cold_fraction, 4),
        }

    def reset_stats(self) -> None:
        self._hits.clear()
        self._misses.clear()
        self._cold_start_ms_total.clear()
        self._total_requests = 0
        self._total_hits = 0
        self._total_misses = 0
        self._total_evictions = 0

    # Internal

    def _load(
        self,
        adapter_id: str,
        t_now: float,
        rate_estimates: Optional[Dict[str, float]],
    ) -> None:
        """Load adapter_id into cache, evicting if necessary."""
        if len(self._warm) >= self.k_warm_max:
            victim = self._choose_eviction(rate_estimates)
            if victim is not None:
                del self._warm[victim]
                self._total_evictions += 1
        self._warm[adapter_id] = t_now
        self._warm.move_to_end(adapter_id)

    def _choose_eviction(
        self, rate_estimates: Optional[Dict[str, float]]
    ) -> Optional[str]:
        """Choose which warm adapter to evict based on the current policy."""
        if not self._warm:
            return None

        if self.policy == "lru":
            # Evict the least recently used (first in OrderedDict = oldest)
            return next(iter(self._warm))

        elif self.policy == "topk" and rate_estimates:
            # Evict the warm adapter with the lowest current arrival rate
            min_rate = float("inf")
            victim = None
            for aid in self._warm:
                lam = rate_estimates.get(aid, 0.0)
                if lam < min_rate:
                    min_rate = lam
                    victim = aid
            return victim

        elif self.policy == "predictive" and rate_estimates:
            # Evict the warm adapter least likely to be requested soon
            # Equivalent to topk but with Poisson-weighted score
            from adapter_slots.prefetch.predictor import PredictivePrefetcher
            predictor = PredictivePrefetcher(tau_load_ms=self.tau_load_ms)
            return predictor.get_eviction_target(
                rate_estimates, set(self._warm.keys())
            )

        else:
            # Fallback: LRU
            return next(iter(self._warm))
