"""
test_prefetch.py -- Unit tests for adapter_slots/prefetch/ (adapter_prefetching).

Covers:
    - PredictivePrefetcher: score formula, priority ordering, eviction choice
    - WarmCacheManager: hit/miss tracking, LRU/topk/predictive eviction
    - make_prefetcher factory
"""

import math
import pytest

from adapter_slots.prefetch.predictor import PredictivePrefetcher
from adapter_slots.prefetch.cache_manager import WarmCacheManager
from adapter_slots.prefetch.policy import PrefetchPolicy, make_prefetcher


# PredictivePrefetcher

class TestPredictivePrefetcher:

    def test_score_zero_lambda(self):
        p = PredictivePrefetcher(tau_load_ms=200.0)
        assert p.score(0.0) == 0.0

    def test_score_positive_lambda(self):
        p = PredictivePrefetcher(tau_load_ms=200.0)
        s = p.score(1.0)
        expected = 1.0 - math.exp(-1.0 * 0.200)
        assert abs(s - expected) < 1e-9

    def test_score_increases_with_lambda(self):
        p = PredictivePrefetcher(tau_load_ms=200.0)
        assert p.score(0.5) < p.score(2.0) < p.score(10.0)

    def test_score_in_unit_interval(self):
        p = PredictivePrefetcher(tau_load_ms=200.0)
        for lam in [0.0, 0.1, 1.0, 5.0, 100.0]:
            s = p.score(lam)
            assert 0.0 <= s <= 1.0

    def test_should_prefetch_above_threshold(self):
        # p_thresh=0.1, τ=200ms → λ_min = -ln(0.9)/0.2 ≈ 0.527 req/s
        p = PredictivePrefetcher(tau_load_ms=200.0, p_thresh=0.1)
        assert p.should_prefetch(1.0) is True

    def test_should_not_prefetch_below_threshold(self):
        p = PredictivePrefetcher(tau_load_ms=200.0, p_thresh=0.1)
        assert p.should_prefetch(0.01) is False

    def test_get_prefetch_priorities_excludes_warm(self):
        p = PredictivePrefetcher(tau_load_ms=200.0, p_thresh=0.001)
        rates = {"A": 5.0, "B": 3.0, "C": 1.0}
        warm = {"A"}
        pq = p.get_prefetch_priorities(rates, warm_set=warm)
        adapter_ids = [aid for _, aid in pq]
        assert "A" not in adapter_ids
        assert "B" in adapter_ids
        assert "C" in adapter_ids

    def test_get_prefetch_priorities_sorted_descending(self):
        p = PredictivePrefetcher(tau_load_ms=200.0, p_thresh=0.001)
        rates = {"A": 5.0, "B": 1.0, "C": 10.0}
        pq = p.get_prefetch_priorities(rates)
        scores = [s for s, _ in pq]
        assert scores == sorted(scores, reverse=True)

    def test_get_eviction_target_lowest_rate(self):
        p = PredictivePrefetcher(tau_load_ms=200.0)
        rates = {"A": 5.0, "B": 0.1, "C": 3.0}
        warm = {"A", "B", "C"}
        evict = p.get_eviction_target(rates, warm)
        assert evict == "B"  # lowest rate → lowest score → evict

    def test_get_eviction_target_empty_warm_returns_none(self):
        p = PredictivePrefetcher(tau_load_ms=200.0)
        assert p.get_eviction_target({}, set()) is None

    def test_lambda_threshold_formula(self):
        tau_ms = 200.0
        p_thresh = 0.1
        p = PredictivePrefetcher(tau_load_ms=tau_ms, p_thresh=p_thresh)
        expected = -math.log(1.0 - p_thresh) / (tau_ms / 1000.0)
        assert abs(p.lambda_threshold - expected) < 1e-9

    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError):
            PredictivePrefetcher(tau_load_ms=0.0)

    def test_invalid_p_thresh_raises(self):
        with pytest.raises(ValueError):
            PredictivePrefetcher(tau_load_ms=200.0, p_thresh=1.0)


# WarmCacheManager

class TestWarmCacheManager:

    def test_cold_request_returns_miss(self):
        c = WarmCacheManager(k_warm_max=4, tau_load_ms=100.0, policy="lru")
        warm, penalty = c.request("A")
        assert warm is False
        assert penalty == 100.0

    def test_warm_request_returns_hit(self):
        c = WarmCacheManager(k_warm_max=4, tau_load_ms=100.0)
        c.request("A")   # cold first
        warm, penalty = c.request("A")  # second = warm
        assert warm is True
        assert penalty == 0.0

    def test_hit_rate_all_warm(self):
        c = WarmCacheManager(k_warm_max=4, tau_load_ms=100.0)
        c.request("A")
        c.request("A")
        c.request("A")
        # 1 miss + 2 hits = hit_rate = 2/3
        assert abs(c.hit_rate - 2 / 3) < 0.001

    def test_lru_eviction_order(self):
        # Fill cache with 2 adapters, then a 3rd should evict the LRU
        c = WarmCacheManager(k_warm_max=2, tau_load_ms=100.0, policy="lru")
        c.request("A")
        c.request("B")
        # A is LRU now; access B to make A older
        c.request("B")
        # Request C -- should evict A
        c.request("C")
        assert c.is_warm("B")
        assert c.is_warm("C")
        assert not c.is_warm("A")

    def test_topk_eviction_prefers_low_rate(self):
        c = WarmCacheManager(k_warm_max=2, tau_load_ms=100.0, policy="topk")
        c.request("A", rate_estimates={"A": 5.0, "B": 0.1, "C": 3.0})
        c.request("B", rate_estimates={"A": 5.0, "B": 0.1, "C": 3.0})
        # Cache full: A and B. Requesting C should evict B (lowest rate)
        c.request("C", rate_estimates={"A": 5.0, "B": 0.1, "C": 3.0})
        assert c.is_warm("A")
        assert c.is_warm("C")
        assert not c.is_warm("B")

    def test_prefetch_loads_cold_adapter(self):
        c = WarmCacheManager(k_warm_max=4, tau_load_ms=100.0)
        assert not c.is_warm("X")
        loaded = c.prefetch("X")
        assert loaded is True
        assert c.is_warm("X")

    def test_prefetch_noop_on_warm_adapter(self):
        c = WarmCacheManager(k_warm_max=4, tau_load_ms=100.0)
        c.request("X")
        loaded = c.prefetch("X")
        assert loaded is False

    def test_throughput_loss_zero_when_all_warm(self):
        c = WarmCacheManager(k_warm_max=4, tau_load_ms=200.0)
        c.prefetch("A")
        c.request("A")
        c.request("A")
        # All hits after prefetch → cold_fraction = 0
        c.reset_stats()
        c.request("A")
        assert c.throughput_loss_estimate(lambda_total=7.0) == pytest.approx(0.0)

    def test_stats_keys_present(self):
        c = WarmCacheManager(k_warm_max=4, tau_load_ms=100.0)
        s = c.stats()
        for key in ["policy", "hit_rate", "cold_fraction", "total_requests",
                    "total_hits", "total_misses"]:
            assert key in s

    def test_reset_stats_clears_counters(self):
        c = WarmCacheManager(k_warm_max=4)
        c.request("A")
        c.request("A")
        c.reset_stats()
        assert c.stats()["total_requests"] == 0
        assert c.stats()["total_hits"] == 0

    def test_n_warm_does_not_exceed_k_warm_max(self):
        c = WarmCacheManager(k_warm_max=3, tau_load_ms=100.0)
        for i in range(10):
            c.request(f"adapter_{i}")
        assert c.n_warm <= 3


# make_prefetcher factory

class TestMakePrefetcher:

    def test_none_policy(self):
        cache, predictor = make_prefetcher("none", k_warm_max=50)
        assert isinstance(cache, WarmCacheManager)
        assert predictor is None

    def test_lru_policy(self):
        cache, predictor = make_prefetcher("lru", k_warm_max=50)
        assert cache.policy == "lru"
        assert predictor is None

    def test_topk_policy(self):
        cache, predictor = make_prefetcher("topk", k_warm_max=50)
        assert cache.policy == "topk"

    def test_predictive_policy_returns_predictor(self):
        cache, predictor = make_prefetcher("predictive", k_warm_max=50, tau_load_ms=200.0)
        assert isinstance(predictor, PredictivePrefetcher)
        assert cache.policy == "predictive"

    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError):
            make_prefetcher("invalid_policy", k_warm_max=50)
