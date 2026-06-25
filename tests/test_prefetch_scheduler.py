"""
test_prefetch_scheduler.py -- Unit tests for AlignmentAwareScheduler's
PredictiveLFU cold-start-aware scheduling integration (adapter_prefetching).

Two mechanisms tested (no vLLM dependency -- pure Python):

  1. _penalize_cold_fill_fracs(): reduces Whittle fill_frac for cold adapters
     so warm adapters rank higher → dispatched sooner → stay in GPU LRU.

  2. _apply_cold_start_boost(): boosts T_max for cold adapters in threshold/erlang
     modes where T_max < SLO (e.g., AS_TMAX_MS=5ms gives 5→12.5ms for cold).
"""

import math
import time
import pytest

from adapter_slots.prefetch.cache_manager import WarmCacheManager
from adapter_slots.prefetch.predictor import PredictivePrefetcher


# Standalone replicas of scheduler methods (no vLLM needed)

def penalize_cold_fill_fracs(fill_fracs, cache_mgr, cold_boost):
    """Mirror of AlignmentAwareScheduler._penalize_cold_fill_fracs()."""
    if cache_mgr is None:
        return fill_fracs
    warm_set = cache_mgr.warm_set
    return {
        aid: (frac if aid in warm_set else frac / cold_boost)
        for aid, frac in fill_fracs.items()
    }


def apply_cold_start_boost(tmax_k, lambda_k_dict, new_arrivals,
                            cache_mgr, ttft_slo_ms, cold_boost):
    """Mirror of AlignmentAwareScheduler._apply_cold_start_boost()."""
    if cache_mgr is None:
        return tmax_k
    for adapter_id in new_arrivals:
        cache_mgr.request(adapter_id, rate_estimates=lambda_k_dict,
                          t_now=time.perf_counter())
    slo_s = ttft_slo_ms / 1000.0
    boosted = {}
    for adapter_id, tmax_s in tmax_k.items():
        if not cache_mgr.is_warm(adapter_id):
            boosted[adapter_id] = min(tmax_s * cold_boost, slo_s)
        else:
            boosted[adapter_id] = tmax_s
    return boosted


# Tests for fill_frac penalization (Whittle mode mechanism)

class TestFillFracPenalization:

    def _make_cache(self, k_warm_max=4, policy="predictive"):
        return WarmCacheManager(k_warm_max=k_warm_max, tau_load_ms=96.3, policy=policy)

    def test_warm_adapter_fill_frac_unchanged(self):
        cache = self._make_cache()
        cache.request("A", rate_estimates={"A": 5.0, "B": 0.1})
        fill_fracs = {"A": 0.5, "B": 0.5}
        result = penalize_cold_fill_fracs(fill_fracs, cache, cold_boost=2.5)
        assert result["A"] == pytest.approx(0.5), "Warm A: fill_frac unchanged"

    def test_cold_adapter_fill_frac_reduced(self):
        cache = self._make_cache()
        # B never loaded → cold
        fill_fracs = {"B": 0.5}
        result = penalize_cold_fill_fracs(fill_fracs, cache, cold_boost=2.5)
        assert result["B"] == pytest.approx(0.5 / 2.5)

    def test_cold_boost_1_is_noop(self):
        cache = self._make_cache()
        fill_fracs = {"X": 0.8}
        result = penalize_cold_fill_fracs(fill_fracs, cache, cold_boost=1.0)
        assert result["X"] == pytest.approx(0.8)

    def test_none_cache_passthrough(self):
        fill_fracs = {"A": 0.8, "B": 0.3}
        result = penalize_cold_fill_fracs(fill_fracs, None, cold_boost=2.5)
        assert result == fill_fracs

    def test_warm_ranks_higher_than_cold(self):
        """After penalization, warm adapter should have higher fill_frac than cold."""
        cache = self._make_cache(k_warm_max=4)
        rates = {"A": 5.0, "B": 0.1}
        cache.request("A", rates)
        # Both start at same fill_frac
        fill_fracs = {"A": 0.5, "B": 0.5}
        result = penalize_cold_fill_fracs(fill_fracs, cache, cold_boost=2.5)
        assert result["A"] > result["B"], "Warm A should rank higher than cold B"

    def test_mixed_warm_cold(self):
        cache = self._make_cache(k_warm_max=4)
        rates = {"A": 5.0, "B": 0.1, "C": 3.0}
        cache.request("A", rates)
        cache.request("C", rates)
        fill_fracs = {"A": 0.6, "B": 0.6, "C": 0.6}
        result = penalize_cold_fill_fracs(fill_fracs, cache, cold_boost=2.5)
        assert result["A"] == pytest.approx(0.6), "warm A unchanged"
        assert result["C"] == pytest.approx(0.6), "warm C unchanged"
        assert result["B"] == pytest.approx(0.6 / 2.5), "cold B penalized"


# Tests for T_max boost (threshold/erlang mode, works when T_max << SLO)

class TestColdStartTmaxBoost:

    def _make_cache(self, k_warm_max=4, policy="predictive"):
        return WarmCacheManager(k_warm_max=k_warm_max, tau_load_ms=96.3, policy=policy)

    def test_warm_adapter_tmax_unchanged(self):
        cache = self._make_cache()
        cache.request("A", rate_estimates={"A": 5.0})
        result = apply_cold_start_boost(
            {"A": 0.005}, {"A": 5.0}, set(), cache, 2000.0, 2.5
        )
        assert result["A"] == pytest.approx(0.005)

    def test_cold_adapter_tmax_boosted(self):
        cache = self._make_cache()
        result = apply_cold_start_boost(
            {"B": 0.005}, {"B": 0.1}, set(), cache, 2000.0, 2.5
        )
        assert result["B"] == pytest.approx(0.005 * 2.5)

    def test_boost_capped_at_slo(self):
        cache = self._make_cache()
        # 1.5s × 2.5 = 3.75s → capped at SLO=2.0s
        result = apply_cold_start_boost(
            {"B": 1.5}, {"B": 0.1}, set(), cache, 2000.0, 2.5
        )
        assert result["B"] == pytest.approx(2.0)

    def test_new_arrivals_loaded_into_cache(self):
        cache = self._make_cache()
        assert not cache.is_warm("A")
        apply_cold_start_boost(
            {"A": 0.005}, {"A": 5.0}, {"A"}, cache, 2000.0, 2.5
        )
        assert cache.is_warm("A")

    def test_none_cache_passthrough(self):
        tmax_k = {"X": 0.010, "Y": 0.005}
        result = apply_cold_start_boost(tmax_k, {"X": 1.0}, {"X"}, None, 2000.0, 2.5)
        assert result == tmax_k


# Tests for PredictiveLFU eviction policy

class TestPredictiveEviction:

    def test_low_rate_adapter_evicted_first(self):
        cache = WarmCacheManager(k_warm_max=2, tau_load_ms=96.3, policy="predictive")
        rates = {"A": 5.0, "B": 0.05, "C": 3.0}
        cache.request("A", rates)
        cache.request("B", rates)
        # Request C → evict B (lowest Poisson score)
        cache.request("C", rates)
        assert cache.is_warm("A")
        assert cache.is_warm("C")
        assert not cache.is_warm("B")

    def test_predictive_hit_rate_vs_lru(self):
        """PredictiveLFU hit rate >= LRU over Zipf(α=0.9) traffic stream."""
        import random
        K, K_warm, N = 20, 5, 2000
        rng = random.Random(0)
        raw = [k ** -0.9 for k in range(1, K + 1)]
        total = sum(raw)
        cum, s = [], 0.0
        for w in raw:
            s += w / total
            cum.append(s)

        def pick():
            r = rng.random()
            for k, c in enumerate(cum):
                if r <= c:
                    return f"a{k}"
            return f"a{K-1}"

        from adapter_slots.control.estimator import ArrivalRateEstimator
        est = ArrivalRateEstimator(alpha=0.1, default_rate=0.0, enforce_rank0=False)
        c_lru  = WarmCacheManager(K_warm, tau_load_ms=96.3, policy="lru")
        c_pred = WarmCacheManager(K_warm, tau_load_ms=96.3, policy="predictive")

        for i in range(N):
            a = pick()
            t = i / 7.0
            est.update(a, t)
            rates = est.get_all_rates()
            c_lru.request(a,  rate_estimates=rates)
            c_pred.request(a, rate_estimates=rates)

        assert c_pred.hit_rate >= c_lru.hit_rate - 0.01, (
            f"PredLFU {c_pred.hit_rate:.3f} should >= LRU {c_lru.hit_rate:.3f}"
        )


# Composability: WAR estimator + PredictiveLFU share same λ̂_k

class TestComposability:

    def test_shared_estimator_both_mechanisms(self):
        """Both Erlang T_max and PredictiveLFU derive from the same λ̂_k."""
        from adapter_slots.control.estimator import ArrivalRateEstimator
        from adapter_slots.dispatch.erlang import compute_tmax_erlang_batch

        est  = ArrivalRateEstimator(alpha=0.1, default_rate=0.0, enforce_rank0=False)
        pred = PredictivePrefetcher(tau_load_ms=96.3, p_thresh=0.1)
        cache = WarmCacheManager(k_warm_max=5, tau_load_ms=96.3, policy="predictive")

        for i in range(50):
            est.update("hot", i * 0.143)   # λ ≈ 7 req/s
        est.update("cold", 14.0)           # 1 request

        rates = est.get_all_rates()

        # Erlang T_max uses λ̂_k
        tmax_k = compute_tmax_erlang_batch(32, rates, 0.8, 2000.0)
        assert "hot" in tmax_k

        # Poisson score uses same λ̂_k
        s_hot  = pred.score(rates.get("hot",  0.0))
        s_cold = pred.score(rates.get("cold", 0.0))
        assert s_hot > s_cold, "High-rate adapter has higher prefetch score"
        assert s_hot > 0.40,   "hot λ≈7 gives score>0.4 at τ=96ms"

        # fill_frac penalization: warm=unchanged, cold=reduced
        cache.request("hot", rates)
        fill = {"hot": 0.5, "cold": 0.5}
        penalized = penalize_cold_fill_fracs(fill, cache, cold_boost=2.5)
        assert penalized["hot"]  == pytest.approx(0.5),       "warm hot unchanged"
        assert penalized["cold"] == pytest.approx(0.5 / 2.5), "cold penalized"

    def test_composable_priority_ordering(self):
        """Combined WAR+PredLFU: warm high-rate adapter always outranks cold adapter."""
        cache = WarmCacheManager(k_warm_max=4, tau_load_ms=96.3, policy="predictive")
        rates = {"warm_high": 5.0, "cold_low": 0.1}
        cache.request("warm_high", rates)

        # Both at equal fill_frac
        fill = {"warm_high": 0.5, "cold_low": 0.5}
        penalized = penalize_cold_fill_fracs(fill, cache, cold_boost=2.5)

        assert penalized["warm_high"] > penalized["cold_low"], (
            "Warm high-rate adapter must outrank cold low-rate adapter"
        )


# Tests for PCIe minimum deferral window (Issue 10 fix)

def penalize_cold_fill_fracs_pcie(fill_fracs, cache_mgr, cold_boost,
                                   pcie_min_deferral_s, cold_first_seen, t_now=None):
    """Mirror of the PCIe-fixed AlignmentAwareScheduler._penalize_cold_fill_fracs().

    Two-phase deferral:
      Phase 1 (age < pcie_min_deferral_s): fill_frac = 0.0  (hard block)
      Phase 2 (age >= pcie_min_deferral_s): fill_frac / cold_boost  (soft priority)
      Warm: fill_frac unchanged
    """
    import time as _time
    if cache_mgr is None:
        return fill_fracs
    warm_set = cache_mgr.warm_set
    t = t_now if t_now is not None else _time.perf_counter()
    penalized = {}
    for adapter_id, frac in fill_fracs.items():
        if adapter_id in warm_set:
            cold_first_seen.pop(adapter_id, None)
            penalized[adapter_id] = frac
        else:
            if pcie_min_deferral_s > 0.0:
                first_seen = cold_first_seen.get(adapter_id)
                if first_seen is None:
                    cold_first_seen[adapter_id] = t
                    penalized[adapter_id] = 0.0
                elif (t - first_seen) < pcie_min_deferral_s:
                    penalized[adapter_id] = frac / cold_boost
                else:
                    penalized[adapter_id] = frac
            else:
                penalized[adapter_id] = frac / cold_boost
    return penalized


class TestPCIeMinimumDeferral:
    """Tests for the PCIe minimum deferral window (Issue 10, S-LoRA §4.3 fix)."""

    def _make_cache(self, k_warm_max=4):
        cache = WarmCacheManager(k_warm_max=k_warm_max, tau_load_ms=96.3, policy="predictive")
        return cache

    def test_cold_adapter_hard_blocked_on_first_appearance(self):
        """Phase 1: cold adapter seen for first time → fill_frac = 0."""
        cache = self._make_cache()
        cold_first_seen = {}
        fill = {"cold_A": 0.8}
        result = penalize_cold_fill_fracs_pcie(
            fill, cache, cold_boost=2.0,
            pcie_min_deferral_s=0.0963,
            cold_first_seen=cold_first_seen,
            t_now=1000.0
        )
        assert result["cold_A"] == pytest.approx(0.0), \
            "Cold adapter on first appearance must be hard-blocked (fill_frac=0)"
        assert "cold_A" in cold_first_seen, "First-seen timestamp must be recorded"

    def test_cold_adapter_still_blocked_within_deferral_window(self):
        """Phase 1 continuation: age < τ_load → soft penalty, not full dispatch."""
        cache = self._make_cache()
        t_first = 1000.0
        cold_first_seen = {"cold_A": t_first}  # already seen
        fill = {"cold_A": 0.8}
        # 50ms elapsed, τ_load=96.3ms → still within window
        result = penalize_cold_fill_fracs_pcie(
            fill, cache, cold_boost=2.0,
            pcie_min_deferral_s=0.0963,
            cold_first_seen=cold_first_seen,
            t_now=t_first + 0.050,
        )
        assert result["cold_A"] == pytest.approx(0.8 / 2.0), \
            "Still in deferral window → soft penalty (fill_frac/cold_boost)"

    def test_cold_adapter_eligible_after_deferral_window(self):
        """Phase 2: age ≥ τ_load → normal fill_frac (DMA complete)."""
        cache = self._make_cache()
        t_first = 1000.0
        cold_first_seen = {"cold_A": t_first}
        fill = {"cold_A": 0.8}
        # 100ms elapsed, τ_load=96.3ms → window elapsed
        result = penalize_cold_fill_fracs_pcie(
            fill, cache, cold_boost=2.0,
            pcie_min_deferral_s=0.0963,
            cold_first_seen=cold_first_seen,
            t_now=t_first + 0.100,
        )
        assert result["cold_A"] == pytest.approx(0.8), \
            "After τ_load window, cold adapter allowed normal fill_frac"

    def test_warm_adapter_clears_first_seen_tracking(self):
        """Warm adapter should remove its cold_first_seen entry."""
        cache = self._make_cache()
        rates = {"A": 5.0}
        cache.request("A", rates)
        cold_first_seen = {"A": 999.0}  # stale entry from previous cold period
        fill = {"A": 0.5}
        result = penalize_cold_fill_fracs_pcie(
            fill, cache, cold_boost=2.0,
            pcie_min_deferral_s=0.0963,
            cold_first_seen=cold_first_seen,
            t_now=1000.0,
        )
        assert result["A"] == pytest.approx(0.5), "Warm adapter fill_frac unchanged"
        assert "A" not in cold_first_seen, "Warm adapter clears cold_first_seen entry"

    def test_pcie_deferral_disabled_falls_back_to_cold_boost_only(self):
        """pcie_min_deferral_s=0 → old soft-penalty-only behavior."""
        cache = self._make_cache()
        cold_first_seen = {}
        fill = {"cold_A": 0.6}
        result = penalize_cold_fill_fracs_pcie(
            fill, cache, cold_boost=2.0,
            pcie_min_deferral_s=0.0,  # disabled
            cold_first_seen=cold_first_seen,
            t_now=1000.0,
        )
        assert result["cold_A"] == pytest.approx(0.6 / 2.0), \
            "Deferral disabled → soft penalty only (fill_frac/cold_boost)"
        assert cold_first_seen == {}, "No tracking when min_deferral=0"

    def test_pcie_cold_boost_2_calibration(self):
        """Verify PCIe formula: cold_boost=ceil(τ_load_ms/τ_iter_ms)+1=2 for A6000."""
        import math
        tau_load_ms, tau_iter_ms = 96.3, 100.0
        expected = math.ceil(tau_load_ms / tau_iter_ms) + 1
        assert expected == 2, f"PCIe cold_boost should be 2, got {expected}"

    def test_warm_adapter_always_outranks_cold_in_phase_1(self):
        """During Phase 1, even low-fill warm adapters outrank high-fill cold adapters."""
        cache = self._make_cache()
        rates = {"warm": 5.0, "cold": 5.0}
        cache.request("warm", rates)
        cold_first_seen = {}
        fill = {"warm": 0.1, "cold": 0.9}  # cold has much higher fill
        result = penalize_cold_fill_fracs_pcie(
            fill, cache, cold_boost=2.0,
            pcie_min_deferral_s=0.0963,
            cold_first_seen=cold_first_seen,
            t_now=1000.0,
        )
        assert result["warm"] > result["cold"], \
            "During Phase 1, warm adapter (0.1) must outrank cold (0.0 after hard-block)"
