"""
test_erlang.py -- Unit tests for erlang_scheduler Erlang timeout system.

Tests cover:
    - compute_tmax_erlang(): Theorem 5.3 CDF inversion correctness
    - compute_tmax_erlang_batch(): batch vectorisation
    - erlang_cdf(): round-trip validation (F_k(T_max^(k)*) ≈ WAR*)
    - fairness_constrained_war(): Theorem 5.2 fairness cap
    - quantization_conservatism_bound(): Proposition 5.5
    - ArrivalRateEstimator: EWMA convergence and update correctness
    - AlignmentBuffer.form_batch_erlang(): per-adapter T_max dispatch

Run:
    pytest tests/test_erlang.py -v
"""

import time
import pytest

from adapter_slots.dispatch.erlang import (
    compute_tmax_erlang,
    compute_tmax_erlang_batch,
    erlang_cdf,
    erlang_pdf,
    fairness_constrained_war,
    quantization_conservatism_bound,
)
from adapter_slots.control.estimator import ArrivalRateEstimator
from adapter_slots.buffer import AlignmentBuffer


# Fixtures

@pytest.fixture
def estimator():
    """ArrivalRateEstimator with enforce_rank0=False for testing."""
    return ArrivalRateEstimator(alpha=0.1, default_rate=1.0, enforce_rank0=False)


@pytest.fixture
def buffer_k4():
    """AlignmentBuffer with 4 adapters, W=32, large T_max to prevent global fires."""
    return AlignmentBuffer(
        adapters=["a0", "a1", "a2", "a3"],
        warp_size=32,
        tmax_ms=10_000.0,   # 10 s global T_max -- won't fire in tests
        ttft_slo_ms=60_000.0,  # 60 s TTFT SLO -- won't fire in tests
    )


# compute_tmax_erlang

class TestComputeTmaxErlang:

    def test_round_trip_war_target(self):
        """F_k(T_max^(k)*) should equal WAR* (Theorem 5.3 round-trip)."""
        for war_target in [0.3, 0.5, 0.7, 0.8, 0.9]:
            for lambda_k in [0.5, 1.0, 5.0, 14.0]:
                t_max = compute_tmax_erlang(
                    warp_size=32, lambda_k=lambda_k, war_target=war_target,
                    ttft_slo_ms=1_000_000.0,  # no fairness cap
                )
                p_fill = erlang_cdf(t_max, warp_size=32, lambda_k=lambda_k)
                assert abs(p_fill - war_target) < 1e-6, (
                    f"Round-trip failed: lambda_k={lambda_k}, WAR*={war_target}, "
                    f"T_max={t_max:.4f}s, P_fill={p_fill:.6f}"
                )

    def test_zero_lambda_returns_slo_cap(self):
        """lambda_k=0 -> return TTFT SLO (fairness: don't leave adapter T_max undefined)."""
        t = compute_tmax_erlang(32, 0.0, 0.8, ttft_slo_ms=2000.0)
        assert t == pytest.approx(2.0)  # 2000 ms = 2 s

    def test_negative_lambda_returns_slo_cap(self):
        """Negative lambda_k treated same as zero."""
        t = compute_tmax_erlang(32, -1.0, 0.8, ttft_slo_ms=2000.0)
        assert t == pytest.approx(2.0)

    def test_fairness_cap_applied(self):
        """Unconstrained T_max > TTFT SLO -> clamped to TTFT SLO (Theorem 5.2)."""
        # Very low arrival rate -> huge unconstrained T_max
        t = compute_tmax_erlang(32, 0.001, 0.8, ttft_slo_ms=2000.0)
        assert t == pytest.approx(2.0), f"Fairness cap not applied: T_max={t:.2f}s"

    def test_fast_adapter_below_slo(self):
        """High arrival rate -> T_max well below TTFT SLO."""
        t = compute_tmax_erlang(32, 50.0, 0.8, ttft_slo_ms=2000.0)
        assert t < 2.0, f"Fast adapter T_max={t:.3f}s should be below 2s SLO"
        assert t > 0.0, "T_max must be positive"

    def test_higher_war_target_gives_longer_tmax(self):
        """Higher WAR* requires waiting longer -> T_max^(k)* is monotone in WAR*."""
        lambda_k = 5.0
        t_low = compute_tmax_erlang(32, lambda_k, 0.5, ttft_slo_ms=1_000_000.0)
        t_high = compute_tmax_erlang(32, lambda_k, 0.9, ttft_slo_ms=1_000_000.0)
        assert t_high > t_low, (
            f"T_max should increase with WAR*: t(0.5)={t_low:.4f} >= t(0.9)={t_high:.4f}"
        )

    def test_higher_lambda_gives_shorter_tmax(self):
        """Higher arrival rate -> warp fills faster -> T_max shorter."""
        t_slow = compute_tmax_erlang(32, 1.0, 0.8, ttft_slo_ms=1_000_000.0)
        t_fast = compute_tmax_erlang(32, 10.0, 0.8, ttft_slo_ms=1_000_000.0)
        assert t_fast < t_slow, (
            f"T_max should decrease with lambda: t(1.0)={t_slow:.4f} <= t(10.0)={t_fast:.4f}"
        )

    def test_return_type_is_float(self):
        t = compute_tmax_erlang(32, 5.0, 0.8)
        assert isinstance(t, float)


# compute_tmax_erlang_batch

class TestComputeTmaxErlangBatch:

    def test_batch_matches_individual(self):
        """Batch computation must match calling compute_tmax_erlang per adapter."""
        lambda_dict = {"a0": 1.0, "a1": 5.0, "a2": 0.1, "a3": 0.0}
        batch = compute_tmax_erlang_batch(32, lambda_dict, 0.8, ttft_slo_ms=2000.0)

        for adapter_id, lam in lambda_dict.items():
            expected = compute_tmax_erlang(32, lam, 0.8, ttft_slo_ms=2000.0)
            assert batch[adapter_id] == pytest.approx(expected), (
                f"Batch mismatch for {adapter_id}: got {batch[adapter_id]:.6f}, "
                f"expected {expected:.6f}"
            )

    def test_empty_dict(self):
        result = compute_tmax_erlang_batch(32, {}, 0.8)
        assert result == {}

    def test_all_keys_returned(self):
        keys = ["adapter_0", "adapter_1", "adapter_2"]
        lambda_dict = {k: float(i + 1) for i, k in enumerate(keys)}
        result = compute_tmax_erlang_batch(32, lambda_dict, 0.8)
        assert set(result.keys()) == set(keys)


# erlang_cdf

class TestErlangCdf:

    def test_cdf_zero_time(self):
        assert erlang_cdf(0.0, 32, 5.0) == 0.0

    def test_cdf_zero_lambda(self):
        assert erlang_cdf(1.0, 32, 0.0) == 0.0

    def test_cdf_increases_with_time(self):
        lam = 5.0
        vals = [erlang_cdf(t, 32, lam) for t in [1.0, 5.0, 10.0, 100.0]]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    def test_cdf_approaches_one(self):
        assert erlang_cdf(1000.0, 32, 5.0) > 0.9999

    def test_cdf_range(self):
        for lam in [0.1, 1.0, 10.0]:
            for t in [0.1, 1.0, 10.0]:
                p = erlang_cdf(t, 32, lam)
                assert 0.0 <= p <= 1.0


# quantization_conservatism_bound

class TestQuantizationConservatism:

    def test_conservatism_nonneg(self):
        """Proposition 5.5: WAR_quantized - WAR* >= 0, bound is non-negative."""
        bound = quantization_conservatism_bound(
            t_max_s=0.5, tau_iter_s=0.1, warp_size=32, lambda_k=5.0
        )
        assert bound >= 0.0

    def test_smaller_tau_gives_smaller_bound(self):
        """Smaller tau_iter -> less quantization -> smaller over-delivery bound."""
        bound_pcie = quantization_conservatism_bound(0.5, 0.1, 32, 5.0)
        bound_nvlink = quantization_conservatism_bound(0.5, 0.005, 32, 5.0)
        assert bound_nvlink <= bound_pcie, (
            f"NVLink bound={bound_nvlink:.6f} should be <= PCIe bound={bound_pcie:.6f}"
        )

    def test_zero_tau(self):
        """Zero tau_iter -> no quantization -> zero over-delivery."""
        bound = quantization_conservatism_bound(0.5, 0.0, 32, 5.0)
        assert bound == pytest.approx(0.0, abs=1e-9)


# fairness_constrained_war

class TestFairnessConstrainedWar:

    def test_war_cost_nonneg(self):
        lams = [10.0, 1.0, 0.1, 0.001]
        p_k = [0.6, 0.25, 0.1, 0.05]
        result = fairness_constrained_war(32, lams, p_k, 0.8, ttft_slo_ms=2000.0)
        assert result["war_cost"] >= 0.0

    def test_no_rare_adapters_zero_cost(self):
        """All adapters with high lambda_k -> no fairness constraint -> ~zero cost.

        Erlang_inv(32, lam, 0.8) < 2 s requires lam > ~19 req/s.
        Use lams = [100, 80, 60, 50] so T_max*(k)* << TTFT_SLO = 2s for all.
        """
        lams = [100.0, 80.0, 60.0, 50.0]
        p_k = [0.4, 0.3, 0.2, 0.1]
        result = fairness_constrained_war(32, lams, p_k, 0.8, ttft_slo_ms=2000.0)
        assert result["war_cost"] < 0.01, (
            f"High-traffic-only workload should have near-zero fairness cost, "
            f"got {result['war_cost']:.4f}"
        )

    def test_rare_adapters_are_constrained(self):
        """Ultra-rare adapters (lambda -> 0) must be marked as constrained."""
        lams = [10.0, 0.0001]
        p_k = [0.99, 0.01]
        result = fairness_constrained_war(32, lams, p_k, 0.8, ttft_slo_ms=2000.0)
        assert 1 in result["constrained_adapters"], (
            "Adapter with lambda=0.0001 should hit fairness cap"
        )

    def test_result_keys_present(self):
        result = fairness_constrained_war(32, [5.0], [1.0], 0.8)
        for key in ["war_nofair", "war_fair", "war_cost", "constrained_adapters", "n_constrained"]:
            assert key in result

    def test_war_cost_under_5pct_with_low_weight_rare(self):
        """EC 11.1.4: System-wide WAR cost of fairness < 5% when rare adapters
        have small p_k weight.

        K=2: one dominant adapter (99% traffic, λ=99 req/s → T_max*≈370ms < 2s,
        unconstrained) and one rare adapter (1% traffic, λ=1 req/s → constrained).

        The rare adapter IS capped by the TTFT_SLO fairness constraint, but its
        p_k = 0.01 weight means its contribution to war_cost is at most
        0.01 × WAR* = 0.01 × 0.8 = 0.8 pp, well under the 5% threshold.

        This directly validates that small p_k weight bounds the fairness cost,
        independent of how constrained the rare adapter is.
        """
        lams = [99.0, 1.0]   # adapter_0: 99 req/s (unconstrained); adapter_1: 1 req/s (constrained)
        p_k = [0.99, 0.01]
        result = fairness_constrained_war(32, lams, p_k, 0.8, ttft_slo_ms=2000.0)
        # Adapter_1 is constrained; max cost = p_k[1] * WAR* = 0.01 * 0.8 = 0.8%
        assert result["war_cost"] < 0.05, (
            f"EC 11.1.4 FAIL: WAR cost={result['war_cost']:.4f} >= 0.05"
        )


# ArrivalRateEstimator

class TestArrivalRateEstimator:

    def test_first_arrival_seeds_default(self, estimator):
        """First arrival returns the default rate (no IAT yet)."""
        rate = estimator.update("a0", t_now=0.0)
        assert rate == estimator.default_rate

    def test_constant_rate_converges(self, estimator):
        """Feed arrivals at constant 5 req/s -> estimate converges to 5.0."""
        lam_true = 5.0
        iat = 1.0 / lam_true
        t = 0.0
        for _ in range(500):
            estimator.update("a0", t_now=t)
            t += iat
        rate = estimator.get_rate("a0")
        # After 500 arrivals at alpha=0.1, estimate should be within +-5% of truth
        assert abs(rate - lam_true) / lam_true < 0.05, (
            f"EWMA estimate={rate:.4f} not within 5% of true={lam_true}"
        )

    def test_step_change_convergence(self, estimator):
        """After 2x step change, converge within +-20% in 50 arrivals (EC 11.1.5)."""
        # Phase 1: arrive at 5 req/s for 200 arrivals
        lam_phase1 = 5.0
        iat1 = 1.0 / lam_phase1
        t = 0.0
        for _ in range(200):
            estimator.update("a0", t_now=t)
            t += iat1

        # Phase 2: 2x step change to 10 req/s
        lam_phase2 = 10.0
        iat2 = 1.0 / lam_phase2
        for i in range(50):
            estimator.update("a0", t_now=t)
            t += iat2

        rate = estimator.get_rate("a0")
        tolerance = 0.20
        relative_err = abs(rate - lam_phase2) / lam_phase2
        assert relative_err <= tolerance, (
            f"EC 11.1.5 FAIL: After 50 arrivals at new rate, estimate={rate:.4f} "
            f"vs true={lam_phase2}, relative_err={relative_err:.3f} > {tolerance}"
        )

    def test_multiple_adapters_independent(self, estimator):
        """Each adapter maintains independent EWMA state."""
        lam_a = 2.0
        t = 0.0
        for _ in range(300):
            estimator.update("a0", t_now=t)
            t += 1.0 / lam_a
        rate_a = estimator.get_rate("a0")
        assert abs(rate_a - lam_a) / lam_a < 0.10

    def test_get_all_rates_returns_all_seen(self, estimator):
        estimator.update("x", t_now=0.0)
        estimator.update("y", t_now=0.0)
        rates = estimator.get_all_rates()
        assert "x" in rates
        assert "y" in rates

    def test_reset_single_adapter(self, estimator):
        estimator.update("a0", t_now=0.0)
        estimator.update("a0", t_now=1.0)
        estimator.reset("a0")
        assert estimator.get_rate("a0") == estimator.default_rate
        assert estimator.get_arrival_count("a0") == 0

    def test_reset_all(self, estimator):
        estimator.update("a0", t_now=0.0)
        estimator.update("a1", t_now=0.0)
        estimator.reset()
        assert estimator.get_all_rates() == {}

    def test_convergence_check_method(self, estimator):
        """convergence_check() returns True when within tolerance."""
        lam = 5.0
        t = 0.0
        for _ in range(300):
            estimator.update("a0", t_now=t)
            t += 1.0 / lam
        assert estimator.convergence_check("a0", true_rate=5.0, tolerance=0.10)

    def test_arrival_count_tracked(self, estimator):
        for i in range(10):
            estimator.update("a0", t_now=float(i))
        assert estimator.get_arrival_count("a0") == 10

    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            ArrivalRateEstimator(alpha=0.0, enforce_rank0=False)
        with pytest.raises(ValueError):
            ArrivalRateEstimator(alpha=1.0, enforce_rank0=False)
        with pytest.raises(ValueError):
            ArrivalRateEstimator(alpha=-0.1, enforce_rank0=False)


# AlignmentBuffer.form_batch_erlang

class TestFormBatchErlang:

    def test_full_warp_dispatched_immediately(self, buffer_k4):
        """Full warp (>= W tokens) dispatched on first call regardless of T_max."""
        for i in range(32):
            buffer_k4.enqueue("a0", seq_id=i)

        tmax_k = {"a0": 100.0}  # 100 s -- won't fire
        batch = buffer_k4.form_batch_erlang(tmax_k)
        a0_tokens = [seq_id for aid, seq_id in batch if aid == "a0"]
        assert len(a0_tokens) == 32, f"Expected 32 tokens, got {len(a0_tokens)}"

    def test_partial_warp_held_while_within_tmax(self, buffer_k4):
        """Partial warp not dispatched while age < per-adapter T_max."""
        buffer_k4.enqueue("a0", seq_id=0)  # 1 token, below warp size

        tmax_k = {"a0": 1000.0}  # 1000 s -- definitely won't fire
        batch = buffer_k4.form_batch_erlang(tmax_k)
        assert len(batch) == 0, f"Partial warp should be held, got {len(batch)} tokens"

    def test_per_adapter_tmax_fires_independently(self):
        """Adapter a0 with small T_max fires while a1 with large T_max does not."""
        buf = AlignmentBuffer(
            adapters=["a0", "a1"],
            warp_size=32,
            tmax_ms=10_000.0,
            ttft_slo_ms=60_000.0,
        )
        buf.enqueue("a0", seq_id=10)
        buf.enqueue("a1", seq_id=20)

        t_now = time.monotonic()
        buf.enqueue_time["a0"] = t_now - 5.0   # 5 s old
        buf.enqueue_time["a1"] = t_now - 0.001  # 1 ms old

        tmax_k = {
            "a0": 2.0,    # T_max=2s -> should fire (age=5s > 2s)
            "a1": 100.0,  # T_max=100s -> should NOT fire
        }
        batch = buf.form_batch_erlang(tmax_k)

        dispatched_adapters = {aid for aid, _ in batch}
        assert "a0" in dispatched_adapters, "a0 should have fired (age > T_max)"
        assert "a1" not in dispatched_adapters, "a1 should be held (age << T_max)"

    def test_fairness_cap_overrides_per_adapter_tmax(self):
        """TTFT SLO cap is enforced even if tmax_k is larger."""
        buf = AlignmentBuffer(
            adapters=["a0"],
            warp_size=32,
            tmax_ms=10_000.0,
            ttft_slo_ms=1_000.0,   # 1 s SLO
        )
        buf.enqueue("a0", seq_id=0)

        t_now = time.monotonic()
        buf.enqueue_time["a0"] = t_now - 2.0   # 2 s old -> exceeds 1 s SLO

        tmax_k = {"a0": 10_000.0}   # 10_000 s -> SLO cap should fire instead
        batch = buf.form_batch_erlang(tmax_k)

        assert len(batch) == 1, (
            "Fairness SLO cap should have forced dispatch after 2s (> 1s SLO)"
        )

    def test_missing_adapter_in_tmax_k_uses_slo(self):
        """If adapter_id absent from tmax_k dict, falls back to TTFT SLO."""
        buf = AlignmentBuffer(
            adapters=["a0"],
            warp_size=32,
            tmax_ms=10_000.0,
            ttft_slo_ms=1_000.0,  # 1 s SLO
        )
        buf.enqueue("a0", seq_id=0)

        t_now = time.monotonic()
        buf.enqueue_time["a0"] = t_now - 2.0   # 2 s > 1 s SLO

        tmax_k = {}   # a0 not in dict -> fallback to SLO
        batch = buf.form_batch_erlang(tmax_k)
        assert len(batch) == 1, "Missing key should fallback to SLO cap and dispatch"

    def test_budget_respected(self, buffer_k4):
        """max_tokens budget is respected even if multiple adapters have full warps."""
        for adapter_id in ["a0", "a1", "a2", "a3"]:
            for i in range(32):
                buffer_k4.enqueue(adapter_id, seq_id=i)

        tmax_k = {aid: 100.0 for aid in ["a0", "a1", "a2", "a3"]}
        batch = buffer_k4.form_batch_erlang(tmax_k, max_tokens=32)
        assert len(batch) <= 32, f"Budget exceeded: {len(batch)} > 32"

    def test_no_loss(self, buffer_k4):
        """All enqueued tokens eventually dispatched (no token loss)."""
        n_per_adapter = 10
        adapters = ["a0", "a1", "a2", "a3"]
        for j, adapter_id in enumerate(adapters):
            for i in range(n_per_adapter):
                buffer_k4.enqueue(adapter_id, seq_id=j * n_per_adapter + i)

        # tmax_k = 0 -> all fire immediately on next call
        tmax_k = {aid: 0.0 for aid in ["a0", "a1", "a2", "a3"]}
        all_dispatched = []
        for _ in range(20):
            batch = buffer_k4.form_batch_erlang(tmax_k)
            all_dispatched.extend(batch)
            if not any(len(q) > 0 for q in buffer_k4.queues.values()):
                break

        total_dispatched = len(all_dispatched)
        total_enqueued = 4 * n_per_adapter
        assert total_dispatched == total_enqueued, (
            f"Token loss detected: dispatched={total_dispatched}, "
            f"enqueued={total_enqueued}"
        )
