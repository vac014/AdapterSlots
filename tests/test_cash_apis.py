"""
test_cash_apis.py -- the CASH score I_k = lambda_k s_k P_k(s_k) (Proposition 1)
and the APIS rate-sorted round-robin router (Sec. IV).
"""

import pytest
from scipy.stats import erlang

from adapter_slots.dispatch.cash import CASHDispatcher
from adapter_slots.integrations.apis_router import APISRouter


class TestCASHScore:

    def test_matches_formula(self):
        """I_k = lambda * (q / W) * F_Erlang(W - q, lambda)(deadline - age)."""
        cash = CASHDispatcher(warp_size=32)
        lam, q, age, deadline = 120.0, 20, 0.03, 0.2
        got = cash.compute_indices({"a": q}, {"a": age}, {"a": lam}, {"a": deadline})["a"]
        p_fill = erlang.cdf(deadline - age, a=32 - q, scale=1.0 / lam)
        assert got == pytest.approx(lam * (q / 32) * p_fill, rel=1e-9)

    def test_full_queue_fill_probability_is_one(self):
        cash = CASHDispatcher(warp_size=32)
        got = cash.compute_indices({"a": 40}, {"a": 0.5}, {"a": 10.0}, {"a": 0.2})["a"]
        assert got == pytest.approx(10.0)

    def test_empty_queue_and_expired_deadline_score_zero(self):
        cash = CASHDispatcher(warp_size=32)
        idx = cash.compute_indices({"e": 0, "x": 10}, {"x": 0.3}, {"e": 50.0, "x": 50.0},
                                   {"e": 0.2, "x": 0.2})
        assert idx == {"e": 0.0, "x": 0.0}

    def test_rank_orders_by_score(self):
        """A faster, fuller queue scores higher."""
        cash = CASHDispatcher(warp_size=32)
        idx = cash.compute_indices(
            {"slow": 8, "fast": 8, "full": 24},
            {"slow": 0.0, "fast": 0.0, "full": 0.0},
            {"slow": 5.0, "fast": 200.0, "full": 200.0},
            {"slow": 0.2, "fast": 0.2, "full": 0.2},
        )
        assert cash.rank(idx) == ["full", "fast", "slow"]

    def test_score_increases_with_fill(self):
        cash = CASHDispatcher(warp_size=32)
        lens = {str(q): q for q in range(1, 33)}
        idx = cash.compute_indices(lens, {k: 0.0 for k in lens}, {k: 100.0 for k in lens},
                                   {k: 0.2 for k in lens})
        vals = [idx[str(q)] for q in range(1, 33)]
        assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


class TestAPISRouter:

    def test_rate_sorted_round_robin(self):
        rates = {0: 9.0, 1: 7.0, 2: 5.0, 3: 3.0, 4: 1.0}
        router = APISRouter(n_shards=2, initial_rates=rates)
        assert router.assignment == {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}

    def test_requests_follow_their_adapter(self):
        router = APISRouter(n_shards=2, initial_rates={0: 2.0, 1: 1.0})
        assert [router.route(a, t) for t, a in enumerate([0, 1, 0, 1])] == [0, 1, 0, 1]

    def test_unknown_adapter_joins_round_robin(self):
        router = APISRouter(n_shards=2, initial_rates={0: 2.0})
        assert router.route(7, 0.0) == 1
        assert router.route(8, 0.1) == 0

    def test_refresh_follows_rate_change(self):
        """After the rates swap order, the refresh swaps the shards."""
        router = APISRouter(n_shards=2, rebalance_s=30.0, initial_rates={0: 10.0, 1: 1.0})
        assert router.assignment[0] == 0
        t = 0.0
        while t < 61.0:
            router.route(1, t)
            t += 0.05
            if int(t * 20) % 10 == 0:
                router.route(0, t)
        assert router.n_rebalances >= 2
        assert router.assignment[1] == 0 and router.assignment[0] == 1

    def test_shard_count_validated(self):
        with pytest.raises(ValueError):
            APISRouter(n_shards=0)
