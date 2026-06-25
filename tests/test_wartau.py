"""
test_wartau.py -- Unit tests for compute_wartau() and compute_wartau_per_adapter().
"""

import pytest
from adapter_slots.metrics.war import compute_wartau, compute_wartau_per_adapter


# Helpers

def make_tokens(adapter_ids, arrival_times=None):
    """
    Build token dicts. If arrival_times is None, tokens have no arrival_time_ms.
    """
    if arrival_times is None:
        return [{"adapter_id": a} for a in adapter_ids]
    return [
        {"adapter_id": a, "arrival_time_ms": t}
        for a, t in zip(adapter_ids, arrival_times)
    ]


# compute_wartau

class TestComputeWartau:

    def test_empty_batch(self):
        assert compute_wartau([], dispatch_time_ms=100.0) == 0.0

    def test_no_arrival_times(self):
        # Tokens without arrival_time_ms → WARτ = 0.0
        tokens = make_tokens([0, 1, 0, 1])
        assert compute_wartau(tokens, dispatch_time_ms=50.0) == 0.0

    def test_all_same_wait(self):
        # All tokens arrived at t=0, dispatched at t=100 → mean wait = 100 ms
        tokens = make_tokens([0, 1, 2, 3], arrival_times=[0.0, 0.0, 0.0, 0.0])
        result = compute_wartau(tokens, dispatch_time_ms=100.0)
        assert abs(result - 100.0) < 1e-9

    def test_varying_wait_times(self):
        # Arrivals at [0, 10, 20, 30], dispatched at 100 → mean wait = 80, 90... → 75
        tokens = make_tokens([0, 0, 1, 1], arrival_times=[0.0, 10.0, 20.0, 30.0])
        result = compute_wartau(tokens, dispatch_time_ms=100.0)
        expected = (100 + 90 + 80 + 70) / 4  # = 85.0
        assert abs(result - expected) < 1e-9

    def test_negative_age_clipped_to_zero(self):
        # arrival_time > dispatch_time → age should be clipped to 0
        tokens = make_tokens([0], arrival_times=[200.0])
        result = compute_wartau(tokens, dispatch_time_ms=100.0)
        assert result == 0.0

    def test_mixed_with_and_without_arrival_times(self):
        # Only tokens with arrival_time_ms contribute
        tokens = [
            {"adapter_id": 0, "arrival_time_ms": 0.0},
            {"adapter_id": 1},                          # no arrival time
            {"adapter_id": 2, "arrival_time_ms": 10.0},
        ]
        result = compute_wartau(tokens, dispatch_time_ms=100.0)
        expected = (100.0 + 90.0) / 2  # = 95.0
        assert abs(result - expected) < 1e-9

    def test_single_token(self):
        tokens = make_tokens([0], arrival_times=[50.0])
        result = compute_wartau(tokens, dispatch_time_ms=75.0)
        assert abs(result - 25.0) < 1e-9

    def test_zero_wait_time(self):
        # Token arrived exactly at dispatch time
        tokens = make_tokens([0], arrival_times=[100.0])
        result = compute_wartau(tokens, dispatch_time_ms=100.0)
        assert result == 0.0


# compute_wartau_per_adapter

class TestComputeWartauPerAdapter:

    def test_empty_batch(self):
        result = compute_wartau_per_adapter([], dispatch_time_ms=100.0)
        assert result == {}

    def test_single_adapter(self):
        tokens = make_tokens([0, 0], arrival_times=[0.0, 10.0])
        result = compute_wartau_per_adapter(tokens, dispatch_time_ms=100.0)
        assert 0 in result
        assert abs(result[0] - 95.0) < 1e-9   # mean of (100, 90) = 95 → wait of (100-0, 100-10)

    def test_two_adapters_different_wait(self):
        # adapter 0: arrived at 0 → wait 100; adapter 1: arrived at 50 → wait 50
        tokens = [
            {"adapter_id": 0, "arrival_time_ms": 0.0},
            {"adapter_id": 1, "arrival_time_ms": 50.0},
        ]
        result = compute_wartau_per_adapter(tokens, dispatch_time_ms=100.0)
        assert abs(result[0] - 100.0) < 1e-9
        assert abs(result[1] - 50.0) < 1e-9

    def test_returns_only_adapters_with_arrival_times(self):
        tokens = [
            {"adapter_id": 0, "arrival_time_ms": 0.0},
            {"adapter_id": 1},   # no arrival time
        ]
        result = compute_wartau_per_adapter(tokens, dispatch_time_ms=100.0)
        assert 0 in result
        assert 1 not in result

    def test_multiple_tokens_per_adapter(self):
        # adapter 2: arrivals at 0, 20, 40 → waits 100, 80, 60 → mean = 80
        tokens = make_tokens([2, 2, 2], arrival_times=[0.0, 20.0, 40.0])
        result = compute_wartau_per_adapter(tokens, dispatch_time_ms=100.0)
        assert abs(result[2] - 80.0) < 1e-9
