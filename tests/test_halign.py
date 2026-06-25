"""
test_halign.py -- Unit tests for compute_halign() and halign_upper_bound().

All cases from instrumentation.md §3.3 are covered, plus the Theorem 7.4
bound check: H_align ≤ log2(K) and H_align = 0 when WAR = 1.0.
"""

import math
import pytest
from adapter_slots.metrics.war import compute_halign, compute_war, WARP_SIZE
from adapter_slots.metrics.entropy import halign_upper_bound


# Helpers

def make_tokens(adapter_ids):
    return [{"adapter_id": a} for a in adapter_ids]


# compute_halign

class TestComputeHalign:

    def test_empty_batch(self):
        assert compute_halign([]) == 0.0

    def test_fewer_than_one_warp(self):
        tokens = make_tokens([0] * 16)
        assert compute_halign(tokens) == 0.0

    def test_perfectly_aligned_single_warp(self):
        # All tokens same adapter → entropy = 0
        tokens = make_tokens([0] * 32)
        assert compute_halign(tokens) == 0.0

    def test_perfectly_aligned_two_warps(self):
        tokens = make_tokens([0] * 32 + [1] * 32)
        assert compute_halign(tokens) == 0.0

    def test_fully_mixed_two_adapters_single_warp(self):
        # 50/50 split within 1 warp → H = log2(2) = 1.0
        tokens = make_tokens([0, 1] * 16)   # 32 tokens
        result = compute_halign(tokens, warp_size=32)
        assert abs(result - 1.0) < 1e-9

    def test_fully_mixed_four_adapters_single_warp(self):
        # Each of 4 adapters appears exactly 8 times in 32-token warp
        ids = [i % 4 for i in range(32)]
        tokens = make_tokens(ids)
        result = compute_halign(tokens, warp_size=32)
        assert abs(result - math.log2(4)) < 1e-9

    def test_halign_zero_iff_war_one(self):
        # When WAR = 1.0, H_align must be 0.0 (Theorem 7.4)
        for adapter in range(4):
            tokens = make_tokens([adapter] * 64)
            war = compute_war(tokens)
            halign = compute_halign(tokens)
            assert war == 1.0
            assert halign == 0.0

    def test_halign_upper_bound_respected(self):
        # H_align <= log2(K) for any batch (Theorem 7.4)
        K = 4
        upper = math.log2(K)
        for _ in range(50):
            import random
            random.seed(_ + 100)
            ids = [random.randint(0, K - 1) for _ in range(128)]
            tokens = make_tokens(ids)
            h = compute_halign(tokens)
            assert h <= upper + 1e-9, f"H_align={h:.4f} exceeded upper bound {upper:.4f}"

    def test_partial_tail_ignored(self):
        # 48 tokens, warp_size=32: 1 complete warp + 16-token tail
        # warp0 = [0]*32 (aligned) → H_align = 0.0
        tokens = make_tokens([0] * 32 + [1] * 16)
        assert compute_halign(tokens) == 0.0

    def test_custom_warp_size(self):
        # W=4, 2 warps: [0,1,2,3] and [0,0,0,0]
        # warp0 entropy = log2(4) = 2.0; warp1 entropy = 0.0 → mean = 1.0
        tokens = make_tokens([0, 1, 2, 3, 0, 0, 0, 0])
        result = compute_halign(tokens, warp_size=4)
        assert abs(result - 1.0) < 1e-9

    def test_halign_range(self):
        import random
        random.seed(42)
        for _ in range(100):
            n = random.randint(32, 256)
            K = random.randint(2, 8)
            ids = [random.randint(0, K - 1) for _ in range(n)]
            tokens = make_tokens(ids)
            h = compute_halign(tokens)
            assert 0.0 <= h <= math.log2(K) + 1e-9


# halign_upper_bound

class TestHalignUpperBound:

    def test_k1(self):
        assert halign_upper_bound(1) == 0.0

    def test_k0(self):
        assert halign_upper_bound(0) == 0.0

    def test_k2(self):
        assert abs(halign_upper_bound(2) - 1.0) < 1e-9

    def test_k4(self):
        assert abs(halign_upper_bound(4) - 2.0) < 1e-9

    def test_k8(self):
        assert abs(halign_upper_bound(8) - 3.0) < 1e-9

    def test_k16(self):
        assert abs(halign_upper_bound(16) - 4.0) < 1e-9
