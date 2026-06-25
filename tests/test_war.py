"""
test_war.py -- Unit tests for compute_war() and compute_war_from_ids().

All cases from instrumentation.md §3.1 are covered.
"""

import pytest
from adapter_slots.metrics.war import compute_war, compute_war_from_ids, WARP_SIZE


# Helpers

def make_tokens(adapter_ids):
    """Convert a list of ints to token dicts."""
    return [{"adapter_id": a} for a in adapter_ids]


# compute_war

class TestComputeWar:

    def test_empty_batch(self):
        assert compute_war([]) == 0.0

    def test_fewer_than_one_warp(self):
        # 16 tokens < warp_size=32 → no complete warp → 0.0
        tokens = make_tokens([0] * 16)
        assert compute_war(tokens) == 0.0

    def test_single_warp_all_same_adapter(self):
        tokens = make_tokens([0] * 32)
        assert compute_war(tokens) == 1.0

    def test_single_warp_alternating_adapters(self):
        tokens = make_tokens([0, 1] * 16)   # 32 tokens, ABAB... → not aligned
        assert compute_war(tokens) == 0.0

    def test_two_warps_both_aligned(self):
        # [32×adapter_0, 32×adapter_1] → both warps aligned → WAR = 1.0
        tokens = make_tokens([0] * 32 + [1] * 32)
        assert compute_war(tokens) == 1.0

    def test_two_warps_one_aligned_one_not(self):
        # [32×0] aligned + [16×0, 16×1] mixed → WAR = 0.5
        tokens = make_tokens([0] * 32 + [0] * 16 + [1] * 16)
        assert compute_war(tokens) == 0.5

    def test_partial_tail_ignored(self):
        # 48 tokens with warp_size=32: 1 complete warp + 16-token tail
        # warp0 = [0]*32 (aligned); tail is ignored
        tokens = make_tokens([0] * 32 + [1] * 16)
        assert compute_war(tokens) == 1.0

    def test_custom_warp_size_aligned(self):
        # W=4: [0,0,0,0, 1,1,1,1] → 2 aligned warps → 1.0
        tokens = make_tokens([0] * 4 + [1] * 4)
        assert compute_war(tokens, warp_size=4) == 1.0

    def test_custom_warp_size_mixed(self):
        # W=4: [0,1,0,1, 0,1,0,1] → 0 aligned → 0.0
        tokens = make_tokens([0, 1, 0, 1, 0, 1, 0, 1])
        assert compute_war(tokens, warp_size=4) == 0.0

    def test_war_range(self):
        # WAR must be in [0, 1] for any batch
        import random
        random.seed(0)
        for _ in range(100):
            n = random.randint(1, 256)
            ids = [random.randint(0, 3) for _ in range(n)]
            w = compute_war(make_tokens(ids))
            assert 0.0 <= w <= 1.0

    def test_war_constant_adapter_large_batch(self):
        # 512 tokens, all same adapter → WAR = 1.0
        tokens = make_tokens([2] * 512)
        assert compute_war(tokens) == 1.0

    def test_war_four_adapters_interleaved(self):
        # Cycle 0,1,2,3 repeated → each warp has all 4 adapters → WAR = 0.0
        tokens = make_tokens([i % 4 for i in range(128)])
        assert compute_war(tokens) == 0.0


# compute_war_from_ids (fast NumPy path)

class TestComputeWarFromIds:

    def test_empty(self):
        assert compute_war_from_ids([]) == 0.0

    def test_matches_dict_version_aligned(self):
        ids = [0] * 64 + [1] * 64
        assert compute_war_from_ids(ids) == 1.0

    def test_matches_dict_version_mixed(self):
        ids = [i % 4 for i in range(128)]
        assert compute_war_from_ids(ids) == 0.0

    def test_matches_dict_version_partial(self):
        import random
        random.seed(7)
        ids = [random.randint(0, 3) for _ in range(256)]
        tokens = [{"adapter_id": a} for a in ids]
        assert abs(compute_war_from_ids(ids) - compute_war(tokens)) < 1e-9

    def test_single_adapter_all(self):
        assert compute_war_from_ids([3] * 32) == 1.0

    def test_below_warp_size(self):
        assert compute_war_from_ids([0, 1, 0]) == 0.0
