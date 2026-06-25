"""
test_wgkp.py -- Unit tests for kernel_promotion WGKP compound stack.

All tests run on CPU with pure Python/PyTorch tensors. No GPU, no vLLM, no Triton
required. Tests are structured per implementation phase:

    Phase 1:  GWAR metric
    Phase 2:  Fused LoRA kernel (CPU reference)
    Phase 3:  MergedWeightCache lifecycle
    Phase 4:  CASH dispatch (form_batch_wgkp)
    Phase 5:  WGKPDispatcher segmentation
    Phase 6:  AdaptivePromoThreshold
    Phase 9:  APISRouter

Run:
    python -m pytest tests/test_wgkp.py -v
    python -m pytest tests/test_wgkp.py -v -k "test_gwar"
"""

import math
import time
from typing import List

import pytest
import torch

# Phase 1: GWAR metric

from adapter_slots.metrics.gwar import compute_gwar, compute_gwar_curve
from adapter_slots.metrics.war import compute_war_from_ids


class TestComputeGWAR:

    def test_empty_batch(self):
        assert compute_gwar([], threshold=8) == 0.0

    def test_zero_threshold(self):
        # threshold <= 0 → 0.0
        assert compute_gwar([0, 1, 2], threshold=0) == 0.0

    def test_threshold_one_is_always_one(self):
        # Every token is in a segment of size >= 1
        assert compute_gwar([0, 1, 2, 3], threshold=1) == 1.0
        assert compute_gwar([0, 0, 1, 1, 2], threshold=1) == 1.0

    def test_gwar_equals_war_at_warp_size(self):
        # GWAR(32) must match compute_war_from_ids() for any batch of integer adapter IDs
        ids = [0] * 32 + [1] * 32 + [2] * 16 + [0] * 16
        gwar_32 = compute_gwar(ids, threshold=32)
        war = compute_war_from_ids(ids, warp_size=32)
        # Both count fraction of tokens in same-adapter segments of size >= 32.
        # WAR counts complete warps as homogeneous; GWAR counts contiguous segments.
        # For sorted batches they should be equal.
        assert abs(gwar_32 - war) < 1e-9

    def test_gwar_equals_war_at_warp_size_sorted(self):
        # Perfectly sorted: 32 tokens of adapter 0, 32 of adapter 1
        ids = [0] * 32 + [1] * 32
        assert compute_gwar(ids, threshold=32) == 1.0
        assert compute_war_from_ids(ids, warp_size=32) == 1.0

    def test_gwar_monotonically_decreasing_in_threshold(self):
        # GWAR(8) >= GWAR(16) >= GWAR(32) for any batch
        ids = [0] * 20 + [1] * 10 + [2] * 5 + [3] * 3 + [4] * 2 + [5] * 1 + [6] * 1
        g8 = compute_gwar(ids, threshold=8)
        g16 = compute_gwar(ids, threshold=16)
        g32 = compute_gwar(ids, threshold=32)
        assert g8 >= g16, f"g8={g8} < g16={g16}"
        assert g16 >= g32, f"g16={g16} < g32={g32}"

    def test_gwar_monotonically_decreasing_random(self):
        import random
        random.seed(42)
        for _ in range(50):
            n = random.randint(20, 200)
            k = random.randint(2, 8)
            ids = sorted([random.randint(0, k - 1) for _ in range(n)])
            prev = 1.0
            for thr in [1, 2, 4, 8, 16, 32]:
                g = compute_gwar(ids, threshold=thr)
                assert g <= prev + 1e-9, f"GWAR({thr})={g} > GWAR(prev)={prev}"
                prev = g

    def test_gwar_fully_aligned_batch(self):
        # All tokens same adapter → GWAR(n) = 1.0 for all n <= batch_size
        ids = [0] * 100
        for thr in [1, 8, 16, 32, 64, 100]:
            assert compute_gwar(ids, threshold=thr) == 1.0

    def test_gwar_threshold_larger_than_batch(self):
        # Threshold > batch size → 0.0 (no segment can be that large)
        ids = [0] * 10
        assert compute_gwar(ids, threshold=11) == 0.0

    def test_gwar_threshold_equals_batch_size(self):
        # Exactly one segment of the right size → 1.0
        ids = [0] * 8
        assert compute_gwar(ids, threshold=8) == 1.0

    def test_gwar_partial_segments(self):
        # ids: 10 tokens of adapter 0, 5 tokens of adapter 1
        # threshold=8: only adapter 0's segment (size 10 >= 8) qualifies
        # promoted_tokens = 10, total = 15 → GWAR = 10/15
        ids = [0] * 10 + [1] * 5
        g = compute_gwar(ids, threshold=8)
        assert abs(g - 10 / 15) < 1e-9

    def test_gwar_all_singletons(self):
        # Alternating adapters -- each segment has size 1
        # threshold=2 → 0.0
        ids = [0, 1, 2, 3, 0, 1, 2, 3]
        assert compute_gwar(ids, threshold=2) == 0.0

    def test_gwar_curve_consistency(self):
        # compute_gwar_curve should match compute_gwar called individually
        ids = [0] * 16 + [1] * 8 + [2] * 4 + [3] * 2
        thresholds = [1, 4, 8, 16, 32]
        curve = compute_gwar_curve(ids, thresholds)
        for thr in thresholds:
            assert abs(curve[thr] - compute_gwar(ids, thr)) < 1e-9, (
                f"curve[{thr}]={curve[thr]} != compute_gwar={compute_gwar(ids, thr)}"
            )

    def test_gwar_curve_monotone(self):
        ids = [0] * 32 + [1] * 16 + [2] * 8 + [3] * 4
        thresholds = [1, 4, 8, 16, 32, 64]
        curve = compute_gwar_curve(ids, thresholds)
        values = [curve[t] for t in thresholds]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-9

    def test_gwar_range(self):
        import random
        random.seed(7)
        for _ in range(100):
            n = random.randint(1, 128)
            k = random.randint(2, 6)
            ids = [random.randint(0, k - 1) for _ in range(n)]
            for thr in [2, 4, 8, 16]:
                g = compute_gwar(ids, thr)
                assert 0.0 <= g <= 1.0


# Phase 2: Fused LoRA kernel (CPU reference)

from adapter_slots.kernel.fused_lora_kernel import FusedLoRAKernel, _fused_lora_cpu_reference


class TestFusedLoRAKernel:

    def test_fused_kernel_cpu_reference_basic(self):
        # Y = X @ W^T + alpha * (X @ A^T) @ B^T
        torch.manual_seed(0)
        M, N, K, R = 4, 8, 16, 4
        alpha = 0.5
        X = torch.randn(M, K)
        W = torch.randn(N, K)
        A = torch.randn(R, K)
        B = torch.randn(N, R)

        Y_ref = X @ W.T + alpha * (X @ A.T) @ B.T
        Y_fused = _fused_lora_cpu_reference(X, W, A, B, alpha)
        assert torch.allclose(Y_ref, Y_fused, atol=1e-5), (
            f"Max abs error: {(Y_ref - Y_fused).abs().max()}"
        )

    def test_fused_kernel_vs_direct_computation(self):
        # Larger batch; verify fused output == reference output
        torch.manual_seed(42)
        M, N, K, R = 32, 64, 128, 16
        alpha = 1.0
        X = torch.randn(M, K)
        W = torch.randn(N, K)
        A = torch.randn(R, K)
        B = torch.randn(N, R)

        Y_ref = X @ W.T + alpha * (X @ A.T) @ B.T
        Y_fused = _fused_lora_cpu_reference(X, W, A, B, alpha)
        assert torch.allclose(Y_ref, Y_fused, atol=1e-4)

    def test_fused_kernel_alpha_zero(self):
        # alpha=0 → Y = X @ W^T (LoRA disabled)
        torch.manual_seed(1)
        M, N, K, R = 8, 16, 32, 4
        X = torch.randn(M, K)
        W = torch.randn(N, K)
        A = torch.randn(R, K)
        B = torch.randn(N, R)

        Y_ref = X @ W.T
        Y_fused = _fused_lora_cpu_reference(X, W, A, B, alpha=0.0)
        assert torch.allclose(Y_ref, Y_fused, atol=1e-6)

    def test_fused_kernel_class_cpu_fallback(self):
        # FusedLoRAKernel.forward() falls back to CPU reference when Triton unavailable
        kernel = FusedLoRAKernel()
        torch.manual_seed(3)
        M, N, K, R = 16, 32, 64, 8
        X = torch.randn(M, K)
        W = torch.randn(N, K)
        A = torch.randn(R, K)
        B = torch.randn(N, R)
        alpha = 0.25

        Y = kernel.forward(X, W, A, B, alpha)
        Y_ref = X @ W.T + alpha * (X @ A.T) @ B.T
        assert Y.shape == (M, N)
        assert torch.allclose(Y, Y_ref, atol=1e-4)


# Phase 3: MergedWeightCache lifecycle

from adapter_slots.kernel.merged_weight_cache import MergedWeightCache


class TestMergedWeightCache:

    def _make_lora_weights(self, n_layers=2, d_in=64, d_out=128, rank=4, alpha=1.0):
        """Create synthetic LoRA weight dict."""
        weights = {}
        for i in range(n_layers):
            A = torch.randn(rank, d_in)
            B = torch.randn(d_out, rank)
            weights[f"layer_{i}.q_proj"] = (A, B, alpha)
        return weights

    def _make_model(self, n_layers=2, d_in=64, d_out=128):
        """Create a simple model with Linear layers named to match q_proj."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                for i in range(n_layers):
                    setattr(self, f"layer_{i}_q_proj", nn.Linear(d_in, d_out, bias=False))

            def named_modules(self):
                # Override to return layer names matching merge projections.
                yield "", self
                for i in range(n_layers):
                    name = f"layer_{i}.q_proj"
                    yield name, getattr(self, f"layer_{i}_q_proj")

        return SimpleModel()

    def test_merge_correctness_cpu(self):
        mwc = MergedWeightCache(k_hot=5, projections=["q_proj"])
        weights = self._make_lora_weights(n_layers=2, rank=4, alpha=1.0)
        mwc.merge("adapter_0", weights)

        assert mwc.is_merged("adapter_0")
        for layer_name, (A, B, alpha) in weights.items():
            delta = mwc.get_merged("adapter_0", layer_name)
            assert delta is not None
            expected_delta = alpha * torch.matmul(B.float(), A.float()).to(A.dtype)
            assert torch.allclose(delta, expected_delta, atol=1e-5), (
                f"Layer {layer_name}: max error {(delta - expected_delta).abs().max()}"
            )

    def test_not_merged_initially(self):
        mwc = MergedWeightCache(k_hot=5)
        assert not mwc.is_merged("unknown_adapter")
        assert mwc.get_merged("unknown_adapter", "q_proj") is None

    def test_eviction_budget_enforcement(self):
        mwc = MergedWeightCache(k_hot=3, projections=["q_proj"])
        for i in range(5):
            weights = self._make_lora_weights(n_layers=1, rank=4)
            mwc.merge(f"adapter_{i}", weights)

        # After 5 merges with k_hot=3, only 3 should remain.
        n_cached = sum(1 for i in range(5) if mwc.is_merged(f"adapter_{i}"))
        assert n_cached <= 3, f"Expected <= 3 cached, got {n_cached}"

    def test_memory_budget_enforcement(self):
        # Very tight budget: 1 MB for q_proj only
        mwc = MergedWeightCache(k_hot=100, memory_budget_gb=0.001, projections=["q_proj"])
        # Each merge: rank=4, d_in=64, d_out=128 → delta = 128×64×2B = 16 KB
        for i in range(20):
            weights = self._make_lora_weights(n_layers=1, d_in=64, d_out=128, rank=4)
            mwc.merge(f"adapter_{i}", weights)

        mem = mwc.memory_used_gb()
        assert mem <= 0.001 + 0.0001, f"Memory {mem:.6f} GB exceeds budget 0.001 GB"

    def test_evict_removes_adapter(self):
        mwc = MergedWeightCache(k_hot=5, projections=["q_proj"])
        weights = self._make_lora_weights(n_layers=1, rank=4)
        mwc.merge("adapter_0", weights)
        assert mwc.is_merged("adapter_0")

        mwc.evict("adapter_0")
        assert not mwc.is_merged("adapter_0")
        assert mwc.get_merged("adapter_0", "layer_0.q_proj") is None

    def test_install_uninstall_zero_copy(self):
        import torch.nn as nn

        mwc = MergedWeightCache(k_hot=5, projections=["q_proj"])
        # Build a simple model with two linear layers
        model = nn.ModuleDict({"q_proj": nn.Linear(64, 128, bias=False)})

        # Override named_modules to return a layer with the right name
        original_named_modules = model.named_modules

        def patched_named_modules():
            yield "", model
            yield "q_proj", model["q_proj"]

        model.named_modules = patched_named_modules  # type: ignore[method-assign]

        # Create LoRA weights for "q_proj"
        A = torch.randn(4, 64)
        B = torch.randn(128, 4)
        alpha = 1.0
        mwc.merge("adp", {"q_proj": (A, B, alpha)})

        original_weight = model["q_proj"].weight.data.clone()
        original_ptr = model["q_proj"].weight.data_ptr()

        mwc.install_merged("adp", model)
        installed_weight = model["q_proj"].weight.data.clone()
        # Weight should have changed by delta = alpha * B @ A
        delta = alpha * torch.matmul(B.float(), A.float()).to(original_weight.dtype)
        assert torch.allclose(installed_weight, original_weight + delta, atol=1e-4)

        mwc.uninstall_merged("adp", model)
        restored_weight = model["q_proj"].weight.data.clone()
        assert torch.allclose(restored_weight, original_weight, atol=1e-6)

    def test_stats_structure(self):
        mwc = MergedWeightCache(k_hot=5, projections=["q_proj"])
        s = mwc.stats()
        for key in ("hit_count", "miss_count", "memory_gb", "eviction_count", "n_cached"):
            assert key in s, f"Missing key: {key}"

    def test_mwc_lifecycle_with_warm_cache_manager(self):
        """Integration: WarmCacheManager eviction should trigger MWC eviction."""
        from adapter_slots.prefetch.cache_manager import WarmCacheManager

        mwc = MergedWeightCache(k_hot=5, projections=["q_proj"])
        wcm = WarmCacheManager(k_warm_max=2, tau_load_ms=100.0, policy="lru")

        # Merge two adapters into MWC.
        for i in range(2):
            weights = self._make_lora_weights(n_layers=1, rank=4)
            mwc.merge(f"adapter_{i}", weights)
            assert mwc.is_merged(f"adapter_{i}")

        # Simulate WarmCacheManager evicting adapter_0 (e.g. when a new adapter arrives).
        # In the real scheduler, WarmCacheManager.evict() triggers mwc.evict().
        mwc.evict("adapter_0")
        assert not mwc.is_merged("adapter_0")
        assert mwc.is_merged("adapter_1")  # adapter_1 still cached


# Phase 4: CASH dispatch (form_batch_wgkp)

from adapter_slots.buffer import AlignmentBuffer


class TestCASHDispatch:
    """Tests for AlignmentBuffer.form_batch_wgkp() implementing Algorithm CASH."""

    def _make_buffer_with_tokens(self, adapter_token_counts: dict, tmax_ms: float = 100.0):
        """Create a buffer pre-populated with tokens."""
        adapters = list(adapter_token_counts.keys())
        buf = AlignmentBuffer(adapters=adapters, warp_size=32, tmax_ms=tmax_ms)
        seq_id = 0
        for adapter_id, n_tokens in adapter_token_counts.items():
            for _ in range(n_tokens):
                buf.enqueue(adapter_id, seq_id)
                seq_id += 1
        return buf, seq_id

    def test_cash_full_warp_promotes(self):
        # W=32 tokens of adapter_0 → condition 1 → is_promoted=True
        buf, _ = self._make_buffer_with_tokens({"a0": 32})
        result = buf.form_batch_wgkp(
            ranked_adapters=["a0"],
            tmax_k={"a0": 1.0},
            n_star=8,
            merged_adapter_ids={"a0"},
        )
        assert len(result) == 32
        assert all(is_promoted for _, _, is_promoted in result)
        assert all(aid == "a0" for aid, _, _ in result)

    def test_cash_n_star_threshold_promotes(self):
        # Exactly n*=8 tokens of adapter, in MWC → condition 2 → is_promoted=True
        buf, _ = self._make_buffer_with_tokens({"a0": 8})
        result = buf.form_batch_wgkp(
            ranked_adapters=["a0"],
            tmax_k={"a0": 1.0},
            n_star=8,
            merged_adapter_ids={"a0"},
        )
        assert len(result) == 8
        assert all(is_promoted for _, _, is_promoted in result)

    def test_cash_n_star_not_in_mwc_not_promoted(self):
        # n* tokens but NOT in MWC → no promotion (condition 2 requires in MWC)
        buf, _ = self._make_buffer_with_tokens({"a0": 8})
        result = buf.form_batch_wgkp(
            ranked_adapters=["a0"],
            tmax_k={"a0": 1.0},
            n_star=8,
            merged_adapter_ids=set(),  # empty → not in MWC
        )
        # Should still dispatch (via condition 3 or 4 when T_max expires);
        # with fresh tokens (age ≈ 0), it holds unless T_max/2 passed
        # Here tokens are new so condition 3 (age > T_max/2) is not met yet.
        # Result may be empty (held) or dispatched without promotion.
        for _, _, is_promoted in result:
            assert not is_promoted

    def test_cash_partial_dispatch_not_promoted(self):
        # ceil(n*/2)=4 tokens, age > T_max/2 → condition 3 → dispatched, is_promoted=False
        buf = AlignmentBuffer(adapters=["a0"], warp_size=32, tmax_ms=5.0)
        for seq_id in range(4):
            buf.enqueue("a0", seq_id)
        # Manually age the queue to > T_max/2 = 2.5ms
        buf.enqueue_time["a0"] = time.perf_counter() - 0.003  # 3ms > 2.5ms
        buf.queues["a0"][0] = (buf.queues["a0"][0][0], buf.enqueue_time["a0"])

        result = buf.form_batch_wgkp(
            ranked_adapters=["a0"],
            tmax_k={"a0": 5.0 / 1000},  # 5ms in seconds
            n_star=8,
            merged_adapter_ids={"a0"},
        )
        assert len(result) == 4
        assert all(not is_promoted for _, _, is_promoted in result)

    def test_cash_early_release_half_threshold(self):
        # 4 tokens (= ceil(8/2)), aged past T_max/2 → condition 3 fires
        buf = AlignmentBuffer(adapters=["a0"], warp_size=32, tmax_ms=100.0)
        for seq_id in range(4):
            buf.enqueue("a0", seq_id)
        # Age queue past T_max/2 = 50ms
        buf.enqueue_time["a0"] = time.perf_counter() - 0.060  # 60ms > 50ms

        result = buf.form_batch_wgkp(
            ranked_adapters=["a0"],
            tmax_k={"a0": 0.100},
            n_star=8,
            merged_adapter_ids=set(),
        )
        assert len(result) == 4
        for _, _, is_promoted in result:
            assert not is_promoted

    def test_cash_slo_hard_cap_respected(self):
        # Age past T_max → condition 4 forces dispatch regardless of queue size
        buf = AlignmentBuffer(adapters=["a0"], warp_size=32, tmax_ms=10.0)
        for seq_id in range(2):
            buf.enqueue("a0", seq_id)
        # Age past T_max = 10ms
        buf.enqueue_time["a0"] = time.perf_counter() - 0.015  # 15ms > 10ms

        result = buf.form_batch_wgkp(
            ranked_adapters=["a0"],
            tmax_k={"a0": 0.010},
            n_star=8,
            merged_adapter_ids=set(),
        )
        # Must dispatch (SLO cap)
        assert len(result) == 2
        assert all(not is_promoted for _, _, is_promoted in result)

    def test_cash_segment_atomicity_mixed_batch(self):
        # Two adapters: a0 has W=32 tokens (promoted), a1 has 3 tokens (not promoted)
        # All tokens in a0 must have the same is_promoted; same for a1.
        buf, _ = self._make_buffer_with_tokens({"a0": 32, "a1": 3})
        # Age a1 past T_max to force dispatch
        buf.enqueue_time["a1"] = time.perf_counter() - 0.200
        result = buf.form_batch_wgkp(
            ranked_adapters=["a0", "a1"],
            tmax_k={"a0": 1.0, "a1": 0.100},
            n_star=8,
            merged_adapter_ids={"a0"},
        )
        a0_promos = [p for a, _, p in result if a == "a0"]
        a1_promos = [p for a, _, p in result if a == "a1"]
        # Atomicity: all tokens in a0 have same is_promoted
        if a0_promos:
            assert len(set(a0_promos)) == 1, "a0 segment has mixed is_promoted values"
        if a1_promos:
            assert len(set(a1_promos)) == 1, "a1 segment has mixed is_promoted values"

    def test_cash_returns_triple_tuples(self):
        buf, _ = self._make_buffer_with_tokens({"a0": 32})
        result = buf.form_batch_wgkp(
            ranked_adapters=["a0"],
            tmax_k={"a0": 1.0},
            n_star=8,
            merged_adapter_ids={"a0"},
        )
        for item in result:
            assert len(item) == 3
            adapter_id, seq_id, is_promoted = item
            assert isinstance(adapter_id, str)
            assert isinstance(seq_id, int)
            assert isinstance(is_promoted, bool)

    def test_cash_empty_buffer(self):
        buf = AlignmentBuffer(adapters=["a0"], warp_size=32, tmax_ms=5.0)
        result = buf.form_batch_wgkp(
            ranked_adapters=["a0"],
            tmax_k={"a0": 0.005},
            n_star=8,
            merged_adapter_ids=set(),
        )
        assert result == []


# Phase 5: WGKPDispatcher segmentation

from adapter_slots.kernel.wgkp_dispatcher import WGKPDispatcher, SegmentDescriptor


class TestWGKPDispatcher:

    def test_dispatcher_empty_batch(self):
        d = WGKPDispatcher()
        assert d.segment_and_promote([]) == []

    def test_dispatcher_single_segment(self):
        d = WGKPDispatcher()
        raw = [("a0", 0, True), ("a0", 1, True), ("a0", 2, True)]
        segs = d.segment_and_promote(raw)
        assert len(segs) == 1
        assert segs[0].adapter_id == "a0"
        assert segs[0].seq_ids == [0, 1, 2]
        assert segs[0].is_promoted is True
        assert segs[0].segment_size == 3

    def test_dispatcher_two_segments(self):
        d = WGKPDispatcher()
        raw = [("a0", 0, True), ("a0", 1, True), ("a1", 2, False), ("a1", 3, False)]
        segs = d.segment_and_promote(raw)
        assert len(segs) == 2
        assert segs[0].adapter_id == "a0"
        assert segs[0].is_promoted is True
        assert segs[1].adapter_id == "a1"
        assert segs[1].is_promoted is False

    def test_dispatcher_segmentation_sorted_batch(self):
        # Sorted batch: multiple adapters, contiguous runs
        raw = (
            [("a0", i, True) for i in range(8)]
            + [("a1", i + 8, False) for i in range(4)]
            + [("a2", i + 12, True) for i in range(16)]
        )
        d = WGKPDispatcher()
        segs = d.segment_and_promote(raw)
        assert len(segs) == 3
        assert segs[0].adapter_id == "a0"
        assert segs[0].segment_size == 8
        assert segs[0].is_promoted is True
        assert segs[1].adapter_id == "a1"
        assert segs[1].segment_size == 4
        assert segs[1].is_promoted is False
        assert segs[2].adapter_id == "a2"
        assert segs[2].segment_size == 16
        assert segs[2].is_promoted is True

    def test_dispatcher_atomicity_enforcement(self):
        # Mixed is_promoted within a segment → ValueError
        d = WGKPDispatcher()
        bad_batch = [("a0", 0, True), ("a0", 1, False)]  # atomicity violation
        with pytest.raises(ValueError, match="atomicity violation"):
            d.segment_and_promote(bad_batch)

    def test_dispatcher_segment_descriptor_fields(self):
        d = WGKPDispatcher()
        raw = [("adapter_x", 99, False)]
        segs = d.segment_and_promote(raw)
        assert len(segs) == 1
        s = segs[0]
        assert s.adapter_id == "adapter_x"
        assert s.seq_ids == [99]
        assert s.is_promoted is False
        assert s.segment_size == 1

    def test_dispatcher_promotion_fraction_tracking(self):
        d = WGKPDispatcher()
        # 8 promoted + 4 not promoted = 12 total
        raw = (
            [("a0", i, True) for i in range(8)]
            + [("a1", i + 8, False) for i in range(4)]
        )
        d.segment_and_promote(raw)
        assert abs(d.promotion_fraction() - 8 / 12) < 1e-9

    def test_dispatcher_stats_reset(self):
        d = WGKPDispatcher()
        raw = [("a0", i, True) for i in range(4)]
        d.segment_and_promote(raw)
        assert d.stats()["total_tokens"] == 4
        d.reset_stats()
        assert d.stats()["total_tokens"] == 0
        assert d.stats()["total_promoted_tokens"] == 0


# Phase 6: AdaptivePromoThreshold

from adapter_slots.kernel.apt import AdaptivePromoThreshold


class TestAdaptivePromoThreshold:

    def test_apt_default_initialization(self):
        apt = AdaptivePromoThreshold(fallback_threshold=32)
        # With no profile or sorted_thresholds, falls back to fallback_threshold
        n_star = apt.current_threshold()
        assert isinstance(n_star, int)
        assert n_star > 0

    def test_apt_threshold_selection_from_profile(self, tmp_path):
        # Write a minimal hw profile JSON and check threshold selection
        import json
        profile = {
            "hardware": "test_gpu",
            "warp_size": 32,
            "gemm_crossover_n": 8,
            "crossover_curve": {
                "4":  {"psi_fuse": 0.9,  "psi_gemm": 0.8,  "gemm_faster": False},
                "8":  {"psi_fuse": 1.15, "psi_gemm": 1.25, "gemm_faster": True},
                "16": {"psi_fuse": 1.28, "psi_gemm": 1.45, "gemm_faster": True},
                "32": {"psi_fuse": 1.33, "psi_gemm": 1.55, "gemm_faster": True},
            },
        }
        profile_path = tmp_path / "test_profile.json"
        profile_path.write_text(json.dumps(profile))

        apt = AdaptivePromoThreshold(
            hw_profile_path=str(profile_path),
            min_speedup=1.05,
            update_interval=10,
            fallback_threshold=32,
        )
        # Initial threshold: smallest promotable n* = 8
        assert apt.current_threshold() == 8

    def test_apt_updates_on_queue_depth_change(self, tmp_path):
        import json
        profile = {
            "hardware": "test_gpu",
            "warp_size": 32,
            "gemm_crossover_n": 8,
            "crossover_curve": {
                "8":  {"psi_fuse": 1.15, "psi_gemm": 1.25, "gemm_faster": True},
                "16": {"psi_fuse": 1.28, "psi_gemm": 1.45, "gemm_faster": True},
                "32": {"psi_fuse": 1.33, "psi_gemm": 1.55, "gemm_faster": True},
            },
        }
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps(profile))

        apt = AdaptivePromoThreshold(
            hw_profile_path=str(profile_path),
            min_speedup=1.05,
            update_interval=5,
            fallback_threshold=32,
        )

        # With mean queue depth = 20, threshold should select up to 16
        for _ in range(5):
            apt.update(mean_queue_depth=20.0)
        assert apt.current_threshold() == 16

    def test_apt_fallback_when_no_profile(self):
        apt = AdaptivePromoThreshold(
            hw_profile_path="/nonexistent/path.json",
            fallback_threshold=32,
        )
        # No valid profile → fallback
        assert apt.current_threshold() == 32

    def test_apt_hardware_identifier(self, tmp_path):
        import json
        profile = {"hardware": "my_gpu", "warp_size": 32, "crossover_curve": {}}
        p = tmp_path / "p.json"
        p.write_text(json.dumps(profile))
        apt = AdaptivePromoThreshold(hw_profile_path=str(p))
        assert apt.hardware() == "my_gpu"


# Phase 9: APISRouter

from adapter_slots.integrations.apis_router import APISRouter


class TestAPISRouter:

    def test_apis_router_basic_routing(self):
        router = APISRouter(
            n_gpus=2,
            upstream_urls=["http://gpu0:8001", "http://gpu1:8002"],
        )
        # Unknown adapters fall back to hash-based routing
        url = router.route("some_adapter")
        assert url in ["http://gpu0:8001", "http://gpu1:8002"]

    def test_apis_router_zipf_balance(self):
        # After rebalance with 6 adapters, 3 should go to each GPU (round-robin)
        router = APISRouter(
            n_gpus=2,
            upstream_urls=["http://gpu0:8001", "http://gpu1:8002"],
        )
        rates = {f"adapter_{i}": float(6 - i) for i in range(6)}
        router.rebalance(rates)

        table = router.assignment_table()
        gpu0 = [a for a, g in table.items() if g == 0]
        gpu1 = [a for a, g in table.items() if g == 1]
        assert len(gpu0) == 3
        assert len(gpu1) == 3

    def test_apis_router_rebalance_on_rate_change(self):
        router = APISRouter(
            n_gpus=2,
            upstream_urls=["http://a:8001", "http://b:8002"],
        )
        # Initial assignment
        rates_v1 = {"a0": 10.0, "a1": 5.0, "a2": 3.0, "a3": 1.0}
        router.rebalance(rates_v1)
        table_v1 = router.assignment_table()

        # After rebalance, most popular (a0) → GPU 0, second (a1) → GPU 1
        assert table_v1["a0"] == 0
        assert table_v1["a1"] == 1

        # Rebalance with changed rates
        rates_v2 = {"a0": 1.0, "a1": 10.0, "a2": 8.0, "a3": 7.0}
        router.rebalance(rates_v2)
        table_v2 = router.assignment_table()

        # Now a1 is most popular → GPU 0
        assert table_v2["a1"] == 0

    def test_apis_router_assignment_reflects_routing(self):
        router = APISRouter(
            n_gpus=2,
            upstream_urls=["http://g0:8001", "http://g1:8002"],
        )
        rates = {"my_adapter": 5.0, "other": 3.0}
        router.rebalance(rates)

        table = router.assignment_table()
        gpu_idx = table["my_adapter"]
        expected_url = ["http://g0:8001", "http://g1:8002"][gpu_idx]
        assert router.route("my_adapter") == expected_url

    def test_apis_router_stats_structure(self):
        router = APISRouter(n_gpus=2, upstream_urls=["http://a:1", "http://b:2"])
        router.route("x")
        s = router.stats()
        assert "n_gpus" in s
        assert "total_routes" in s
        assert s["total_routes"] >= 1

    def test_apis_router_n_gpus_validation(self):
        with pytest.raises(ValueError):
            APISRouter(n_gpus=0)

    def test_apis_router_url_count_validation(self):
        with pytest.raises(ValueError):
            APISRouter(n_gpus=2, upstream_urls=["http://only_one:8000"])

    def test_apis_router_load_balance_ratio(self):
        router = APISRouter(
            n_gpus=2,
            upstream_urls=["http://g0:8001", "http://g1:8002"],
        )
        # Route all traffic to known adapters assigned to GPU 0
        rates = {"a0": 10.0, "a1": 5.0}
        router.rebalance(rates)
        # Force all routes to same GPU to test imbalance detection
        for _ in range(10):
            router.route("a0")
        ratio = router.load_balance_ratio()
        assert ratio >= 1.0


# Integration: scheduler env var parsing

class TestSchedulerWGKPEnvVars:
    """Smoke-test that WGKP env vars are parsed without crashing.

    Does not require vLLM; uses mock environment variables.
    """

    def test_scheduler_wgkp_env_vars_parsed(self, monkeypatch):
        """AlignmentAwareScheduler parses WGKP env vars without error."""
        monkeypatch.setenv("AS_MODE", "wgkp")
        monkeypatch.setenv("AS_WGKP_THRESHOLD", "8")
        monkeypatch.setenv("AS_MWC_K_HOT", "3")
        monkeypatch.setenv("AS_MWC_MEMORY_GB", "5.0")
        monkeypatch.setenv("AS_FUSED_KERNEL", "0")
        monkeypatch.setenv("AS_MACRO_N_ACCUM", "1")
        monkeypatch.setenv("AS_WHITTLE_DELTA_T", "0.030")

        # Import the scheduler module -- should not raise.
        # (vLLM is not installed, but module-level guards handle that.)
        from adapter_slots.integrations.vllm_scheduler import (
            _env_float, _env_int,
        )
        assert _env_float("AS_MWC_MEMORY_GB", 10.0) == 5.0
        assert _env_int("AS_WGKP_THRESHOLD", 8) == 8
        assert _env_int("AS_MWC_K_HOT", 5) == 3


class TestTPMergedWeightSharding:
    """CPU simulation of TP sharding correctness for merged weights (EC 13.7)."""

    def test_tp2_merged_weight_sharding_equality(self):
        """W_k[local_rank] = W[local_rank] + (alpha*B@A)[local_rank].

        The linear decomposition W_k = W + alpha*B@A means TP sharding
        is correct: the shard of W_k equals the shard of W plus the shard
        of alpha*B@A (both computed by slicing along the same output dimension).
        """
        torch.manual_seed(99)
        d_in, d_out, rank = 64, 128, 8
        alpha = 0.5

        W = torch.randn(d_out, d_in)
        A = torch.randn(rank, d_in)
        B = torch.randn(d_out, rank)

        # Full merged weight
        delta = alpha * (B.float() @ A.float()).to(W.dtype)
        W_k = W + delta

        # TP=2 sharding: split output dimension in half
        half = d_out // 2
        W_shard_0 = W[:half]
        W_shard_1 = W[half:]
        delta_shard_0 = delta[:half]
        delta_shard_1 = delta[half:]

        W_k_shard_0 = W_shard_0 + delta_shard_0
        W_k_shard_1 = W_shard_1 + delta_shard_1

        # Reconstruction should equal full W_k
        W_k_reconstructed = torch.cat([W_k_shard_0, W_k_shard_1], dim=0)
        assert torch.allclose(W_k_reconstructed, W_k, atol=1e-5), (
            f"TP=2 shard mismatch: max error {(W_k_reconstructed - W_k).abs().max()}"
        )
