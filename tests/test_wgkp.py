"""
test_wgkp.py -- WGKP components on the CPU (no GPU, vLLM or Triton needed):
GWAR, the fused kernel's CPU reference, CASH holdback
dispatch (form_batch_wgkp), segmentation and the adaptive promotion threshold.
"""

import time

import pytest
import torch

# GWAR metric

from adapter_slots.metrics.gwar import compute_gwar, compute_gwar_curve
from adapter_slots.metrics.war import compute_war_from_ids


class TestComputeGWAR:

    def test_empty_batch(self):
        assert compute_gwar([], threshold=8) == 0.0

    def test_zero_threshold(self):
        # threshold <= 0 -> 0.0
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
        # All tokens same adapter -> GWAR(n) = 1.0 for all n <= batch_size
        ids = [0] * 100
        for thr in [1, 8, 16, 32, 64, 100]:
            assert compute_gwar(ids, threshold=thr) == 1.0

    def test_gwar_threshold_larger_than_batch(self):
        # Threshold > batch size -> 0.0 (no segment can be that large)
        ids = [0] * 10
        assert compute_gwar(ids, threshold=11) == 0.0

    def test_gwar_threshold_equals_batch_size(self):
        # Exactly one segment of the right size -> 1.0
        ids = [0] * 8
        assert compute_gwar(ids, threshold=8) == 1.0

    def test_gwar_partial_segments(self):
        # ids: 10 tokens of adapter 0, 5 tokens of adapter 1
        # threshold=8: only adapter 0's segment (size 10 >= 8) qualifies
        # promoted_tokens = 10, total = 15 -> GWAR = 10/15
        ids = [0] * 10 + [1] * 5
        g = compute_gwar(ids, threshold=8)
        assert abs(g - 10 / 15) < 1e-9

    def test_gwar_all_singletons(self):
        # Alternating adapters; each segment has size 1
        # threshold=2 -> 0.0
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


# Fused LoRA kernel (CPU reference)

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
        # alpha=0 -> Y = X @ W^T (LoRA disabled)
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


# CASH dispatch (form_batch_wgkp)

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
        # W=32 tokens of adapter_0 -> condition 1 -> is_promoted=True
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
        # Exactly n*=8 tokens of adapter, in MWC -> condition 2 -> is_promoted=True
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
        # n* tokens but NOT in MWC -> no promotion (condition 2 requires in MWC)
        buf, _ = self._make_buffer_with_tokens({"a0": 8})
        result = buf.form_batch_wgkp(
            ranked_adapters=["a0"],
            tmax_k={"a0": 1.0},
            n_star=8,
            merged_adapter_ids=set(),  # empty -> not in MWC
        )
        # Should still dispatch (via condition 3 or 4 when T_max expires);
        # with fresh tokens (age ~ 0), it holds unless T_max/2 passed
        # Here tokens are new so condition 3 (age > T_max/2) is not met yet.
        # Result may be empty (held) or dispatched without promotion.
        for _, _, is_promoted in result:
            assert not is_promoted

    def test_cash_partial_dispatch_not_promoted(self):
        # ceil(n*/2)=4 tokens, age > T_max/2 -> condition 3 -> dispatched, is_promoted=False
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
        # 4 tokens (= ceil(8/2)), aged past T_max/2 -> condition 3 fires
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
        # Age past T_max -> condition 4 forces dispatch regardless of queue size
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


# WGKPDispatcher segmentation

from adapter_slots.kernel.wgkp_dispatcher import WGKPDispatcher


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
        # Mixed is_promoted within a segment -> ValueError
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


# AdaptivePromoThreshold

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
        # No valid profile -> fallback
        assert apt.current_threshold() == 32

    def test_apt_hardware_identifier(self, tmp_path):
        import json
        profile = {"hardware": "my_gpu", "warp_size": 32, "crossover_curve": {}}
        p = tmp_path / "p.json"
        p.write_text(json.dumps(profile))
        apt = AdaptivePromoThreshold(hw_profile_path=str(p))
        assert apt.hardware() == "my_gpu"


# Integration: scheduler env var parsing

class TestSchedulerWGKPEnvVars:
    """The scheduler's environment parsing, without vLLM."""

    def test_scheduler_wgkp_env_vars_parsed(self, monkeypatch):
        monkeypatch.setenv("AS_WGKP_THRESHOLD", "8")
        monkeypatch.setenv("AS_MWC_K_HOT", "3")
        monkeypatch.setenv("AS_TTFT_SLO_MS", "150")
        from adapter_slots.integrations.vllm_scheduler import _env_float, _env_int
        assert _env_int("AS_WGKP_THRESHOLD", 8) == 8
        assert _env_int("AS_MWC_K_HOT", 5) == 3
        assert _env_float("AS_TTFT_SLO_MS", 200.0) == 150.0
        monkeypatch.setenv("AS_MWC_K_HOT", "not-a-number")
        assert _env_int("AS_MWC_K_HOT", 5) == 5
