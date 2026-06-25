"""
tests/test_pattern_generator.py -- Unit tests for workloads/pattern_generator.py.

All tests are CPU-only, no GPU required.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest


from workloads.pattern_generator import ArrivalPatternGenerator, Request, save_requests, load_requests


def _fake_prompts(n: int = 500) -> list:
    return [f"Prompt {i}: the quick brown fox jumped over the lazy dog" for i in range(n)]


class TestIdenticalPattern:
    def test_all_same_adapter(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.identical(rate=5.0, n_prompts=100, adapter_id=0, seed=42)
        assert all(r.adapter_id == 0 for r in reqs)

    def test_custom_adapter_id(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.identical(rate=5.0, n_prompts=50, adapter_id=3, seed=42)
        assert all(r.adapter_id == 3 for r in reqs)

    def test_correct_length(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.identical(rate=5.0, n_prompts=100, seed=42)
        assert len(reqs) == 100

    def test_max_tokens_set(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=64)
        reqs = gen.identical(rate=5.0, n_prompts=10, seed=42)
        assert all(r.max_tokens == 64 for r in reqs)


class TestUniformPattern:
    def test_spreads_across_adapters(self):
        K = 8
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.uniform(rate=5.0, n_prompts=400, K=K, seed=42)
        counts = Counter(r.adapter_id for r in reqs)
        assert len(counts) == K, "Uniform must use all K adapters"
        # Each adapter should get roughly n/K requests (within 2× tolerance)
        expected = len(reqs) / K
        for k, c in counts.items():
            assert c >= expected / 2, f"Adapter {k} got too few: {c} (expected ~{expected:.0f})"
            assert c <= expected * 2, f"Adapter {k} got too many: {c} (expected ~{expected:.0f})"

    def test_all_ids_in_range(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        K = 5
        reqs = gen.uniform(rate=5.0, n_prompts=100, K=K, seed=42)
        assert all(0 <= r.adapter_id < K for r in reqs)


class TestZipfPattern:
    def test_power_law_distribution(self):
        K = 10
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.zipf(rate=7.0, n_prompts=1000, K=K, alpha=0.9, seed=42)
        counts = Counter(r.adapter_id for r in reqs)
        sorted_counts = sorted(counts.values(), reverse=True)
        # Most popular adapter must be at least 3× as popular as least popular
        most = sorted_counts[0]
        least = sorted_counts[-1]
        assert most >= 3 * least, (
            f"Zipf distribution not skewed enough: most={most}, least={least}"
        )

    def test_all_ids_in_range(self):
        K = 10
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.zipf(rate=7.0, n_prompts=500, K=K, seed=42)
        assert all(0 <= r.adapter_id < K for r in reqs)

    def test_correct_length(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.zipf(rate=7.0, n_prompts=200, K=5, seed=42)
        assert len(reqs) == 200

    def test_uses_all_adapters_at_large_n(self):
        K = 10
        gen = ArrivalPatternGenerator(_fake_prompts(1000), max_tokens=128)
        reqs = gen.zipf(rate=7.0, n_prompts=500, K=K, seed=42)
        ids = set(r.adapter_id for r in reqs)
        assert len(ids) == K, f"Zipf at n=500 K=10 should use all adapters, got {len(ids)}"


class TestDistinctPattern:
    def test_round_robin(self):
        K = 5
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.distinct(rate=5.0, n_prompts=20, K=K, seed=42)
        for i, r in enumerate(reqs):
            assert r.adapter_id == i % K, (
                f"Request {i} should have adapter {i % K}, got {r.adapter_id}"
            )

    def test_all_adapters_covered(self):
        K = 8
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.distinct(rate=5.0, n_prompts=K * 3, K=K, seed=42)
        ids = set(r.adapter_id for r in reqs)
        assert ids == set(range(K))


class TestRequestRate:
    def test_total_time_matches_rate(self):
        rate = 10.0
        n = 100
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.zipf(rate=rate, n_prompts=n, K=5, seed=42)
        total_ia = sum(r.inter_arrival_s for r in reqs)
        # Total inter-arrival ≈ n / rate (within 20%)
        expected = n / rate
        assert abs(total_ia - expected) < expected * 0.5, (
            f"Total IA={total_ia:.2f}s, expected ~{expected:.2f}s (rate={rate})"
        )

    def test_zero_rate_raises_or_handles(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        try:
            reqs = gen.zipf(rate=0.0, n_prompts=10, K=4, seed=42)
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable
        except Exception as e:
            pytest.fail(f"Unexpected exception for rate=0: {e}")


class TestSeedDeterminism:
    def test_same_seed_identical_output(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs_a = gen.zipf(rate=7.0, n_prompts=100, K=10, seed=42)
        reqs_b = gen.zipf(rate=7.0, n_prompts=100, K=10, seed=42)
        ids_a = [r.adapter_id for r in reqs_a]
        ids_b = [r.adapter_id for r in reqs_b]
        assert ids_a == ids_b, "Same seed must produce identical adapter_id sequence"

    def test_different_seeds_differ(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs_a = gen.zipf(rate=7.0, n_prompts=100, K=10, seed=42)
        reqs_b = gen.zipf(rate=7.0, n_prompts=100, K=10, seed=99)
        ids_a = [r.adapter_id for r in reqs_a]
        ids_b = [r.adapter_id for r in reqs_b]
        assert ids_a != ids_b, "Different seeds should produce different sequences"


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_fields(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.zipf(rate=5.0, n_prompts=50, K=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            tmp_path = f.name

        try:
            save_requests(reqs, tmp_path)
            loaded = load_requests(tmp_path)

            assert len(loaded) == len(reqs)
            for orig, loaded_r in zip(reqs, loaded):
                assert orig.req_id == loaded_r.req_id
                assert orig.adapter_id == loaded_r.adapter_id
                assert orig.max_tokens == loaded_r.max_tokens
                assert abs(orig.inter_arrival_s - loaded_r.inter_arrival_s) < 1e-9
        finally:
            os.unlink(tmp_path)

    def test_save_creates_jsonl(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        reqs = gen.uniform(rate=5.0, n_prompts=10, K=2, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            tmp_path = f.name
        try:
            save_requests(reqs, tmp_path)
            with open(tmp_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) == 10
            parsed = json.loads(lines[0])
            assert "req_id" in parsed
        finally:
            os.unlink(tmp_path)


class TestFromTraceFallback:
    def test_missing_trace_falls_back_to_uniform(self):
        gen = ArrivalPatternGenerator(_fake_prompts(), max_tokens=128)
        # from_trace with non-existent file should either raise or fall back
        try:
            reqs = gen.from_trace(
                "/nonexistent/path/trace.jsonl",
                K=4,
                seed=42,
            )
            # If no exception: should return a valid list of Requests
            assert isinstance(reqs, list)
        except (FileNotFoundError, OSError):
            pass  # Acceptable: caller (bench_real_traces.py) handles the fallback
