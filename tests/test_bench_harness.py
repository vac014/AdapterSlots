"""
tests/test_bench_harness.py -- Unit tests for bench.py (sota_evaluation harness).

All tests run without GPU via --dry-run mode. No servers are started.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is importable

from backends.backend_adapterslots import AdapterSlotsBackend
from workloads.pattern_generator import ArrivalPatternGenerator, Request


# Helpers

def _fake_prompts(n: int) -> list:
    return [f"Prompt number {i}: the quick brown fox" for i in range(n)]


def _run_bench_dry(extra_args=None) -> dict:
    """Call run_benchmark with dry_run=True and return the result dict."""
    from benchmarks.ablations.bench import run_benchmark, _list_adapters
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name

    result = run_benchmark(
        backend_name="adapterslots",
        mode="C7",
        model="./models/llama-7b",
        adapter_dirs=["./adapters"],
        num_adapters=4,
        rank=32,
        target_modules="all_linear",
        request_rate=7.0,
        dataset="sharegpt",
        pattern="zipf",
        num_prompts=50,
        warmup=5,
        reps=3,
        seeds=[42, 43, 44],
        output=out_path,
        tp=1,
        tmax_ms=90,
        wgkp_threshold=8,
        port=8100,
        dry_run=True,
    )
    return result


# Tests

class TestResultSchema:
    def test_required_top_level_keys(self):
        result = _run_bench_dry()
        for key in ("config", "reps", "summary"):
            assert key in result, f"Missing top-level key: {key}"

    def test_config_has_backend(self):
        result = _run_bench_dry()
        assert "backend" in result["config"]
        assert result["config"]["backend"] == "adapterslots"

    def test_config_has_mode(self):
        result = _run_bench_dry()
        assert result["config"]["mode"] == "C7"

    def test_config_has_k(self):
        result = _run_bench_dry()
        assert "K" in result["config"]

    def test_summary_has_throughput(self):
        result = _run_bench_dry()
        assert "throughput_toks_mean" in result["summary"]
        assert result["summary"]["throughput_toks_mean"] > 0


class TestThreeRepStructure:
    def test_reps_is_list(self):
        result = _run_bench_dry()
        assert isinstance(result["reps"], list)

    def test_reps_has_three_entries(self):
        result = _run_bench_dry()
        assert len(result["reps"]) == 3

    def test_reps_have_seeds(self):
        result = _run_bench_dry()
        seeds = [r["seed"] for r in result["reps"]]
        assert sorted(seeds) == [42, 43, 44]

    def test_reps_have_throughput(self):
        result = _run_bench_dry()
        for rep in result["reps"]:
            assert "throughput_toks" in rep
            assert rep["throughput_toks"] > 0


class TestDryRunNoServer:
    def test_no_subprocess_started(self):
        """dry_run=True must not start any subprocess."""
        import subprocess
        original_popen = subprocess.Popen
        started = []

        class _Spy(subprocess.Popen):
            def __init__(self, *a, **kw):
                started.append(a)
                super().__init__(*a, **kw)

        subprocess.Popen = _Spy
        try:
            _run_bench_dry()
        finally:
            subprocess.Popen = original_popen

        assert len(started) == 0, "dry_run must not start any subprocess"


class TestConfigMapCompleteness:
    def test_all_configs_present(self):
        from backends.backend_adapterslots import AdapterSlotsBackend, CONFIG_MAP
        for c in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            assert c in CONFIG_MAP, f"Missing config: {c}"

    def test_c7_has_all_mechanisms(self):
        from backends.backend_adapterslots import CONFIG_MAP
        c7 = CONFIG_MAP["C7"]
        assert c7.get("AS_SCHEDULER") == "1"
        assert c7.get("AS_FUSED_KERNEL") == "1"

    def test_c0_disables_scheduler(self):
        from backends.backend_adapterslots import CONFIG_MAP
        c0 = CONFIG_MAP["C0"]
        assert c0.get("AS_SCHEDULER") == "0"


class TestRateLimiting:
    def test_inter_arrival_matches_rate(self):
        prompts = _fake_prompts(200)
        gen = ArrivalPatternGenerator(prompts, max_tokens=128)
        rate = 5.0
        reqs = gen.zipf(rate, 100, K=4, seed=42)

        total_ia = sum(r.inter_arrival_s for r in reqs if r.inter_arrival_s > 0)
        n = len([r for r in reqs if r.inter_arrival_s > 0])
        if n > 0:
            mean_ia = total_ia / n
            assert abs(mean_ia - 1.0 / rate) < 0.5, (
                f"Mean inter-arrival {mean_ia:.3f}s, expected ~{1/rate:.3f}s"
            )

    def test_uniform_rate(self):
        prompts = _fake_prompts(200)
        gen = ArrivalPatternGenerator(prompts, max_tokens=128)
        reqs = gen.uniform(10.0, 100, K=4, seed=42)
        assert len(reqs) == 100


class TestWarmupExclusion:
    def test_warmup_requests_excluded(self):
        """Warmup requests should not be counted in metrics."""
        from benchmarks.metrics_collector import MetricsCollector
        import time
        col = MetricsCollector()
        col.mark_run_start()

        # Simulate 5 warmup + 10 measured
        for i in range(5):
            col.mark_warmup(f"warmup_{i}")
            col.record_request_start(f"warmup_{i}", 0, time.perf_counter())
            col.record_first_token(f"warmup_{i}", time.perf_counter() + 0.01)
            col.record_completion(f"warmup_{i}", time.perf_counter() + 0.1, 10)

        for i in range(10):
            req_id = f"req_{i}"
            col.record_request_start(req_id, 0, time.perf_counter())
            col.record_first_token(req_id, time.perf_counter() + 0.01)
            col.record_completion(req_id, time.perf_counter() + 0.1, 10)

        col.mark_run_end()
        summary = col.compute()
        assert summary.n_completed <= 10, (
            f"Expected ≤10 non-warmup completions, got {summary.n_completed}"
        )


class TestResultJson:
    def test_result_is_valid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = f.name
        from benchmarks.ablations.bench import run_benchmark
        run_benchmark(
            backend_name="adapterslots", mode="C7",
            model="./models/llama-7b",
            adapter_dirs=["./adapters"],
            num_adapters=4, rank=32,
            target_modules="all_linear",
            request_rate=7.0, dataset="sharegpt", pattern="zipf",
            num_prompts=10, warmup=2, reps=1, seeds=[42],
            output=out_path, dry_run=True,
        )
        with open(out_path) as f:
            loaded = json.load(f)
        assert isinstance(loaded, dict)
        assert "config" in loaded
