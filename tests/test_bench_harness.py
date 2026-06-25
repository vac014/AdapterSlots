"""
tests/test_bench_harness.py -- Unit tests for the serving harness pieces that
run without a GPU: the lattice mode table, the arrival generator and warmup
handling in the metrics collector.
"""

from __future__ import annotations

import time

from benchmarks.metrics_collector import MetricsCollector
from workloads.pattern_generator import ArrivalPatternGenerator


def _prompts(n: int) -> list:
    return [f"Prompt number {i}: the quick brown fox" for i in range(n)]


class TestConfigMapCompleteness:
    def test_all_configs_present(self):
        from backends.backend_adapterslots import CONFIG_MAP
        for c in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            assert c in CONFIG_MAP, f"Missing config: {c}"

    def test_c7_has_all_mechanisms(self):
        from backends.backend_adapterslots import CONFIG_MAP
        c7 = CONFIG_MAP["C7"]
        assert c7.get("AS_SCHEDULER") == "1"
        assert c7.get("AS_MODE") == "wgkp"
        assert c7.get("AS_SEGGEMM") == "3"
        assert CONFIG_MAP["C6"].get("AS_SEGGEMM") == "0"

    def test_c0_disables_scheduler(self):
        from backends.backend_adapterslots import CONFIG_MAP
        assert CONFIG_MAP["C0"].get("AS_SCHEDULER") == "0"


class TestRateLimiting:
    def test_inter_arrival_matches_rate(self):
        gen = ArrivalPatternGenerator(_prompts(200), max_tokens=128)
        rate = 5.0
        reqs = gen.zipf(rate, 100, K=4, seed=42)
        gaps = [r.inter_arrival_s for r in reqs if r.inter_arrival_s > 0]
        mean_ia = sum(gaps) / len(gaps)
        assert abs(mean_ia - 1.0 / rate) < 0.5, (
            f"Mean inter-arrival {mean_ia:.3f}s, expected ~{1 / rate:.3f}s"
        )

    def test_uniform_rate(self):
        gen = ArrivalPatternGenerator(_prompts(200), max_tokens=128)
        assert len(gen.uniform(10.0, 100, K=4, seed=42)) == 100


class TestWarmupExclusion:
    def test_warmup_requests_excluded(self):
        """Warmup requests are not counted in the metrics."""
        col = MetricsCollector()
        col.mark_run_start()
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
            f"Expected at most 10 non-warmup completions, got {summary.n_completed}"
        )
