"""
benchmarks/metrics_collector.py -- Per-request timestamp collection and metric aggregation.

Collects TTFT, TBT, TPOT, SLO attainment; computes percentiles and throughput.
Merges AdapterSlots alignment metrics (WAR, GWAR, promotion_fraction) from AS_METRICS_PATH JSONL.
"""

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RequestRecord:
    req_id: str
    adapter_id: int
    submit_time: float
    first_token_time: Optional[float] = None
    token_times: List[float] = field(default_factory=list)
    end_time: Optional[float] = None
    output_token_count: int = 0


@dataclass
class MetricsSummary:
    throughput_toks: float
    throughput_reqs: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    tbt_p50_ms: float
    tbt_p99_ms: float
    tpot_ms: float
    slo_attainment: float
    war: Optional[float] = None
    gwar8: Optional[float] = None
    promotion_fraction: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    gpu_mem_gb: Optional[float] = None
    n_completed: int = 0
    wall_time_s: float = 0.0


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


class MetricsCollector:
    """Collects per-request timestamps; computes benchmark metrics at end of run."""

    def __init__(self, ttft_slo_ms: float = 1000.0, tbt_slo_ms: float = 100.0):
        self._records: Dict[str, RequestRecord] = {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._ttft_slo = ttft_slo_ms
        self._tbt_slo = tbt_slo_ms
        self._warmup_req_ids: set = set()

    def mark_run_start(self) -> None:
        self._start_time = time.perf_counter()

    def mark_run_end(self) -> None:
        self._end_time = time.perf_counter()

    def mark_warmup(self, req_id: str) -> None:
        self._warmup_req_ids.add(req_id)

    def record_request_start(self, req_id: str, adapter_id: int, submit_time: float) -> None:
        self._records[req_id] = RequestRecord(
            req_id=req_id, adapter_id=adapter_id, submit_time=submit_time
        )

    def record_first_token(self, req_id: str, first_token_time: float) -> None:
        if req_id in self._records:
            self._records[req_id].first_token_time = first_token_time

    def record_token(self, req_id: str, token_time: float) -> None:
        if req_id in self._records:
            self._records[req_id].token_times.append(token_time)

    def record_completion(
        self, req_id: str, end_time: float, output_token_count: int
    ) -> None:
        if req_id in self._records:
            r = self._records[req_id]
            r.end_time = end_time
            r.output_token_count = output_token_count

    def compute(self) -> MetricsSummary:
        wall_time = (self._end_time or time.perf_counter()) - (
            self._start_time or 0.0
        )
        measured = [
            r
            for rid, r in self._records.items()
            if rid not in self._warmup_req_ids and r.end_time is not None
        ]

        if not measured:
            return MetricsSummary(
                throughput_toks=0.0,
                throughput_reqs=0.0,
                ttft_p50_ms=float("nan"),
                ttft_p99_ms=float("nan"),
                tbt_p50_ms=float("nan"),
                tbt_p99_ms=float("nan"),
                tpot_ms=float("nan"),
                slo_attainment=0.0,
            )

        total_output_tokens = sum(r.output_token_count for r in measured)
        throughput_toks = total_output_tokens / max(wall_time, 1e-9)
        throughput_reqs = len(measured) / max(wall_time, 1e-9)

        ttfts_ms = []
        for r in measured:
            if r.first_token_time is not None:
                ttfts_ms.append((r.first_token_time - r.submit_time) * 1000.0)

        all_tbts_ms = []
        tpots = []
        for r in measured:
            times = [r.first_token_time] + r.token_times if r.first_token_time else r.token_times
            if len(times) >= 2:
                tbts = [(times[i] - times[i - 1]) * 1000.0 for i in range(1, len(times))]
                all_tbts_ms.extend(tbts)
                if r.output_token_count > 1:
                    decode_time_ms = (r.end_time - (r.first_token_time or r.submit_time)) * 1000.0
                    tpots.append(decode_time_ms / max(r.output_token_count - 1, 1))

        slo_pass = 0
        for r in measured:
            ttft_ok = (
                r.first_token_time is not None
                and (r.first_token_time - r.submit_time) * 1000.0 < self._ttft_slo
            )
            tbt_ok = True
            if r.token_times:
                times = [r.first_token_time] + r.token_times if r.first_token_time else r.token_times
                if len(times) >= 2:
                    max_tbt = max(
                        (times[i] - times[i - 1]) * 1000.0 for i in range(1, len(times))
                    )
                    tbt_ok = max_tbt < self._tbt_slo
            if ttft_ok and tbt_ok:
                slo_pass += 1

        return MetricsSummary(
            throughput_toks=throughput_toks,
            throughput_reqs=throughput_reqs,
            ttft_p50_ms=_percentile(ttfts_ms, 50),
            ttft_p99_ms=_percentile(ttfts_ms, 99),
            tbt_p50_ms=_percentile(all_tbts_ms, 50),
            tbt_p99_ms=_percentile(all_tbts_ms, 99),
            tpot_ms=statistics.mean(tpots) if tpots else float("nan"),
            slo_attainment=slo_pass / len(measured) if measured else 0.0,
            n_completed=len(measured),
            wall_time_s=wall_time,
        )


def parse_as_metrics_jsonl(metrics_path: str) -> Dict[str, float]:
    """Parse AS_METRICS_PATH JSONL and return mean WAR, GWAR8, promotion_fraction."""
    p = Path(metrics_path)
    if not p.exists():
        return {}
    wars, gwar8s, promos = [], [], []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "war" in rec:
                    wars.append(float(rec["war"]))
                if "gwar8" in rec:
                    gwar8s.append(float(rec["gwar8"]))
                if "promotion_fraction" in rec:
                    promos.append(float(rec["promotion_fraction"]))
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    result = {}
    if wars:
        result["war"] = statistics.mean(wars)
    if gwar8s:
        result["gwar8"] = statistics.mean(gwar8s)
    if promos:
        result["promotion_fraction"] = statistics.mean(promos)
    return result
