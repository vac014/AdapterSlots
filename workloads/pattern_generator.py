"""
workloads/pattern_generator.py -- Arrival pattern generator for AdapterSlots benchmark harness.

Generates Poisson-arrival workload request lists with four configurable adapter
distribution patterns: Identical, Uniform, Zipf, Distinct.

All patterns produce deterministic output keyed by seed for cross-backend fairness
(identical prompts and inter-arrival times sent to each backend in a comparison).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Request:
    req_id: str
    adapter_id: int
    prompt: str
    inter_arrival_s: float  # time before this request is sent
    max_tokens: int = 128


class ArrivalPatternGenerator:
    """Generates Poisson-arrival workloads with configurable adapter distribution."""

    def __init__(self, prompts: List[str], max_tokens: int = 128):
        self._prompts = prompts
        self._max_tokens = max_tokens

    def _poisson_arrivals(self, rate: float, n: int, rng: random.Random) -> List[float]:
        """Return list of n Poisson inter-arrival times at given rate (req/s)."""
        if rate <= 0:
            return [0.0] * n
        return [rng.expovariate(rate) for _ in range(n)]

    def _make_requests(
        self,
        adapter_ids: List[int],
        inter_arrivals: List[float],
        rng: random.Random,
        prompt_list: Optional[List[str]] = None,
    ) -> List[Request]:
        prompts = prompt_list or self._prompts
        n = len(adapter_ids)
        prompt_indices = rng.choices(range(len(prompts)), k=n)
        return [
            Request(
                req_id=f"req_{i:05d}",
                adapter_id=adapter_ids[i],
                prompt=prompts[prompt_indices[i]],
                inter_arrival_s=inter_arrivals[i],
                max_tokens=self._max_tokens,
            )
            for i in range(n)
        ]

    def identical(
        self, rate: float, n_prompts: int, adapter_id: int = 0, seed: int = 42
    ) -> List[Request]:
        """All requests go to a single adapter -- WAR=1.0, maximum promotion."""
        rng = random.Random(seed)
        arrivals = self._poisson_arrivals(rate, n_prompts, rng)
        adapter_ids = [adapter_id] * n_prompts
        return self._make_requests(adapter_ids, arrivals, rng)

    def uniform(
        self, rate: float, n_prompts: int, K: int, seed: int = 42
    ) -> List[Request]:
        """Round-robin uniform assignment across K adapters."""
        rng = random.Random(seed)
        arrivals = self._poisson_arrivals(rate, n_prompts, rng)
        adapter_ids = [i % K for i in range(n_prompts)]
        rng2 = random.Random(seed + 1)
        rng2.shuffle(adapter_ids)
        return self._make_requests(adapter_ids, arrivals, rng)

    def zipf(
        self,
        rate: float,
        n_prompts: int,
        K: int,
        alpha: float = 0.9,
        seed: int = 42,
    ) -> List[Request]:
        """Zipf-distributed adapter selection: adapter k ∝ (k+1)^(-alpha)."""
        rng = random.Random(seed)
        arrivals = self._poisson_arrivals(rate, n_prompts, rng)
        # Build CDF over K adapters
        weights = [(k + 1) ** (-alpha) for k in range(K)]
        total = sum(weights)
        cdf = []
        cumulative = 0.0
        for w in weights:
            cumulative += w / total
            cdf.append(cumulative)

        adapter_ids = []
        for _ in range(n_prompts):
            u = rng.random()
            # Binary search into CDF
            lo, hi = 0, K - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cdf[mid] < u:
                    lo = mid + 1
                else:
                    hi = mid
            adapter_ids.append(lo)
        return self._make_requests(adapter_ids, arrivals, rng)

    def distinct(
        self, rate: float, n_prompts: int, K: int, seed: int = 42
    ) -> List[Request]:
        """Cycling deterministic assignment -- request i → adapter i % K. Maximum diversity."""
        rng = random.Random(seed)
        arrivals = self._poisson_arrivals(rate, n_prompts, rng)
        adapter_ids = [i % K for i in range(n_prompts)]
        return self._make_requests(adapter_ids, arrivals, rng)

    def from_trace(
        self, trace_file: str, K: int, seed: int = 42
    ) -> List[Request]:
        """Replay recorded inter-arrival timestamps from LMSYS or BurstGPT trace file.

        Trace file format (JSONL): {"timestamp": float, "adapter_id": int, "prompt": str}
        If adapter_id absent, assigns adapter_id via zipf(alpha=0.9).
        """
        rng = random.Random(seed)
        records = []
        with open(trace_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        records.sort(key=lambda r: r.get("timestamp", 0.0))

        requests = []
        prev_ts = 0.0
        for i, rec in enumerate(records):
            ts = float(rec.get("timestamp", 0.0))
            inter = max(0.0, ts - prev_ts)
            prev_ts = ts
            aid = int(rec.get("adapter_id", i % K))
            prompt = rec.get("prompt", self._prompts[i % len(self._prompts)])
            requests.append(
                Request(
                    req_id=f"trace_{i:05d}",
                    adapter_id=aid % K,
                    prompt=prompt,
                    inter_arrival_s=inter,
                    max_tokens=self._max_tokens,
                )
            )
        return requests


def save_requests(requests: List[Request], path: str) -> None:
    """Serialise request list to JSONL for deterministic replay."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in requests:
            f.write(
                json.dumps(
                    {
                        "req_id": r.req_id,
                        "adapter_id": r.adapter_id,
                        "prompt": r.prompt,
                        "inter_arrival_s": r.inter_arrival_s,
                        "max_tokens": r.max_tokens,
                    }
                )
                + "\n"
            )


def load_requests(path: str) -> List[Request]:
    """Deserialise request list from JSONL."""
    requests = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            requests.append(
                Request(
                    req_id=d["req_id"],
                    adapter_id=d["adapter_id"],
                    prompt=d["prompt"],
                    inter_arrival_s=d["inter_arrival_s"],
                    max_tokens=d.get("max_tokens", 128),
                )
            )
    return requests
