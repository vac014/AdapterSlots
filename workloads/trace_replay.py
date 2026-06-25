"""
workloads/trace_replay.py -- LMSYS / BurstGPT trace replay for bench_real_traces.py.

Loads a pre-processed trace file and constructs a Request list preserving the original
inter-arrival timing. Adapter IDs are assigned via Zipf(alpha=0.9) over K adapters
when not present in the trace.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from workloads.pattern_generator import Request


_DEFAULT_LMSYS = "data/lmsys/lmsys_trace.jsonl"
_DEFAULT_BURSTGPT = "data/burstgpt/burstgpt_trace.jsonl"


def _zipf_adapter(rng: random.Random, K: int, alpha: float = 0.9) -> int:
    weights = [(k + 1) ** (-alpha) for k in range(K)]
    total = sum(weights)
    cdf = []
    cumulative = 0.0
    for w in weights:
        cumulative += w / total
        cdf.append(cumulative)
    u = rng.random()
    lo, hi = 0, K - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < u:
            lo = mid + 1
        else:
            hi = mid
    return lo


def load_trace(
    path: str,
    K: int,
    fallback_prompts: List[str],
    max_tokens: int = 128,
    max_requests: Optional[int] = None,
    duration_s: Optional[float] = None,
    seed: int = 42,
) -> List[Request]:
    """Load a trace JSONL file and return a Request list.

    Trace JSONL format (one record per line):
      {"timestamp": float, "adapter_id": int (optional), "prompt": str (optional)}

    Records are sorted by timestamp. adapter_id is assigned via Zipf if absent.
    If duration_s is set, only requests within that time window are kept.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    rng = random.Random(seed)
    records = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records.sort(key=lambda r: float(r.get("timestamp", 0.0)))

    if duration_s is not None:
        t0 = float(records[0].get("timestamp", 0.0)) if records else 0.0
        records = [r for r in records if float(r.get("timestamp", 0.0)) - t0 <= duration_s]

    if max_requests is not None:
        records = records[:max_requests]

    requests = []
    prev_ts = 0.0
    for i, rec in enumerate(records):
        ts = float(rec.get("timestamp", 0.0))
        inter = max(0.0, ts - prev_ts)
        prev_ts = ts

        if "adapter_id" in rec:
            aid = int(rec["adapter_id"]) % K
        else:
            aid = _zipf_adapter(rng, K)

        prompt = rec.get("prompt", fallback_prompts[i % len(fallback_prompts)])

        requests.append(
            Request(
                req_id=f"trace_{i:06d}",
                adapter_id=aid,
                prompt=prompt,
                inter_arrival_s=inter,
                max_tokens=max_tokens,
            )
        )

    return requests


def preprocess_lmsys_raw(
    raw_path: str,
    output_path: str,
    max_tokens: int = 128,
) -> int:
    """Convert lmsys-chat-1m raw JSONL to trace JSONL with timestamps."""
    raw = Path(raw_path)
    if not raw.exists():
        raise FileNotFoundError(f"LMSYS raw file not found: {raw_path}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    fake_ts = 0.0
    with raw.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            conversations = rec.get("conversation", [])
            for turn in conversations:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user" and isinstance(content, str):
                    words = len(content.split())
                    if 4 <= words <= max_tokens:
                        # synthetic Poisson-like inter-arrival at ~5 req/s
                        import random
                        fake_ts += random.expovariate(5.0)
                        fout.write(
                            json.dumps({"timestamp": round(fake_ts, 4), "prompt": content})
                            + "\n"
                        )
                        n_written += 1
    return n_written


def preprocess_burstgpt_raw(
    raw_path: str,
    output_path: str,
    max_tokens: int = 128,
) -> int:
    """Convert BurstGPT raw JSONL to trace JSONL preserving original timestamps."""
    raw = Path(raw_path)
    if not raw.exists():
        raise FileNotFoundError(f"BurstGPT raw file not found: {raw_path}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with raw.open() as fin, out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompt = rec.get("prompt", rec.get("input", ""))
            ts = float(rec.get("timestamp", rec.get("time", 0.0)))
            if isinstance(prompt, str) and 4 <= len(prompt.split()) <= max_tokens:
                fout.write(json.dumps({"timestamp": ts, "prompt": prompt}) + "\n")
                n_written += 1
    return n_written
