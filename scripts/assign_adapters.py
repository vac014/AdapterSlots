"""
assign_adapters.py -- Adapter Assignment for BurstGPT Traces (workload_characterization, §5.3)

Assigns synthetic adapter IDs to BurstGPT requests based on request type frequency.
BurstGPT only has two models (ChatGPT / GPT-4); we need to expand to K=4, 16, 32.

Strategy:
  1. Sort by model (existing adapter_id from preprocess_burstgpt.py) and arrival_time_ms
  2. Within each model, cluster requests into sub-adapters by hash of request_id
  3. Result: Zipf-like popularity (popular model → more sub-adapters with higher traffic)

This produces K adapter IDs such that:
  - The top adapter (k=0) has the highest traffic (from ChatGPT 0)
  - k=K-1 is a catch-all for rare request types

Outputs one JSONL file per K value with an updated adapter_id field.

Usage:
  # K=4 (Single A6000)
  python scripts/assign_adapters.py \\
      --input data/burstgpt/burstgpt.jsonl \\
      --K 4 \\
      --output data/burstgpt/burstgpt_k4.jsonl

  # K=16 (Two A6000 PCIe)
  python scripts/assign_adapters.py \\
      --input data/burstgpt/burstgpt.jsonl \\
      --K 16 \\
      --output data/burstgpt/burstgpt_k16.jsonl

  # K=32 (Two H100 NVLink)
  python scripts/assign_adapters.py \\
      --input data/burstgpt/burstgpt.jsonl \\
      --K 32 \\
      --output data/burstgpt/burstgpt_k32.jsonl

  # Generate all K variants at once:
  python scripts/assign_adapters.py \\
      --input data/burstgpt/burstgpt.jsonl \\
      --K-values 4 16 32 \\
      --output-dir data/burstgpt/

  # Extract 30-minute segment for replay harness:
  python scripts/assign_adapters.py \\
      --input data/burstgpt/burstgpt.jsonl \\
      --K 4 \\
      --segment-minutes 30 \\
      --output data/burstgpt/burstgpt_k4_30min.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Adapter assignment logic

def assign_adapters_from_burstgpt(
    trace: List[dict],
    K: int = 4,
    zipf_alpha: float = 0.9,
    seed: int = 42,
) -> List[dict]:
    """
    Assign K synthetic adapter IDs to BurstGPT requests.

    The original trace has adapter_id ∈ {0, 1} (ChatGPT, GPT-4).
    We expand to K adapters by sub-partitioning each model class by
    request_id hash, producing a Zipf-like popularity distribution.

    BurstGPT ChatGPT traffic >> GPT-4 traffic, so:
      - adapter 0..N_chatgpt-1  get Zipf-weighted traffic from ChatGPT bucket
      - adapter N_chatgpt..K-1  get remaining traffic from GPT-4 bucket
    """
    # Count traffic per original model
    model_counts: Dict[int, int] = collections.Counter(r["adapter_id"] for r in trace)
    sorted_models = sorted(model_counts, key=model_counts.get, reverse=True)

    # Allocate K sub-adapters proportional to model traffic
    if K == 1:
        return [{**r, "adapter_id": 0} for r in trace]

    if K <= len(sorted_models):
        # Just map each model to one adapter (K ≤ 2 for BurstGPT)
        model_to_adapter = {m: i for i, m in enumerate(sorted_models[:K])}
        for m in sorted_models[K:]:
            model_to_adapter[m] = K - 1
        return [{**r, "adapter_id": model_to_adapter[r["adapter_id"]]} for r in trace]

    # K > 2: sub-partition models into K adapters with Zipf weights
    # Give more adapters to the more popular model
    total = sum(model_counts.values())
    n_primary = max(1, round(K * model_counts[sorted_models[0]] / total))
    n_secondary = K - n_primary
    n_secondary = max(1, n_secondary)
    n_primary = K - n_secondary

    # Zipf weights for sub-adapter assignments
    def zipf_weights(n: int) -> np.ndarray:
        ranks = np.arange(1, n + 1, dtype=float)
        w = ranks ** (-zipf_alpha)
        return w / w.sum()

    primary_weights = zipf_weights(n_primary)
    secondary_weights = zipf_weights(n_secondary)

    # Pre-compute CDFs for deterministic Zipf mapping via inverse-CDF
    primary_cdf = np.cumsum(primary_weights)
    secondary_cdf = np.cumsum(secondary_weights)

    primary_model = sorted_models[0]

    result = []
    for r in trace:
        orig_adapter = r["adapter_id"]
        rid = r.get("request_id", 0)

        if orig_adapter == primary_model:
            # Map request_id hash uniformly into [0,1) then invert Zipf CDF
            bucket = int(rid) % 100000
            u = (bucket + 0.5) / 100000.0
            sub_idx = int(np.searchsorted(primary_cdf, u))
            adapter_id = min(sub_idx, n_primary - 1)
        else:
            # Secondary model(s) → adapters n_primary..K-1
            bucket = int(rid) % 100000
            u = (bucket + 0.5) / 100000.0
            sub_idx = int(np.searchsorted(secondary_cdf, u))
            adapter_id = n_primary + min(sub_idx, n_secondary - 1)

        result.append({**r, "adapter_id": adapter_id})

    return result


def extract_segment(
    trace: List[dict],
    segment_minutes: float,
    offset_minutes: float = 0.0,
) -> List[dict]:
    """
    Extract a contiguous segment from the trace.
    segment_minutes: duration of the segment to extract.
    offset_minutes: start offset from the beginning of the trace.
    """
    if not trace:
        return []

    t_start_ms = trace[0]["arrival_time_ms"] + offset_minutes * 60_000.0
    t_end_ms = t_start_ms + segment_minutes * 60_000.0

    segment = [r for r in trace if t_start_ms <= r["arrival_time_ms"] < t_end_ms]

    # Re-zero arrival times relative to segment start
    if segment:
        t0 = segment[0]["arrival_time_ms"]
        segment = [{**r, "arrival_time_ms": round(r["arrival_time_ms"] - t0, 3)}
                   for r in segment]

    return segment


# I/O helpers

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# Main

def parse_args():
    p = argparse.ArgumentParser(
        description="Assign K synthetic adapter IDs to BurstGPT trace"
    )
    p.add_argument("--input", required=True,
                   help="Input JSONL trace from preprocess_burstgpt.py")
    p.add_argument("--K", type=int, default=None,
                   help="Number of adapters (mutually exclusive with --K-values)")
    p.add_argument("--K-values", nargs="+", type=int, default=None,
                   help="Multiple K values to generate (e.g. 4 16 32)")
    p.add_argument("--output", default=None,
                   help="Output JSONL path (use with --K)")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (use with --K-values; filenames auto-generated)")
    p.add_argument("--zipf-alpha", type=float, default=0.9,
                   help="Zipf alpha for sub-adapter popularity distribution")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--segment-minutes", type=float, default=None,
                   help="Extract a contiguous segment of this duration (minutes)")
    p.add_argument("--offset-minutes", type=float, default=0.0,
                   help="Start offset for segment extraction (minutes from trace start)")
    p.add_argument("--no-segment-suffix", action="store_true",
                   help="Do not append segment info to filename")
    return p.parse_args()


def process_one(
    trace: List[dict],
    K: int,
    zipf_alpha: float,
    seed: int,
    segment_minutes: Optional[float],
    offset_minutes: float,
    output_path: str,
):
    print(f"  Assigning K={K} adapters ...", flush=True)
    assigned = assign_adapters_from_burstgpt(trace, K=K, zipf_alpha=zipf_alpha, seed=seed)

    if segment_minutes is not None:
        print(f"  Extracting {segment_minutes:.0f}-minute segment "
              f"(offset={offset_minutes:.0f} min) ...", flush=True)
        assigned = extract_segment(assigned, segment_minutes, offset_minutes)

    # Summary
    adapter_counts = collections.Counter(r["adapter_id"] for r in assigned)
    total = sum(adapter_counts.values())
    print(f"  {total:,} records, K={K} adapters:", flush=True)
    for k in sorted(adapter_counts):
        cnt = adapter_counts[k]
        print(f"    adapter {k:2d}: {cnt:6,} ({cnt/total:.1%})", flush=True)

    save_jsonl(assigned, output_path)
    print(f"  Written → {output_path}", flush=True)


def main():
    args = parse_args()

    if args.K is None and args.K_values is None:
        print("[ERROR] Specify --K or --K-values")
        sys.exit(1)

    print(f"Loading: {args.input}", flush=True)
    trace = load_jsonl(args.input)
    if not trace:
        print("[ERROR] No records loaded.")
        sys.exit(1)

    span_s = (trace[-1]["arrival_time_ms"] - trace[0]["arrival_time_ms"]) / 1000.0
    print(f"  Loaded {len(trace):,} records, "
          f"span={span_s:.0f}s ({span_s/3600:.1f}h), "
          f"mean_rate={len(trace)/max(span_s,1):.2f} req/s", flush=True)

    # Determine K values and output paths
    if args.K is not None:
        k_list = [args.K]
        if args.output:
            out_paths = [args.output]
        else:
            stem = Path(args.input).stem
            seg_str = (f"_{int(args.segment_minutes)}min"
                       if args.segment_minutes and not args.no_segment_suffix else "")
            out_paths = [str(Path(args.input).parent /
                            f"{stem}_k{args.K}{seg_str}.jsonl")]
    else:
        k_list = args.K_values
        out_dir = args.output_dir or str(Path(args.input).parent)
        stem = Path(args.input).stem
        seg_str = (f"_{int(args.segment_minutes)}min"
                   if args.segment_minutes and not args.no_segment_suffix else "")
        out_paths = [
            os.path.join(out_dir, f"{stem}_k{K}{seg_str}.jsonl")
            for K in k_list
        ]

    for K, out_path in zip(k_list, out_paths):
        process_one(
            trace=trace,
            K=K,
            zipf_alpha=args.zipf_alpha,
            seed=args.seed,
            segment_minutes=args.segment_minutes,
            offset_minutes=args.offset_minutes,
            output_path=out_path,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
