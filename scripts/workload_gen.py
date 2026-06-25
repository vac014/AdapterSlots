"""
workload_gen.py -- Synthetic request-stream generator with 6 traffic patterns.

Usage:
    python scripts/workload_gen.py \
        --n-requests 5000 \
        --arrival-rate 7 \
        --pattern zipf \
        --zipf-alpha 0.9 \
        --K 4 \
        --seed 42 \
        --output workloads/zipf_k4_lam7.jsonl

Output format (one JSON object per line):
    {"request_id": 0, "adapter_id": 2, "arrival_time_ms": 142.3,
     "prompt_len": 128, "output_len": 64}

Supported patterns:
    identical    All requests to adapter 0
    uniform      Equal probability across K adapters
    zipf         Zipf(α) distribution over K adapters
    distinct     Round-robin across K adapters
    adversarial  Strictly alternating: ABABAB...
    correlated   Bursty geometric run-lengths of the same adapter
"""

import argparse
import json
import math
import os
import random

import numpy as np


PATTERNS = ["identical", "uniform", "zipf", "distinct", "adversarial", "correlated"]


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic LoRA request traces")
    p.add_argument("--n-requests", type=int, default=5000,
                   help="Total number of requests to generate")
    p.add_argument("--arrival-rate", type=float, default=7.0,
                   help="Poisson arrival rate λ (req/s). Use 'offline' flag for batch mode.")
    p.add_argument("--offline", action="store_true",
                   help="If set, all arrivals are at t=0 (batch mode, no inter-arrival times)")
    p.add_argument("--pattern", type=str, choices=PATTERNS, default="zipf",
                   help="Adapter assignment pattern")
    p.add_argument("--zipf-alpha", type=float, default=0.9,
                   help="Zipf exponent α (used only with --pattern zipf)")
    p.add_argument("--K", type=int, default=4,
                   help="Number of adapters")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--prompt-len", type=int, default=128,
                   help="Fixed prompt length in tokens")
    p.add_argument("--output-len", type=int, default=64,
                   help="Fixed output length in tokens")
    p.add_argument("--prompt-len-std", type=float, default=0.0,
                   help="Std dev of prompt length (0 = fixed). Gaussian noise added.")
    p.add_argument("--output-dir", type=str, default="./workloads",
                   help="Output directory (created if missing)")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSONL path. Defaults to --output-dir/<pattern>_k<K>_lam<rate>.jsonl")
    p.add_argument("--corr-p", type=float, default=0.1,
                   help="Geometric(p) run-length parameter for correlated pattern")
    return p.parse_args()


# Adapter ID generators

def gen_identical(n: int, K: int, rng: np.random.Generator) -> list:
    return [0] * n


def gen_uniform(n: int, K: int, rng: np.random.Generator) -> list:
    return rng.integers(0, K, size=n).tolist()


def gen_zipf(n: int, K: int, alpha: float, rng: np.random.Generator) -> list:
    """Zipf(α) over K adapters: P(k) ∝ 1/(k+1)^α, normalized."""
    ranks = np.arange(1, K + 1, dtype=float)
    weights = 1.0 / (ranks ** alpha)
    weights /= weights.sum()
    return rng.choice(K, size=n, p=weights).tolist()


def gen_distinct(n: int, K: int, rng: np.random.Generator) -> list:
    return [i % K for i in range(n)]


def gen_adversarial(n: int, K: int, rng: np.random.Generator) -> list:
    """Strictly alternating: 0,1,0,1,... (worst case for K=2)."""
    return [i % K for i in range(n)]


def gen_correlated(n: int, K: int, p: float, rng: np.random.Generator) -> list:
    """
    Bursty traffic: geometric(p) run-lengths of the same adapter.
    Each run picks an adapter uniformly; run length ~ Geom(p) (mean = 1/p).
    """
    ids = []
    current_adapter = int(rng.integers(0, K))
    while len(ids) < n:
        run_len = int(rng.geometric(p))
        ids.extend([current_adapter] * min(run_len, n - len(ids)))
        current_adapter = int(rng.integers(0, K))
    return ids[:n]


def generate_adapter_ids(pattern: str, n: int, K: int, alpha: float,
                          corr_p: float, rng: np.random.Generator) -> list:
    if pattern == "identical":
        return gen_identical(n, K, rng)
    elif pattern == "uniform":
        return gen_uniform(n, K, rng)
    elif pattern == "zipf":
        return gen_zipf(n, K, alpha, rng)
    elif pattern == "distinct":
        return gen_distinct(n, K, rng)
    elif pattern == "adversarial":
        return gen_adversarial(n, K, rng)
    elif pattern == "correlated":
        return gen_correlated(n, K, corr_p, rng)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def generate_arrival_times_poisson(n: int, rate: float, rng: np.random.Generator) -> list:
    """Generate inter-arrival times ~ Exponential(rate) and cumsum."""
    iat = rng.exponential(scale=1000.0 / rate, size=n)  # ms
    times = np.cumsum(iat).tolist()
    times = [0.0] + times[:-1]  # first request at t=0
    return times


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Generate adapter IDs
    adapter_ids = generate_adapter_ids(
        pattern=args.pattern,
        n=args.n_requests,
        K=args.K,
        alpha=args.zipf_alpha,
        corr_p=args.corr_p,
        rng=rng,
    )

    # Generate arrival times
    if args.offline:
        arrival_times = [0.0] * args.n_requests
    else:
        arrival_times = generate_arrival_times_poisson(
            args.n_requests, args.arrival_rate, rng
        )

    # Generate prompt / output lengths
    if args.prompt_len_std > 0:
        prompt_lens = np.maximum(1, rng.normal(
            args.prompt_len, args.prompt_len_std, size=args.n_requests
        ).astype(int)).tolist()
    else:
        prompt_lens = [args.prompt_len] * args.n_requests
    output_lens = [args.output_len] * args.n_requests

    # Build output path
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output is None:
        fname = f"{args.pattern}_k{args.K}_lam{args.arrival_rate:.1f}_n{args.n_requests}.jsonl"
        out_path = os.path.join(args.output_dir, fname)
    else:
        out_path = args.output

    # Write JSONL
    with open(out_path, "w") as f:
        for i in range(args.n_requests):
            rec = {
                "request_id": i,
                "adapter_id": int(adapter_ids[i]),
                "arrival_time_ms": round(float(arrival_times[i]), 3),
                "prompt_len": int(prompt_lens[i]),
                "output_len": int(output_lens[i]),
            }
            f.write(json.dumps(rec) + "\n")

    # Print summary statistics
    from collections import Counter
    counts = Counter(adapter_ids)
    print(f"Wrote {args.n_requests} requests to {out_path}")
    print(f"Pattern: {args.pattern}, K={args.K}, seed={args.seed}")
    print(f"Adapter distribution:")
    for k in sorted(counts.keys()):
        frac = counts[k] / args.n_requests
        print(f"  adapter {k}: {counts[k]:5d} ({frac:.3f})")
    if not args.offline:
        total_time_s = arrival_times[-1] / 1000.0
        print(f"Arrival span: {total_time_s:.1f} s  (mean rate: "
              f"{args.n_requests / total_time_s:.2f} req/s)")


if __name__ == "__main__":
    main()
