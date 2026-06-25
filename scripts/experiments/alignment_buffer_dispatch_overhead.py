"""
alignment_buffer_dispatch_overhead.py -- Measure AlignmentBuffer.form_batch() dispatch overhead.

Implements experiment §5.2 (Single A6000) and §5.5c (Two A6000 PCIe / Two H100 NVLink)
from alignment_buffer.md.

Measures CPU overhead of form_batch() across K values {2, 4, 8, 16, 32, 50}.
Pass condition: < 0.5 ms for K <= 50.

This script requires no GPU -- it measures pure Python dispatch overhead.

Usage:
    # Single A6000 (§5.2):
    python scripts/experiments/alignment_buffer_dispatch_overhead.py \
        --output results/alignment_buffer/a6000_single/dispatch_overhead.csv

    # Two A6000 PCIe K=16 verification (§5.5c):
    python scripts/experiments/alignment_buffer_dispatch_overhead.py \
        --K-values 4 8 16 \
        --output results/alignment_buffer/two_a6000_pcie/dispatch_overhead_k16.csv \
        --label "Two A6000 PCIe (TP=2)"

    # Two H100 NVLink K=32 (§5.6c):
    python scripts/experiments/alignment_buffer_dispatch_overhead.py \
        --K-values 4 8 16 32 \
        --output results/alignment_buffer/two_h100_nvlink/dispatch_overhead_k32.csv \
        --label "Two H100 NVLink (K=32)"
"""

import argparse
import csv
import os
import sys
import time
from typing import List

from adapter_slots.buffer import AlignmentBuffer


PASS_THRESHOLD_MS = 0.5  # §5.2 exit condition


def measure_dispatch_overhead(
    K: int,
    warp_size: int = 32,
    n_reps: int = 10_000,
    warmup: int = 500,
    tokens_per_adapter: int = 64,
    tmax_ms: float = 5.0,
) -> dict:
    """Measure form_batch() wall-clock time for a given K.

    Args:
        K:                   Number of concurrent adapters.
        warp_size:           GPU warp width (32 default).
        n_reps:              Number of timing repetitions.
        warmup:              Warm-up iterations before measurement.
        tokens_per_adapter:  Tokens pre-loaded per adapter (all full warps).
        tmax_ms:             T_max for the buffer (does not fire in this test
                             since all queues are full).

    Returns:
        Dictionary with overhead statistics.
    """
    adapters = [f"adapter_{k}" for k in range(K)]
    elapsed_samples = []

    for rep in range(warmup + n_reps):
        # Create a fresh buffer and fill all queues to exactly 2×W tokens
        # (one full warp, guaranteed to fire dispatch condition A).
        buf = AlignmentBuffer(adapters, warp_size=warp_size, tmax_ms=tmax_ms)
        for k, adapter in enumerate(adapters):
            for i in range(tokens_per_adapter):
                buf.enqueue(adapter, seq_id=k * 1000 + i)

        # Time form_batch() only
        t0 = time.perf_counter()
        batch = buf.form_batch(max_tokens=K * warp_size)
        t1 = time.perf_counter()

        if rep >= warmup:
            elapsed_samples.append((t1 - t0) * 1000.0)  # ms

    import statistics
    sorted_samples = sorted(elapsed_samples)
    n = len(sorted_samples)
    mean_ms = statistics.mean(elapsed_samples)
    p50_ms = statistics.median(elapsed_samples)
    p99_ms = sorted_samples[int(0.99 * n)] if n > 0 else 0.0
    p999_ms = sorted_samples[int(0.999 * n)] if n > 0 else 0.0
    max_ms = max(elapsed_samples)

    return {
        "K": K,
        "warp_size": warp_size,
        "n_reps": n_reps,
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p99_ms": p99_ms,
        "p999_ms": p999_ms,
        "max_ms": max_ms,
        "pass": mean_ms < PASS_THRESHOLD_MS,
    }


def run_sweep(
    K_values: List[int],
    warp_size: int = 32,
    n_reps: int = 10_000,
    warmup: int = 500,
    output_path: str = None,
    label: str = "",
) -> bool:
    """Run the dispatch overhead sweep across K values.

    Returns:
        True if all K values pass the < 0.5 ms condition.
    """
    print(f"\nImpl_4 §5.2 -- Dispatch Overhead Benchmark  {label}")
    print(f"{'K':<6} {'mean_ms':<10} {'p50_ms':<10} {'p99_ms':<10} "
          f"{'p999_ms':<10} {'max_ms':<10} {'Pass?'}")
    print("-" * 66)

    rows = []
    all_pass = True

    for K in K_values:
        result = measure_dispatch_overhead(
            K=K, warp_size=warp_size, n_reps=n_reps, warmup=warmup
        )
        rows.append(result)
        marker = "PASS" if result["pass"] else "FAIL"
        if not result["pass"]:
            all_pass = False
        print(f"  {K:<4} {result['mean_ms']:<10.4f} {result['p50_ms']:<10.4f} "
              f"{result['p99_ms']:<10.4f} {result['p999_ms']:<10.4f} "
              f"{result['max_ms']:<10.4f} {marker}")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\nOverall: {verdict}  (threshold: mean < {PASS_THRESHOLD_MS} ms)")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fieldnames = ["K", "warp_size", "n_reps", "mean_ms", "p50_ms",
                      "p99_ms", "p999_ms", "max_ms", "pass"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results written to: {output_path}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Measure AlignmentBuffer.form_batch() CPU overhead (§5.2, §5.5c)"
    )
    parser.add_argument(
        "--K-values", type=int, nargs="+",
        default=[2, 4, 8, 16, 32, 50],
        help="K values to sweep (default: 2 4 8 16 32 50)",
    )
    parser.add_argument(
        "--warp-size", type=int, default=32,
        help="GPU warp width (default: 32)",
    )
    parser.add_argument(
        "--n-reps", type=int, default=10_000,
        help="Number of timing repetitions per K (default: 10000)",
    )
    parser.add_argument(
        "--warmup", type=int, default=500,
        help="Warm-up iterations before measurement (default: 500)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="CSV output path",
    )
    parser.add_argument(
        "--label", type=str, default="",
        help="Hardware label for printed output (e.g. 'Two A6000 PCIe')",
    )
    args = parser.parse_args()

    ok = run_sweep(
        K_values=args.K_values,
        warp_size=args.warp_size,
        n_reps=args.n_reps,
        warmup=args.warmup,
        output_path=args.output,
        label=args.label,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
