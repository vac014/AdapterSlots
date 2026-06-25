"""
alignment_buffer_queue_monitor.py -- Monitor AlignmentBuffer queue growth during live serving.

Implements experiment §5.3 (Queue Growth Under Load) from alignment_buffer.md.

This script connects to a running vLLM + AlignmentAwareScheduler server and
polls the buffer's queue depth metrics at regular intervals. Used to verify
the bounded-memory invariant: queue depth ≤ 3×W = 96 at λ=10 req/s, T_max=5ms.

Prerequisites:
    vLLM server must be running with AlignmentAwareScheduler and
    --prometheus-port configured, OR the script can scrape the admin endpoint.

Note:
    In this implementation phase, queue metrics are written to the server's
    batch log (AS_BATCH_LOG_PATH env var). This script parses that log file
    in near-real-time to report queue depth over time.

Usage:
    # Monitor during a serving run (run alongside war_improvement_serving_benchmark.py):
    python scripts/experiments/alignment_buffer_queue_monitor.py \
        --batch-log results/alignment_buffer/a6000_single/batch_log_tmax5.0_rate10.jsonl \
        --output results/alignment_buffer/a6000_single/queue_growth.csv \
        --warp-size 32 \
        --max-depth-threshold 96

    # Standalone simulation (no server required) for offline validation:
    python scripts/experiments/alignment_buffer_queue_monitor.py \
        --simulate \
        --K 4 \
        --lambda-total 10 \
        --tmax-ms 5 \
        --duration 60 \
        --warp-size 32 \
        --output results/alignment_buffer/a6000_single/queue_growth.csv
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

from adapter_slots.buffer import AlignmentBuffer


PASS_THRESHOLD_MULTIPLIER = 3  # Queue depth <= 3×W is the pass condition


# Standalone simulation (no server required)

def simulate_queue_growth(
    K: int,
    lambda_total: float,
    tmax_ms: float,
    duration_s: int,
    warp_size: int = 32,
    zipf_alpha: float = 0.9,
    tick_interval_s: float = 0.001,
    output_path: Optional[str] = None,
) -> bool:
    """Simulate AlignmentBuffer under Poisson arrivals and measure queue depth.

    Uses a Zipf arrival distribution to match the real serving workload.

    Args:
        K:               Number of adapters.
        lambda_total:    Total arrival rate (tokens/second).
        tmax_ms:         T_max for the buffer.
        duration_s:      Simulation duration in seconds.
        warp_size:       GPU warp width.
        zipf_alpha:      Zipf skew parameter for per-adapter rates.
        tick_interval_s: Simulation tick interval.
        output_path:     Optional CSV output path.

    Returns:
        True if max queue depth <= 3×W throughout the simulation.
    """
    adapters = [f"adapter_{k}" for k in range(K)]

    # Compute per-adapter arrival rates under Zipf alpha
    weights = [(k + 1) ** (-zipf_alpha) for k in range(K)]
    total_w = sum(weights)
    lambda_k = [lambda_total * w / total_w for w in weights]

    buf = AlignmentBuffer(adapters, warp_size=warp_size, tmax_ms=tmax_ms)

    threshold = PASS_THRESHOLD_MULTIPLIER * warp_size
    rows = []
    max_depth_seen = 0
    t_start = time.monotonic()
    seq_counter = 0

    rng = random.Random(42)

    n_ticks = int(duration_s / tick_interval_s)
    for tick in range(n_ticks):
        t_now = time.monotonic() - t_start

        # Poisson arrivals: sample number of tokens for each adapter this tick
        for k, adapter in enumerate(adapters):
            # Expected arrivals this tick = lambda_k[k] * tick_interval_s
            expected = lambda_k[k] * tick_interval_s
            # Poisson sample: use Bernoulli approximation for small expected values
            n_arrivals = 1 if rng.random() < expected else 0
            for _ in range(n_arrivals):
                buf.enqueue(adapter, seq_id=seq_counter)
                seq_counter += 1

        # Simulate one scheduling tick
        buf.form_batch(max_tokens=warp_size * K)

        depth = buf.max_queue_depth()
        if depth > max_depth_seen:
            max_depth_seen = depth

        if tick % 1000 == 0:
            rows.append({
                "tick": tick,
                "t_s": round(t_now, 3),
                "max_queue_depth": depth,
                "pending_total": buf.stats()["pending_total"],
                "n_timeout_dispatches": buf.stats()["n_timeout_dispatches"],
            })

    passed = max_depth_seen <= threshold

    print(f"\nQueue Growth Simulation (K={K}, λ={lambda_total}, T_max={tmax_ms}ms)")
    print(f"  Threshold:      3×W = {threshold}")
    print(f"  Max depth seen: {max_depth_seen}")
    print(f"  Result:         {'PASS' if passed else 'FAIL'}")
    print(f"  Timeout dispatches: {buf.stats()['n_timeout_dispatches']}")
    print(f"  Total tokens:   {buf.stats()['n_tokens_enqueued']} enqueued, "
          f"{buf.stats()['n_tokens_dispatched']} dispatched")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["tick", "t_s", "max_queue_depth",
                               "pending_total", "n_timeout_dispatches"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Written to: {output_path}")

    return passed


# Live log parser

def monitor_batch_log(
    log_path: str,
    output_path: Optional[str] = None,
    warp_size: int = 32,
    max_depth_threshold: Optional[int] = None,
    poll_interval_s: float = 1.0,
    max_idle_s: float = 60.0,
) -> bool:
    """Monitor the AdapterSlots batch log file and track queue depth over time.

    Parses JSONL lines written by AlignmentBuffer's instrumentation.
    Runs until the log file stops growing (server shut down).

    Returns:
        True if max queue depth stayed within max_depth_threshold (or no
        threshold specified).
    """
    if max_depth_threshold is None:
        max_depth_threshold = PASS_THRESHOLD_MULTIPLIER * warp_size

    print(f"Monitoring: {log_path}")
    print(f"Threshold:  {max_depth_threshold}")

    rows = []
    max_depth_seen = 0
    last_size = -1
    idle_time = 0.0
    t_start = time.monotonic()

    while idle_time < max_idle_s:
        if not os.path.exists(log_path):
            time.sleep(poll_interval_s)
            idle_time += poll_interval_s
            continue

        cur_size = os.path.getsize(log_path)
        if cur_size == last_size:
            idle_time += poll_interval_s
        else:
            idle_time = 0.0
        last_size = cur_size

        # Parse new lines
        try:
            with open(log_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        depth = entry.get("max_queue_depth", 0)
                        if depth > max_depth_seen:
                            max_depth_seen = depth
                        rows.append({
                            "t_s": round(time.monotonic() - t_start, 3),
                            "max_queue_depth": depth,
                            "pending_total": entry.get("pending_total", 0),
                        })
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass

        time.sleep(poll_interval_s)

    passed = max_depth_seen <= max_depth_threshold
    print(f"\nMax queue depth: {max_depth_seen}  Threshold: {max_depth_threshold}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")

    if output_path and rows:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["t_s", "max_queue_depth", "pending_total"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Written to: {output_path}")

    return passed


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="alignment_buffer §5.3 Queue Growth Monitor"
    )
    # Mode selection
    parser.add_argument("--simulate", action="store_true",
                        help="Run standalone simulation (no server required)")
    parser.add_argument("--batch-log", type=str,
                        help="Path to AdapterSlots batch log file to monitor (live mode)")
    # Simulation parameters
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--lambda-total", type=float, default=10.0,
                        help="Total arrival rate (req/s) for simulation")
    parser.add_argument("--tmax-ms", type=float, default=5.0)
    parser.add_argument("--duration", type=int, default=600,
                        help="Simulation duration in seconds")
    parser.add_argument("--zipf-alpha", type=float, default=0.9)
    # Common
    parser.add_argument("--warp-size", type=int, default=32)
    parser.add_argument("--output", type=str,
                        default="results/alignment_buffer/a6000_single/queue_growth.csv")
    parser.add_argument("--max-depth-threshold", type=int, default=None,
                        help="Override queue depth threshold (default: 3×W)")
    args = parser.parse_args()

    if args.simulate:
        ok = simulate_queue_growth(
            K=args.K,
            lambda_total=args.lambda_total,
            tmax_ms=args.tmax_ms,
            duration_s=args.duration,
            warp_size=args.warp_size,
            zipf_alpha=args.zipf_alpha,
            output_path=args.output,
        )
    elif args.batch_log:
        ok = monitor_batch_log(
            log_path=args.batch_log,
            output_path=args.output,
            warp_size=args.warp_size,
            max_depth_threshold=args.max_depth_threshold,
        )
    else:
        parser.error("Specify --simulate or --batch-log")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
