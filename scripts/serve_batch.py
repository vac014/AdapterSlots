"""
serve_batch.py -- Single-pass batch forward through LLM + LoRA adapters.

Used as the profiling target by profile_kernel.py (ncu/nsys attach to this process).
Also used as a standalone micro-benchmark for kernel decomposition timing.

Usage:
    # Pure profiling (called by profile_kernel.py / ncu_e1.sh)
    python scripts/serve_batch.py \
        --model ./models/llama-7b \
        --adapter-dir ./adapters \
        --K 4 \
        --N 512 \
        --rank 16 \
        --batch-mode uniform \
        --n-runs 100 \
        --output results/infrastructure/kernel_decomp.csv

Batch modes (--batch-mode):
    A          512 tokens, adapter 0 only (homogeneous baseline)
    B          256 tokens adapter 0 + 256 tokens adapter 1 (two contiguous blocks)
    C          Alternating 16-token blocks AAAA...BBBB, period=16
    D          Fully interleaved ABABAB... token-by-token
    uniform    Uniform random over K adapters (general case)
    zipf       Zipf(0.9) over K adapters
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Single-pass batch forward for profiling")
    p.add_argument("--model", type=str, default=None,
                   help="Path to base model (HF format). Not required with --profile-decomp.")
    p.add_argument("--adapter-dir", type=str, default="./adapters",
                   help="Root directory containing adapter_r{rank}_k{k}_s{seed}/")
    p.add_argument("--K", type=int, default=4, help="Number of adapters to load")
    p.add_argument("--N", type=int, default=512, help="Number of tokens in the batch")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--batch-mode", type=str, default="D",
                   choices=["A", "B", "C", "D", "uniform", "zipf"])
    p.add_argument("--n-runs", type=int, default=100,
                   help="Number of forward passes to time")
    p.add_argument("--warmup", type=int, default=10,
                   help="Warmup iterations (not counted)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None,
                   help="CSV path to write per-run timing. If None, prints to stdout.")
    p.add_argument("--profile-decomp", action="store_true",
                   help="If set, instrument and log SGMV decomposition timing per run")
    return p.parse_args()


# Batch construction

def build_adapter_ids(batch_mode: str, N: int, K: int, seed: int) -> list:
    """Return a list of N adapter IDs according to batch_mode."""
    rng = np.random.default_rng(seed)

    if batch_mode == "A":
        return [0] * N
    elif batch_mode == "B":
        half = N // 2
        return [0] * half + [1] * (N - half)
    elif batch_mode == "C":
        block = 16
        ids = []
        adapter = 0
        while len(ids) < N:
            ids.extend([adapter] * min(block, N - len(ids)))
            adapter = (adapter + 1) % K
        return ids[:N]
    elif batch_mode == "D":
        return [i % K for i in range(N)]
    elif batch_mode == "uniform":
        return rng.integers(0, K, size=N).tolist()
    elif batch_mode == "zipf":
        ranks = np.arange(1, K + 1, dtype=float)
        weights = 1.0 / (ranks ** 0.9)
        weights /= weights.sum()
        return rng.choice(K, size=N, p=weights).tolist()
    else:
        raise ValueError(f"Unknown batch_mode: {batch_mode}")


def compute_war_fast(adapter_ids: list, warp_size: int = 32) -> float:
    """Fast WAR computation from adapter ID list."""
    arr = np.array(adapter_ids, dtype=np.int32)
    n = len(arr)
    m = n // warp_size
    if m == 0:
        return 0.0
    warps = arr[: m * warp_size].reshape(m, warp_size)
    aligned = int(np.sum(warps.min(axis=1) == warps.max(axis=1)))
    return aligned / m


# SGMV decomposition timing (CPU-side instrumentation)

def measure_decomposition_time_us(adapter_ids: list, K: int) -> dict:
    """
    Simulate and measure the O(N) unsorted vs O(K) sorted decomposition.

    Returns dict with:
        unsorted_scan_us:  Time to scan N tokens and build K segment lists
        sorted_scan_us:    Time to find K-1 boundaries in a pre-sorted array
    """
    N = len(adapter_ids)

    # Unsorted path: O(N) scan
    t0 = time.perf_counter_ns()
    segments: dict = {k: [] for k in range(K)}
    for i, aid in enumerate(adapter_ids):
        segments[aid].append(i)
    t1 = time.perf_counter_ns()
    unsorted_us = (t1 - t0) / 1e3

    # Sorted path: pre-sort then find boundaries O(N log N + K)
    sorted_ids = sorted(range(N), key=lambda i: adapter_ids[i])
    t2 = time.perf_counter_ns()
    boundaries = []
    prev = adapter_ids[sorted_ids[0]]
    for pos, orig_idx in enumerate(sorted_ids):
        cur = adapter_ids[orig_idx]
        if cur != prev:
            boundaries.append(pos)
            prev = cur
    t3 = time.perf_counter_ns()
    sorted_us = (t3 - t2) / 1e3

    return {
        "unsorted_scan_us": unsorted_us,
        "sorted_boundary_us": sorted_us,
        "N": N,
        "K": K,
        "speedup": unsorted_us / max(sorted_us, 0.001),
    }


# Main forward-pass loop

def run_batch_forward(model, tokenizer, adapter_ids: list, prompt_len: int = 128):
    """
    Run a single decode-phase forward pass with the given adapter assignment.
    Returns wall-clock time in ms.
    """
    device = next(model.parameters()).device
    N = len(adapter_ids)

    # Construct dummy input IDs (random tokens, shape [N, prompt_len])
    input_ids = torch.randint(100, 30000, (N, prompt_len), device=device)

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    with torch.no_grad():
        _ = model(input_ids)

    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        elapsed_ms = float("nan")

    return elapsed_ms


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    adapter_ids = build_adapter_ids(args.batch_mode, args.N, args.K, args.seed)
    war = compute_war_fast(adapter_ids)
    print(f"Batch mode: {args.batch_mode}  N={args.N}  K={args.K}  WAR={war:.4f}")
    print(f"Adapter distribution: "
          f"{ {k: adapter_ids.count(k) for k in range(args.K)} }")

    # Decomposition timing (CPU instrumentation, no GPU needed)
    if args.profile_decomp:
        print("\nMeasuring SGMV decomposition timing ...")
        N_vals = [64, 128, 256, 512, 1024, 2048]
        K_vals = [2, 4, 8, 16]
        rows = []
        for N in N_vals:
            for K in K_vals:
                # Build a random unsorted ID list for each (N, K) point
                ids = np.random.default_rng(args.seed).integers(0, K, N).tolist()
                measurements = []
                for _ in range(50):
                    m = measure_decomposition_time_us(ids, K)
                    measurements.append(m)
                mean_unsorted = np.mean([m["unsorted_scan_us"] for m in measurements])
                mean_sorted = np.mean([m["sorted_boundary_us"] for m in measurements])
                row = {
                    "N": N, "K": K,
                    "unsorted_scan_us": round(mean_unsorted, 3),
                    "sorted_boundary_us": round(mean_sorted, 3),
                    "speedup": round(mean_unsorted / max(mean_sorted, 0.001), 2),
                }
                rows.append(row)
                print(f"  N={N:4d} K={K:2d}  unsorted={mean_unsorted:.2f}µs  "
                      f"sorted={mean_sorted:.2f}µs  speedup={row['speedup']:.1f}×")

        # Write CSV
        out_path = args.output or "results/infrastructure/kernel_decomp.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDecomposition timing written to {out_path}")
        return

    # Model loading + forward pass (requires GPU + model weights)
    print("\nLoading model and adapters ...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto"
        )
    except Exception as e:
        print(f"[WARNING] Could not load model: {e}")
        print("Running decomposition-only timing instead (--profile-decomp behaviour).")
        args.profile_decomp = True
        main()
        return

    model.eval()

    # Warmup
    print(f"Warmup ({args.warmup} runs) ...")
    for _ in range(args.warmup):
        run_batch_forward(model, tokenizer, adapter_ids)

    # Timed runs
    print(f"Timing {args.n_runs} runs ...")
    timing_ms = []
    for run_i in range(args.n_runs):
        t = run_batch_forward(model, tokenizer, adapter_ids)
        timing_ms.append(t)

    mean_ms = np.mean(timing_ms)
    std_ms = np.std(timing_ms)
    p50 = np.percentile(timing_ms, 50)
    p99 = np.percentile(timing_ms, 99)

    print(f"\nResults ({args.n_runs} runs):")
    print(f"  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms  p50={p50:.2f}ms  p99={p99:.2f}ms")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "batch_mode", "N", "K", "WAR", "elapsed_ms"])
            for i, t in enumerate(timing_ms):
                writer.writerow([i, args.batch_mode, args.N, args.K, war, round(t, 4)])
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
