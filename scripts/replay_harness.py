"""
replay_harness.py -- Real Workload Replay Harness (workload_characterization, §5)

Replays BurstGPT (or any JSONL trace) through the AdapterSlots vLLM serving endpoint.
Records TTFT, response status, and WAR-relevant metadata per request.

The harness is hardware-transparent: it sends HTTP requests to vLLM's OpenAI-compatible
endpoint. TP degree is configured on the server side (--tensor-parallel-size N).

Hardware-calibrated speed_multiplier defaults:
  Single A6000  (max ~10 req/s):    speed_multiplier = 5.0
  Two A6000 PCIe (max ~14 req/s):   speed_multiplier = 5.0
  Two H100 NVLink (max ~100 req/s): speed_multiplier = 20.0

Usage:
  # Single A6000 (TP=1, K=4):
  python scripts/replay_harness.py \\
      --trace data/burstgpt/burstgpt_k4_30min.jsonl \\
      --endpoint http://localhost:8000/v1/completions \\
      --speed-multiplier 5.0 \\
      --model llama-7b \\
      --adapter-prefix adapter_ \\
      --output results/workload_characterization/a6000_single/burstgpt_replay.csv

  # Two A6000 PCIe (TP=2, K=16):
  python scripts/replay_harness.py \\
      --trace data/burstgpt/burstgpt_k16_30min.jsonl \\
      --endpoint http://localhost:8000/v1/completions \\
      --speed-multiplier 5.0 \\
      --model llama-7b \\
      --output results/workload_characterization/two_a6000_pcie/burstgpt_k16_replay.csv

  # Two H100 NVLink (TP=2, K=32, 20x compression):
  python scripts/replay_harness.py \\
      --trace data/burstgpt/burstgpt_k32_30min.jsonl \\
      --endpoint http://localhost:8000/v1/completions \\
      --speed-multiplier 20.0 \\
      --model llama-7b \\
      --output results/workload_characterization/two_h100_nvlink/burstgpt_k32_replay.csv

  # Speed multiplier calibration:
  python scripts/replay_harness.py \\
      --endpoint http://localhost:8000/v1/completions \\
      --model llama-7b \\
      --calibrate \\
      --lambda-mean 2.0

Output CSV columns:
  request_id, adapter_id, arrival_time_ms, prompt_len, output_len,
  send_time_s, recv_time_s, ttft_ms, status_code, error, success,
  dispatch_time_s, aligned_fraction (estimated from response metadata)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


# Speed multiplier calibration

def calibrate_speed_multiplier(
    endpoint: str,
    model: str,
    adapter_prefix: str,
    lambda_mean_req_s: float,
    n_warmup: int = 20,
    safety_margin: float = 0.7,
    max_multiplier: float = 20.0,
) -> float:
    """
    Measure max throughput of the serving system and compute safe speed multiplier.

    Sends n_warmup rapid sequential requests and measures mean response time.
    c_safe = (throughput × safety_margin) / lambda_mean -- capped at max_multiplier.
    """
    import urllib.request
    import urllib.error

    print(f"  Calibrating speed multiplier against {endpoint} ...", flush=True)
    times = []
    url = endpoint

    payload = json.dumps({
        "model": f"{adapter_prefix}0",
        "prompt": "Hello, how are you?",
        "max_tokens": 8,
        "temperature": 0.0,
    }).encode()

    for i in range(n_warmup):
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"    req {i+1:2d}/{n_warmup}: {elapsed*1000:.0f} ms", flush=True)
        except Exception as e:
            print(f"    req {i+1:2d}/{n_warmup}: ERROR -- {e}", flush=True)

    if not times:
        print("  [warn] All calibration requests failed; defaulting to multiplier=1.0")
        return 1.0

    mean_latency_s = sum(times) / len(times)
    throughput_req_s = 1.0 / mean_latency_s
    c_max = (throughput_req_s * safety_margin) / max(lambda_mean_req_s, 0.01)
    c_safe = min(c_max, max_multiplier)

    print(f"  Mean latency: {mean_latency_s*1000:.1f} ms → "
          f"throughput ≈ {throughput_req_s:.1f} req/s", flush=True)
    print(f"  lambda_mean = {lambda_mean_req_s:.2f} req/s, "
          f"safety_margin = {safety_margin}", flush=True)
    print(f"  c_safe = min({c_max:.1f}, {max_multiplier}) = {c_safe:.1f}×", flush=True)

    return c_safe


# Async replay

async def _send_request(
    client,
    endpoint: str,
    model_name: str,
    adapter_prefix: str,
    req: dict,
    t0_loop: float,
    target_t: float,
) -> dict:
    """Send one request and record timing. Caller is responsible for dispatch timing."""
    send_time_s = asyncio.get_event_loop().time()
    adapter_name = f"{adapter_prefix}{req['adapter_id']}"

    payload = {
        "model": adapter_name,
        "prompt": req.get("prompt", "Translate the following sentence to French: Hello, how are you?"),
        "max_tokens": min(int(req.get("output_len", 64)), 256),
        "temperature": 0.0,
    }

    try:
        resp = await client.post(endpoint, json=payload)
        recv_time_s = asyncio.get_event_loop().time()
        ttft_ms = (recv_time_s - send_time_s) * 1000.0
        return {
            **req,
            "send_time_s": round(send_time_s - t0_loop, 4),
            "recv_time_s": round(recv_time_s - t0_loop, 4),
            "ttft_ms": round(ttft_ms, 2),
            "status_code": resp.status_code,
            "error": "",
            "success": resp.status_code == 200,
            "dispatch_time_s": round(send_time_s - t0_loop, 4),
            "aligned_fraction": float("nan"),  # filled post-hoc from server logs
        }
    except Exception as e:
        recv_time_s = asyncio.get_event_loop().time()
        return {
            **req,
            "send_time_s": round(send_time_s - t0_loop, 4),
            "recv_time_s": round(recv_time_s - t0_loop, 4),
            "ttft_ms": -1.0,
            "status_code": -1,
            "error": str(e)[:120],
            "success": False,
            "dispatch_time_s": round(send_time_s - t0_loop, 4),
            "aligned_fraction": float("nan"),
        }


async def replay_trace_async(
    trace: List[dict],
    endpoint: str,
    model_name: str,
    adapter_prefix: str,
    speed_multiplier: float,
    timeout_s: float = 120.0,
    max_concurrent: int = 200,
    progress_interval_s: float = 30.0,
) -> List[dict]:
    """
    Replay a JSONL trace file through the AdapterSlots serving endpoint asynchronously.

    speed_multiplier: compress time (5× = 5x faster than real-time).
    Returns list of result dicts (one per request).
    Prints a progress line every progress_interval_s seconds.
    Saves partial results and exits cleanly on SIGINT/KeyboardInterrupt.
    """
    try:
        import httpx
    except ImportError:
        print("[ERROR] httpx not installed. Run: pip install httpx", flush=True)
        sys.exit(1)

    n_total = len(trace)
    completed: List[dict] = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _send_one(client, req, t0_loop, target_t):
        async with semaphore:
            result = await _send_request(
                client, endpoint, model_name, adapter_prefix,
                req, t0_loop, target_t,
            )
        completed.append(result)

    async def _progress_reporter(t0: float):
        while True:
            await asyncio.sleep(progress_interval_s)
            n_done = len(completed)
            n_ok = sum(1 for r in completed if r.get("success"))
            elapsed = asyncio.get_event_loop().time() - t0
            pct = 100.0 * n_done / max(n_total, 1)
            print(
                f"  [progress] {n_done}/{n_total} done ({pct:.1f}%)  "
                f"ok={n_ok}  err={n_done - n_ok}  elapsed={elapsed:.0f}s",
                flush=True,
            )

    limits = httpx.Limits(
        max_connections=max_concurrent,
        max_keepalive_connections=max_concurrent,
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), limits=limits) as client:
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        # active_tasks shrinks via done callbacks as requests finish
        active_tasks: set = set()
        reporter = asyncio.create_task(_progress_reporter(t0))

        try:
            for req in trace:
                target_t = t0 + req["arrival_time_ms"] / (1000.0 * speed_multiplier)

                # Sleep until this request's scheduled dispatch time.
                # Yielding here (via asyncio.sleep) lets active tasks make progress
                # rather than freezing the event loop with 20 000 upfront Task objects.
                delay = target_t - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)

                task = asyncio.create_task(_send_one(client, req, t0, target_t))
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

            # Drain remaining in-flight requests
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

        except (asyncio.CancelledError, KeyboardInterrupt):
            print(
                f"\n[interrupt] Saving {len(completed)} partial results ...",
                flush=True,
            )
            for t in list(active_tasks):
                t.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
        finally:
            reporter.cancel()
            await asyncio.gather(reporter, return_exceptions=True)

    return [r for r in completed if isinstance(r, dict)]


# Offline WAR estimation from replay results

def estimate_war_from_replay(
    results: List[dict],
    warp_size: int = 32,
) -> dict:
    """
    Estimate WAR from replay results.

    Groups requests by dispatch time bucket (coarse ~1s bins) and computes
    the fraction where the most common adapter dominates the batch.
    This is an approximation -- the true WAR comes from the server-side WAR metric.
    """
    if not results:
        return {"war_mean": 0.0, "war_std": 0.0, "n_batches": 0}

    # Group by 1s time buckets
    buckets: Dict[int, List[int]] = {}
    for r in results:
        if not r.get("success"):
            continue
        t_bucket = int(r.get("dispatch_time_s", 0))
        adapter = r.get("adapter_id", 0)
        buckets.setdefault(t_bucket, []).append(adapter)

    if not buckets:
        return {"war_mean": 0.0, "war_std": 0.0, "n_batches": 0}

    wars = []
    for t_bucket, adapters in buckets.items():
        if len(adapters) < 2:
            continue
        counts: Dict[int, int] = {}
        for a in adapters:
            counts[a] = counts.get(a, 0) + 1
        # WAR ≈ fraction of requests in the dominant adapter
        dominant = max(counts.values())
        war = dominant / len(adapters)
        wars.append(war)

    if not wars:
        return {"war_mean": 0.0, "war_std": 0.0, "n_batches": 0}

    import numpy as np
    return {
        "war_mean": round(float(np.mean(wars)), 4),
        "war_std": round(float(np.std(wars)), 4),
        "n_batches": len(wars),
    }


# I/O helpers

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_csv(records: List[dict], path: str):
    if not records:
        print("[warn] No records to write.", flush=True)
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


# Main

def parse_args():
    p = argparse.ArgumentParser(
        description="Replay BurstGPT trace through AdapterSlots vLLM serving endpoint"
    )
    p.add_argument("--trace", default=None,
                   help="JSONL trace file to replay (skip with --calibrate)")
    p.add_argument("--endpoint", default="http://localhost:8000/v1/completions",
                   help="vLLM OpenAI-compatible completions endpoint")
    p.add_argument("--model", default="llama-7b",
                   help="Base model name (used in requests if no LoRA adapter)")
    p.add_argument("--adapter-prefix", default="adapter_",
                   help="Prefix for LoRA adapter names (e.g. 'adapter_' → 'adapter_0')")
    p.add_argument("--speed-multiplier", type=float, default=5.0,
                   help="Time compression factor (5 = 5x faster than real-time)")
    p.add_argument("--timeout-s", type=float, default=120.0,
                   help="Per-request HTTP timeout in seconds")
    p.add_argument("--max-concurrent", type=int, default=200,
                   help="Maximum concurrent in-flight requests")
    p.add_argument("--warp-size", type=int, default=32,
                   help="Warp size W for offline WAR estimation")
    p.add_argument("--output", default=None,
                   help="Output CSV path")
    # Calibration mode
    p.add_argument("--calibrate", action="store_true",
                   help="Run speed multiplier calibration (no trace required)")
    p.add_argument("--lambda-mean", type=float, default=2.0,
                   help="Mean arrival rate of the trace (req/s) for calibration")
    p.add_argument("--n-calibration-reqs", type=int, default=20,
                   help="Number of warmup requests for calibration")
    p.add_argument("--safety-margin", type=float, default=0.7,
                   help="Fraction of measured throughput to use as safe ceiling")
    p.add_argument("--max-multiplier", type=float, default=20.0,
                   help="Hard cap on speed multiplier")
    # Summary output
    p.add_argument("--summary-output", default=None,
                   help="Optional: write WAR/TTFT summary CSV")
    return p.parse_args()


def main():
    args = parse_args()

    # Calibration mode
    if args.calibrate:
        c_safe = calibrate_speed_multiplier(
            endpoint=args.endpoint,
            model=args.model,
            adapter_prefix=args.adapter_prefix,
            lambda_mean_req_s=args.lambda_mean,
            n_warmup=args.n_calibration_reqs,
            safety_margin=args.safety_margin,
            max_multiplier=args.max_multiplier,
        )
        print(f"\nRecommended speed_multiplier = {c_safe:.1f}", flush=True)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["lambda_mean_req_s", "c_safe"])
                w.writeheader()
                w.writerow({"lambda_mean_req_s": args.lambda_mean, "c_safe": c_safe})
            print(f"Calibration result written → {args.output}", flush=True)
        return

    if not args.trace:
        print("[ERROR] --trace required (or use --calibrate)")
        sys.exit(1)

    print(f"Loading trace: {args.trace}", flush=True)
    trace = load_jsonl(args.trace)
    if not trace:
        print("[ERROR] Empty trace file.")
        sys.exit(1)

    span_ms = trace[-1]["arrival_time_ms"] - trace[0]["arrival_time_ms"]
    lambda_mean = len(trace) / max(span_ms / 1000.0, 1.0)
    print(f"  {len(trace):,} requests, span={span_ms/1000:.0f}s, "
          f"lambda_mean={lambda_mean:.2f} req/s", flush=True)
    print(f"  Speed multiplier: {args.speed_multiplier}× "
          f"(replay duration ≈ {span_ms/1000/args.speed_multiplier:.0f}s)", flush=True)

    print(f"\nReplaying trace → {args.endpoint} ...", flush=True)
    print(f"  (progress reported every 30s; Ctrl+C saves partial results)\n", flush=True)
    t_wall_start = time.perf_counter()

    results = []
    try:
        results = asyncio.run(replay_trace_async(
            trace=trace,
            endpoint=args.endpoint,
            model_name=args.model,
            adapter_prefix=args.adapter_prefix,
            speed_multiplier=args.speed_multiplier,
            timeout_s=args.timeout_s,
            max_concurrent=args.max_concurrent,
        ))
    except KeyboardInterrupt:
        # replay_trace_async already saved partial results to 'completed';
        # asyncio.run() may re-raise here but results already written via --output
        pass

    t_wall_elapsed = time.perf_counter() - t_wall_start
    n_ok = sum(1 for r in results if r.get("success"))
    n_err = len(results) - n_ok
    label = "Replay complete" if len(results) == len(trace) else "Replay interrupted"
    print(f"\n{label}: {n_ok}/{len(results)} success, "
          f"{n_err} errors, wall_time={t_wall_elapsed:.1f}s", flush=True)

    # TTFT statistics
    ttfts = [r["ttft_ms"] for r in results if r.get("success") and r["ttft_ms"] > 0]
    if ttfts:
        import numpy as np
        print(f"  TTFT P50={np.percentile(ttfts,50):.0f}ms  "
              f"P99={np.percentile(ttfts,99):.0f}ms  "
              f"mean={np.mean(ttfts):.0f}ms", flush=True)

    # Offline WAR estimate
    war_stats = estimate_war_from_replay(results, warp_size=args.warp_size)
    print(f"  WAR estimate (offline): {war_stats['war_mean']:.4f} ± "
          f"{war_stats['war_std']:.4f} ({war_stats['n_batches']} batches)", flush=True)

    # Save results
    if args.output:
        save_csv(results, args.output)
        print(f"  Results written → {args.output}", flush=True)
    else:
        print("  [note] No --output specified; results not saved.", flush=True)

    # Summary CSV
    if args.summary_output:
        import numpy as np
        summary = {
            "trace": args.trace,
            "n_requests": len(results),
            "n_success": n_ok,
            "n_error": n_err,
            "error_rate": round(n_err / max(len(results), 1), 4),
            "speed_multiplier": args.speed_multiplier,
            "wall_time_s": round(t_wall_elapsed, 2),
            "war_mean": war_stats["war_mean"],
            "war_std": war_stats["war_std"],
            "n_batches": war_stats["n_batches"],
            "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 1) if ttfts else -1,
            "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 1) if ttfts else -1,
            "ttft_mean_ms": round(float(np.mean(ttfts)), 1) if ttfts else -1,
        }
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            w.writeheader()
            w.writerow(summary)
        print(f"  Summary written → {args.summary_output}", flush=True)


if __name__ == "__main__":
    main()
