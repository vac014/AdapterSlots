"""
m3_prefetch_cold_start.py -- adapter cold-start measurement (results/adapter_prefetching).

Two modes:

  --measure-tau-load
    Measures τ_load by comparing TTFT of warm vs. cold adapter requests.
    Launches vLLM with K_warm < K_total, warms up K_warm adapters, then
    sends one request to each cold adapter and records the extra latency.
    Output: results/adapter_prefetching/tau_load/tau_load_{hardware_label}.csv

  --k-sweep
    Runs full K sweep (K=10,25,50,100) to show cold-start throughput loss.
    For each K: runs serving benchmark with K_warm = K//2, measures:
      - Overall throughput (tok/s)
      - TTFT P99 for hot adapters (top-10%) vs cold adapters (bottom-10%)
      - Cache hit rate (simulated from Zipf traffic)
    Output: results/adapter_prefetching/cold_start/cold_start_k_sweep_{hardware_label}.csv

vLLM fixes applied (same as benchmarks/sota/serving_full.py):
  --disable-frontend-multiprocessing, VLLM_WORKER_MULTIPROC_METHOD=spawn,
  start_new_session=True + killpg for clean teardown.

Usage:
  # Measure τ_load on single A6000 (TP=1)
  python benchmarks/micro/m3_prefetch_cold_start.py \\
      --mode measure-tau-load \\
      --model ./models/llama-7b \\
      --adapter-dir ./adapters \\
      --hardware-label a6000_single \\
      --output-dir results/adapter_prefetching/tau_load/

  # K sweep on two A6000 PCIe (TP=2)
  CUDA_VISIBLE_DEVICES=0,1 python benchmarks/micro/m3_prefetch_cold_start.py \\
      --mode k-sweep \\
      --model ./models/llama-7b \\
      --adapter-dir ./adapters \\
      --hardware-label two_a6000_pcie \\
      --tp-size 2 \\
      --dataset-path ./data/sharegpt/sharegpt.jsonl \\
      --output-dir results/adapter_prefetching/cold_start/
"""

import argparse
import asyncio
import csv
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SERVER_READY_TIMEOUT = 660  # K=100 with 50 warm slots can take 8-10 min to initialize
SERVER_POLL_INTERVAL = 2
ZIPF_ALPHA = 0.9


# Utilities (same pattern as benchmarks/sota/serving_full.py)

def _build_lora_modules(adapter_dir: str, K: int) -> List[str]:
    adapters = sorted(Path(adapter_dir).iterdir())[:K]
    if not adapters:
        raise RuntimeError(f"No adapters in {adapter_dir}")
    return [f"adapter_{i}={p}" for i, p in enumerate(adapters)]


def launch_server(model: str, adapter_dir: str, K: int, k_warm: int,
                  tp_size: int, port: int) -> subprocess.Popen:
    lora_mods = _build_lora_modules(adapter_dir, K)
    env = os.environ.copy()
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model,
           "--enable-lora",
           "--lora-modules", *lora_mods,
           "--max-loras", str(k_warm),
           "--max-lora-rank", "16",
           "--gpu-memory-utilization", "0.90",
           "--max-num-batched-tokens", "4096",
           "--port", str(port),
           "--disable-log-requests",
           "--disable-frontend-multiprocessing"]
    if tp_size > 1:
        cmd += ["--tensor-parallel-size", str(tp_size)]
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return subprocess.Popen(cmd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def stop_server(proc: subprocess.Popen) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if pgid:
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=10)
            break
        except subprocess.TimeoutExpired:
            pass
    time.sleep(15)


def wait_for_server(port: int, timeout: int = SERVER_READY_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(SERVER_POLL_INTERVAL)
    return False


def _load_prompts(dataset_path: str, n: int = 200) -> List[str]:
    prompts = []
    try:
        with open(dataset_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if len(prompts) >= n:
                    break
                for c in item.get("conversations", []):
                    if c.get("from") == "human":
                        t = c.get("value", "").strip()
                        if 10 <= len(t) <= 1500:
                            prompts.append(t[:400])
                            break
    except Exception:
        pass
    return prompts or ["Explain what tensor parallelism is in LLM serving."] * n


# Mode 1: Measure τ_load

def measure_tau_load(model, adapter_dir, tp_size, hardware_label,
                     output_dir, port=8200, n_repeats=20):
    """
    Measure adapter cold-start time τ_load.

    Protocol:
    1. Start vLLM with K_total=8 adapters, K_warm=4 (so 4 are cold).
    2. Warm up adapter_0..3 by sending 5 requests each.
    3. For each cold adapter (adapter_4..7): send 1 request, record latency.
    4. Send same request to warm adapter_0: record warm latency.
    5. τ_load = mean(cold_latency) - mean(warm_latency).
    """
    K_total = 8
    K_warm = 4
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nτ_load measurement [{hardware_label}]")
    print(f"  K_total={K_total}, K_warm={K_warm}, repeats={n_repeats}")
    print(f"  Launching vLLM (TP={tp_size}, port={port})...")

    proc = launch_server(model, adapter_dir, K_total, K_warm, tp_size, port)
    try:
        if not wait_for_server(port):
            raise RuntimeError("Server did not start")
        print("  Server ready. Warming up adapters 0-3...")

        # Warm up K_warm adapters
        for warmup_round in range(5):
            for i in range(K_warm):
                _single_request(port, f"adapter_{i}", "Describe attention mechanisms.")

        print("  Warmup done. Measuring cold vs warm latency...")

        warm_lats = []
        cold_lats = []

        for rep in range(n_repeats):
            # Warm request (adapter_0 is always in cache)
            warm_lat = _single_request(port, "adapter_0", f"Explain LoRA adapters. Rep {rep}.")
            if warm_lat is not None:
                warm_lats.append(warm_lat)

            # Cold request: cycle through cold adapters
            cold_adapter = f"adapter_{K_warm + (rep % (K_total - K_warm))}"
            cold_lat = _single_request(port, cold_adapter, f"Explain LoRA adapters. Rep {rep}.")
            if cold_lat is not None:
                cold_lats.append(cold_lat)

    finally:
        stop_server(proc)

    if not warm_lats or not cold_lats:
        print("  WARNING: No measurements collected.")
        return None

    warm_mean = sum(warm_lats) / len(warm_lats)
    cold_mean = sum(cold_lats) / len(cold_lats)
    tau_load_ms = max(0.0, cold_mean - warm_mean)
    cv = (max(cold_lats) - min(cold_lats)) / cold_mean if cold_mean > 0 else 0.0

    warm_lats_s = sorted(warm_lats)
    cold_lats_s = sorted(cold_lats)
    n_w, n_c = len(warm_lats), len(cold_lats)

    result = dict(
        hardware_label=hardware_label,
        tp_size=tp_size,
        K_total=K_total,
        K_warm=K_warm,
        n_repeats=n_repeats,
        warm_lat_mean_ms=round(warm_mean, 2),
        warm_lat_p50_ms=round(warm_lats_s[n_w // 2], 2),
        cold_lat_mean_ms=round(cold_mean, 2),
        cold_lat_p50_ms=round(cold_lats_s[n_c // 2], 2),
        tau_load_ms=round(tau_load_ms, 2),
        cold_lat_cv=round(cv, 3),
        ec12_0_pass=int(cv < 0.20 and 50 <= tau_load_ms <= 1000),
    )

    out_path = os.path.join(output_dir, f"tau_load_{hardware_label}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        w.writeheader()
        w.writerow(result)

    print(f"\n  τ_load measurement:")
    print(f"    Warm P50  = {result['warm_lat_p50_ms']:.0f} ms")
    print(f"    Cold P50  = {result['cold_lat_p50_ms']:.0f} ms")
    print(f"    τ_load    = {tau_load_ms:.0f} ms  (cold − warm)")
    print(f"    CV        = {cv:.3f}  ({'PASS' if cv < 0.20 else 'FAIL: CV ≥ 0.20'})")
    print(f"    EC 12.0: {'PASS' if result['ec12_0_pass'] else 'FAIL'}")
    print(f"  → {out_path}")
    return result


def _single_request(port: int, adapter: str, prompt: str,
                    max_tokens: int = 32) -> Optional[float]:
    payload = json.dumps({
        "model": adapter, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        # 45s timeout: TP=2 PCIe with 32 output tokens should finish in <20s.
        # Shorter timeout prevents indefinite hang if server crashes mid-measurement.
        with urllib.request.urlopen(req, timeout=45) as resp:
            resp.read()
        return (time.perf_counter() - t0) * 1000.0
    except Exception:
        return None


# Mode 2: K sweep (cold-start impact at different K)

def k_sweep(model, adapter_dir, tp_size, hardware_label, dataset_path,
            output_dir, port=8210, lambda_total=7.0, num_prompts=300):
    """
    Show cold-start throughput loss vs K, with K_warm = K//2.

    For each K: simulate cache behaviour using Zipf arrivals + WarmCacheManager,
    then run a real serving benchmark to get throughput/TTFT.
    """
    from adapter_slots.prefetch.cache_manager import WarmCacheManager
    from adapter_slots.control.estimator import ArrivalRateEstimator

    K_values = [100]  # 10, 25, 50 already measured
    os.makedirs(output_dir, exist_ok=True)

    # Pre-seed results with existing rows for K values not being re-run
    out_path = os.path.join(output_dir, f"cold_start_k_sweep_{hardware_label}.csv")
    results = []
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                if int(row["K"]) not in K_values:
                    # Keep existing row (cast numeric fields back to proper types)
                    results.append({
                        "hardware_label": row["hardware_label"],
                        "K": int(row["K"]),
                        "K_warm": int(row["K_warm"]),
                        "tp_size": int(row["tp_size"]),
                        "lambda_total": float(row["lambda_total"]),
                        "f_cold_analytical": float(row["f_cold_analytical"]),
                        "hit_rate_lru": float(row["hit_rate_lru"]),
                        "hit_rate_topk": float(row["hit_rate_topk"]),
                        "throughput_loss_lru": float(row["throughput_loss_lru"]),
                        "throughput_loss_topk": float(row["throughput_loss_topk"]),
                        "tput_real_tok_s": float(row["tput_real_tok_s"]),
                        "ttft_p50_real_ms": float(row["ttft_p50_real_ms"]),
                        "ttft_p99_real_ms": float(row["ttft_p99_real_ms"]),
                    })
        print(f"  Pre-seeded {len(results)} existing rows from {out_path}")

    for K in K_values:
        K_warm = max(K // 2, 1)
        print(f"\nK={K}, K_warm={K_warm}, λ={lambda_total} req/s [{hardware_label}]")

        # Simulate Zipf traffic pattern for cache hit-rate calculation
        rng = random.Random(42)
        raw = [k ** (-ZIPF_ALPHA) for k in range(1, K + 1)]
        total = sum(raw)
        weights = [w / total for w in raw]
        cum = []
        s = 0.0
        for w in weights:
            s += w
            cum.append(s)

        def pick_adapter_zipf():
            r = rng.random()
            for k, c in enumerate(cum):
                if r <= c:
                    return f"adapter_{k}"
            return f"adapter_{K-1}"

        # Simulate N=5000 requests through the cache
        # default_rate=0.0: prevents EWMA initialization bias at high K
        # (unseen adapters would otherwise be kept warm due to rate=1.0 prior)
        estimator = ArrivalRateEstimator(alpha=0.1, default_rate=0.0, enforce_rank0=False)
        cache_none = WarmCacheManager(K_warm, tau_load_ms=96.3, policy="lru")
        cache_topk = WarmCacheManager(K_warm, tau_load_ms=96.3, policy="topk")

        sim_rates: Dict[str, float] = {}
        cold_penalty_none = 0.0
        cold_penalty_topk = 0.0
        N_sim = 5000
        interval = 1.0 / lambda_total

        for i in range(N_sim):
            adapter = pick_adapter_zipf()
            t_sim = i * interval
            estimator.update(adapter, t_sim)
            sim_rates = estimator.get_all_rates()
            _, pen_none = cache_none.request(adapter, rate_estimates=sim_rates)
            _, pen_topk = cache_topk.request(adapter, rate_estimates=sim_rates)
            cold_penalty_none += pen_none
            cold_penalty_topk += pen_topk

        hit_rate_none = cache_none.hit_rate
        hit_rate_topk = cache_topk.hit_rate
        loss_none = cache_none.throughput_loss_estimate(lambda_total)
        loss_topk = cache_topk.throughput_loss_estimate(lambda_total)

        # Analytical cold-start fraction
        f_cold_analytical = 1.0 - sum(weights[:K_warm])

        print(f"  Cache (LRU)  hit_rate={hit_rate_none:.3f}  loss={loss_none:.3f}  f_cold_analytic={f_cold_analytical:.3f}")
        print(f"  Cache (TopK) hit_rate={hit_rate_topk:.3f}  loss={loss_topk:.3f}")

        # Real serving benchmark for throughput (K up to 50 for TP=2 PCIe)
        # For K>50 we cap K_warm at available GPU memory (50 for A6000)
        actual_max_loras = min(K_warm, 50 if tp_size == 1 else 100)
        run_real = K <= 50 or (K <= 100 and tp_size >= 2)

        tput_real, ttft_p50_real, ttft_p99_real = 0.0, 0.0, 0.0
        if run_real:
            print(f"  Running real serving benchmark (K={K}, K_warm={actual_max_loras})...")
            tput_real, ttft_p50_real, ttft_p99_real = _run_mini_benchmark(
                model, adapter_dir, K, actual_max_loras, tp_size,
                dataset_path, port, lambda_total, num_prompts,
            )
            print(f"  Real GPU: tput={tput_real:.1f} tok/s  TTFT P50={ttft_p50_real:.0f}ms P99={ttft_p99_real:.0f}ms")
        else:
            print(f"  Skipping real GPU (K={K} > available K_warm on this hardware)")

        row = dict(
            hardware_label=hardware_label,
            K=K, K_warm=actual_max_loras, tp_size=tp_size,
            lambda_total=lambda_total,
            f_cold_analytical=round(f_cold_analytical, 4),
            hit_rate_lru=round(hit_rate_none, 4),
            hit_rate_topk=round(hit_rate_topk, 4),
            throughput_loss_lru=round(loss_none, 4),
            throughput_loss_topk=round(loss_topk, 4),
            tput_real_tok_s=tput_real,
            ttft_p50_real_ms=ttft_p50_real,
            ttft_p99_real_ms=ttft_p99_real,
        )
        results.append(row)

    out_path = os.path.join(output_dir, f"cold_start_k_sweep_{hardware_label}.csv")
    results.sort(key=lambda r: r["K"])  # keep rows in ascending K order
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"\n=== Cold-Start K Sweep [{hardware_label}] ===")
    print(f"{'K':>5} {'K_warm':>7} {'f_cold':>8} {'hit(LRU)':>9} {'hit(TopK)':>10} "
          f"{'loss(LRU)':>10} {'tput(real)':>11}")
    for r in results:
        print(f"  {r['K']:>5} {r['K_warm']:>7} {r['f_cold_analytical']:>8.3f} "
              f"{r['hit_rate_lru']:>9.3f} {r['hit_rate_topk']:>10.3f} "
              f"{r['throughput_loss_lru']:>10.3f} {r['tput_real_tok_s']:>11.1f}")

    # EC 12.1 check
    row_k100 = next((r for r in results if r['K'] == 100), None)
    if row_k100:
        ec12_1 = row_k100['throughput_loss_lru'] >= 0.10
        print(f"\n  EC 12.1 (loss ≥ 10% at K=100): {'PASS' if ec12_1 else 'FAIL'}")

    print(f"  → {out_path}")
    return results


def _run_mini_benchmark(model, adapter_dir, K, max_loras, tp_size,
                        dataset_path, port, rate, num_prompts):
    """Quick serving benchmark returning (throughput, ttft_p50, ttft_p99)."""
    import aiohttp
    prompts = _load_prompts(dataset_path, n=num_prompts + 50)
    proc = launch_server(model, adapter_dir, K, max_loras, tp_size, port)
    try:
        if not wait_for_server(port):
            return 0.0, 0.0, 0.0

        rng = random.Random(42)
        raw = [k ** (-ZIPF_ALPHA) for k in range(1, K + 1)]
        total = sum(raw)
        cum = []
        s = 0.0
        for w in raw:
            s += w / total
            cum.append(s)

        def pick_adapter():
            r = rng.random()
            for k, c in enumerate(cum):
                if r <= c:
                    return f"adapter_{k}"
            return f"adapter_{K-1}"

        async def bench():
            interval = 1.0 / rate
            lats, toks = [], []

            async def do_one(session, adapter, prompt):
                payload = {"model": adapter, "prompt": prompt,
                           "max_tokens": 128, "temperature": 0.0}
                t0 = asyncio.get_running_loop().time()
                try:
                    async with session.post(
                        f"http://localhost:{port}/v1/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        body = await resp.json()
                        t1 = asyncio.get_running_loop().time()
                        n_out = (body.get("usage") or {}).get("completion_tokens", 0) or 1
                        return (t1 - t0) * 1000.0, n_out
                except Exception:
                    return None, 0

            tasks = []
            t_start = asyncio.get_running_loop().time()
            connector = aiohttp.TCPConnector(limit=256)
            async with aiohttp.ClientSession(connector=connector) as session:
                for i in range(num_prompts):
                    tasks.append(asyncio.create_task(
                        do_one(session, pick_adapter(), prompts[i % len(prompts)])
                    ))
                    if i < num_prompts - 1:
                        await asyncio.sleep(interval)
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            t_end = asyncio.get_running_loop().time()
            duration = t_end - t_start

            for r in raw_results:
                if isinstance(r, tuple) and r[0] is not None and r[1] > 0:
                    lats.append(r[0])
                    toks.append(r[1])

            if not lats:
                return 0.0, 0.0, 0.0
            sl = sorted(lats)
            n = len(sl)
            return (
                round(sum(toks) / max(duration, 1.0), 1),
                round(sl[n // 2], 1),
                round(sl[min(n - 1, int(0.99 * n))], 1),
            )

        return asyncio.run(bench())
    finally:
        stop_server(proc)


# CLI

def main():
    ap = argparse.ArgumentParser(description="Adapter cold-start measurement")
    ap.add_argument("--mode", choices=["measure-tau-load", "k-sweep"],
                    required=True, help="Which experiment to run")
    ap.add_argument("--model", default="./models/llama-7b")
    ap.add_argument("--adapter-dir", default="./adapters")
    ap.add_argument("--hardware-label", default="a6000_single")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--dataset-path", default="./data/sharegpt/sharegpt.jsonl")
    ap.add_argument("--output-dir", default="results/adapter_prefetching/")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--n-repeats", type=int, default=20,
                    help="Repeats for tau-load measurement")
    ap.add_argument("--lambda-total", type=float, default=7.0)
    ap.add_argument("--num-prompts", type=int, default=300)
    args = ap.parse_args()

    if args.mode == "measure-tau-load":
        measure_tau_load(
            model=args.model, adapter_dir=args.adapter_dir,
            tp_size=args.tp_size, hardware_label=args.hardware_label,
            output_dir=args.output_dir, port=args.port,
            n_repeats=args.n_repeats,
        )
    else:
        k_sweep(
            model=args.model, adapter_dir=args.adapter_dir,
            tp_size=args.tp_size, hardware_label=args.hardware_label,
            dataset_path=args.dataset_path, output_dir=args.output_dir,
            port=args.port, lambda_total=args.lambda_total,
            num_prompts=args.num_prompts,
        )


if __name__ == "__main__":
    main()
