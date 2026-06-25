"""
prefetch_policy_ablation.py -- prefetch policy ablation (results/adapter_prefetching).

Policy ablation: compare four policies at K=50 or K=100, K_warm = K//2:
    P0: NoFetch      -- load on demand only (vLLM LRU baseline)
    P1: TopK         -- keep top-K_warm by EWMA rate warm always
    P2: LRU          -- standard LRU eviction (explicit vLLM baseline)
    P3: PredictiveLFU-- our Poisson-scored policy (argmin score = evict)

Combined mode: decompose the WAR + Prefetch gain at K=100.

Experiment runs on real GPU via vLLM servers; policy simulation uses
WarmCacheManager to track cache state across the request stream and
compute hit rate / TTFT overhead analytically.

Usage:
  # policy ablation (two A6000 PCIe, K=50)
  CUDA_VISIBLE_DEVICES=0,1 python scripts/experiments/prefetch_policy_ablation.py \\
      --mode policy-ablation \\
      --model ./models/llama-7b \\
      --adapter-dir ./adapters \\
      --K 50 --K-warm 25 \\
      --lambda-total 7.0 \\
      --hardware-label two_a6000_pcie \\
      --tp-size 2 \\
      --dataset-path ./data/sharegpt/sharegpt.jsonl \\
      --tau-load-ms 200 \\
      --output-dir results/adapter_prefetching/prefetch_ablation/

  # combined WAR + Prefetch (K=50, K_warm=50, no cold-starts)
  CUDA_VISIBLE_DEVICES=0,1 python scripts/experiments/prefetch_policy_ablation.py \\
      --mode combined \\
      --model ./models/llama-7b \\
      --adapter-dir ./adapters \\
      --K 50 --K-warm 50 \\
      --lambda-total 7.0 \\
      --hardware-label two_a6000_pcie \\
      --tp-size 2 \\
      --dataset-path ./data/sharegpt/sharegpt.jsonl \\
      --output-dir results/adapter_prefetching/combined/

  # PP=2 (pipeline parallel, τ_iter≈45ms): splits model layers across GPUs,
  # one activation transfer per forward pass instead of TP all-reduce.
  # τ_iter drops from 100ms → ~45ms; T_max=5ms = 0.11×τ_iter (better for WAR).
  # cold_boost = ceil(96.3/45)+1 = ceil(2.14)+1 = 4.  pcie-auto-cold-boost handles this.
  CUDA_VISIBLE_DEVICES=0,1 python scripts/experiments/prefetch_policy_ablation.py \\
      --mode policy-ablation \\
      --model ./models/llama-7b \\
      --adapter-dir ./adapters \\
      --K 50 --K-warm 25 \\
      --lambda-total 4.0 \\
      --hardware-label two_a6000_pp2 \\
      --tp-size 1 --pp-size 2 \\
      --tau-iter-ms 45.0 \\
      --pcie-auto-cold-boost --pcie-min-deferral \\
      --num-prompts 400 --warmup-prompts 20 --max-tokens 128 \\
      --dataset-path ./data/sharegpt/sharegpt.jsonl \\
      --output-dir results/adapter_prefetching/pp2/

  # DP=2 (data parallel, τ_iter≈30ms each): two independent single-GPU servers
  # with adapter-aware routing (adapter_k→server k%2).  Each server runs on one
  # A6000, gets K//2 adapters, τ_iter≈30ms (same as successful AB10 single-GPU).
  # cold_boost = ceil(96.3/30)+1 = ceil(3.21)+1 = 5.  pcie-auto-cold-boost handles this.
  CUDA_VISIBLE_DEVICES=0,1 python scripts/experiments/prefetch_policy_ablation.py \\
      --mode policy-ablation \\
      --model ./models/llama-7b \\
      --adapter-dir ./adapters \\
      --K 50 --K-warm 25 \\
      --lambda-total 4.0 \\
      --hardware-label two_a6000_dp2 \\
      --tp-size 1 --dp-mode \\
      --tau-iter-ms 30.0 \\
      --pcie-auto-cold-boost --pcie-min-deferral \\
      --num-prompts 400 --warmup-prompts 20 --max-tokens 128 \\
      --dataset-path ./data/sharegpt/sharegpt.jsonl \\
      --output-dir results/adapter_prefetching/dp2/
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
from typing import Dict, List, Tuple, Optional

from adapter_slots.prefetch.cache_manager import WarmCacheManager
from adapter_slots.prefetch.predictor import PredictivePrefetcher
from adapter_slots.control.estimator import ArrivalRateEstimator

SERVER_READY_TIMEOUT = 420
ZIPF_ALPHA = 0.9
N_SIM_REQUESTS = 5000


# Server infrastructure (same fixes as benchmarks/sota/serving_full.py)

def _build_lora_modules(adapter_dir: str, K: int) -> List[str]:
    adapters = sorted(Path(adapter_dir).iterdir())[:K]
    if not adapters:
        raise RuntimeError(f"No adapters in {adapter_dir}")
    return [f"adapter_{i}={p}" for i, p in enumerate(adapters)]


def _build_lora_modules_interleaved(adapter_dir: str, K: int,
                                    server_idx: int, n_servers: int) -> List[str]:
    """Build lora modules for one DP server (interleaved adapter partition).

    Server server_idx gets adapters k where k % n_servers == server_idx.
    Interleaving keeps the Zipf load roughly balanced across servers.
    """
    adapters = sorted(Path(adapter_dir).iterdir())[:K]
    result = []
    for k, p in enumerate(adapters):
        if k % n_servers == server_idx:
            result.append(f"adapter_{k}={p}")
    if not result:
        raise RuntimeError(f"No adapters for server {server_idx} in {adapter_dir}")
    return result


def pcie_calibrated_cold_boost(tau_load_ms: float, tau_iter_ms: float) -> float:
    """Compute PCIe-calibrated cold_boost from measured hardware parameters.

    S-LoRA §4.3 / CaraServe §4.2 recommendation:
        cold_boost = ceil(τ_load / τ_iter) + 1

    For Two A6000 PCIe (τ_load=96.3ms, τ_iter=100ms):
        cold_boost = ceil(0.963) + 1 = 1 + 1 = 2.0

    This ensures cold adapters are deferred for at least 1 full decode iteration
    beyond the one during which their DMA completes, giving the vLLM LRU time
    to update before the adapter is re-ranked by Whittle.
    """
    import math
    return float(math.ceil(tau_load_ms / tau_iter_ms) + 1)


def launch_vllm(model, adapter_dir, K, max_loras, tp_size, port,
                use_adapter_slots=False, tau_iter_ms=30.0, tmax_ms=2.0,
                prefetch_policy="none", tau_load_ms=96.3, cold_boost=1.5,
                pcie_min_deferral=False, pp_size=1, cuda_devices=None,
                lora_modules_override=None):
    """Launch vLLM, optionally with AdapterSlots AlignmentAwareScheduler.

    prefetch_policy: "none" | "lru" | "predictive"
    pcie_min_deferral: bool -- PCIe hard deferral window (S-LoRA §4.3)
    pp_size: int -- pipeline parallel degree (1=off, 2=PP=2)
    cuda_devices: str -- overrides CUDA_VISIBLE_DEVICES (e.g. "0" or "1" for DP=2)
    lora_modules_override: list[str] -- replaces auto-generated lora module list
        (used by DP=2 to load only this server's interleaved adapter partition)
    """
    lora_mods = lora_modules_override if lora_modules_override is not None \
        else _build_lora_modules(adapter_dir, K)
    env = os.environ.copy()
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    if use_adapter_slots:
        env.update({
            "AS_SCHEDULER": "1", "AS_MODE": "sort",
            "AS_TMAX_MS": str(float(tmax_ms)),
            "AS_WAR_TARGET": "0.8", "AS_TTFT_SLO_MS": "2000.0",
            "AS_WHITTLE_DELTA_T": str(round(tau_iter_ms / 1000.0, 6)),
            "AS_PI_KP": "0.01", "AS_PI_KI": "0.001",
            "AS_PI_UPDATE_MODE": "iteration_boundary",
        })
        if prefetch_policy != "none":
            env.update({
                "AS_PREFETCH_POLICY": prefetch_policy,
                "AS_TAU_LOAD_MS": str(float(tau_load_ms)),
                "AS_K_WARM": str(int(max_loras)),
                "AS_COLD_BOOST": str(float(cold_boost)),
            })
            if pcie_min_deferral:
                env["AS_PCIE_MIN_DEFERRAL_S"] = str(round(tau_load_ms / 1000.0, 6))
        cmd = [sys.executable, "scripts/vllm_serve_adapter_slots.py"]
    else:
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]

    cmd += ["--model", model, "--enable-lora",
            "--lora-modules", *lora_mods,
            "--max-loras", str(max_loras),
            "--max-lora-rank", "16",
            "--gpu-memory-utilization", "0.90",
            "--max-num-batched-tokens", "4096",
            "--port", str(port),
            "--disable-log-requests",
            "--disable-frontend-multiprocessing"]
    if tp_size > 1:
        cmd += ["--tensor-parallel-size", str(tp_size)]
    if pp_size > 1:
        cmd += ["--pipeline-parallel-size", str(pp_size)]
    if tp_size > 1 or pp_size > 1:
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return subprocess.Popen(cmd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def stop_server(proc):
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


def wait_for_server(port, timeout=SERVER_READY_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def _load_prompts(dataset_path, n=400):
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
    return prompts or ["Explain the architecture of transformer models."] * n


def _zipf_adapter_picker(K, seed=42):
    rng = random.Random(seed)
    raw = [k ** (-ZIPF_ALPHA) for k in range(1, K + 1)]
    total = sum(raw)
    cum = []
    s = 0.0
    for w in raw:
        s += w / total
        cum.append(s)
    def pick():
        r = rng.random()
        for k, c in enumerate(cum):
            if r <= c:
                return f"adapter_{k}"
        return f"adapter_{K-1}"
    return pick


# Core benchmark

async def _async_bench(port, K, rate, num_prompts, prompts, seed=42,
                       max_tokens=128, warmup_prompts=20, port_router=None):
    """Send num_prompts requests at Poisson rate.

    warmup_prompts: first N requests are sent but excluded from metrics.
    Following S-LoRA/CaraServe practice: run long enough that warmup is
    small relative to measurement window (warmup_prompts/num_prompts ≈ 10%).
    max_tokens=128 matches S-LoRA upper-bound and keeps service_time ≈ 12.8s
    at τ_iter=100ms, allowing ρ < 0.6 at λ=2 req/s.
    port_router: callable(adapter_name) -> int, or None.
        When set (DP=2 mode), routes each request to the server that owns
        that adapter (interleaved: adapter_k → port + (k % 2)).
        When None, all requests go to `port`.
    """
    import aiohttp
    loop = asyncio.get_running_loop()
    pick = _zipf_adapter_picker(K, seed)
    interval = 1.0 / rate
    lats, toks = [], []
    total = warmup_prompts + num_prompts

    async def do_one(session, adapter, prompt):
        target_port = port_router(adapter) if port_router is not None else port
        payload = {"model": adapter, "prompt": prompt,
                   "max_tokens": max_tokens, "temperature": 0.0}
        t0 = loop.time()
        try:
            async with session.post(
                f"http://localhost:{target_port}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                body = await resp.json()
                t1 = loop.time()
                n_out = (body.get("usage") or {}).get("completion_tokens", 0)
                if not n_out:
                    n_out = max(1, len((body.get("choices") or [{}])[0].get("text", "").split()))
                return (t1 - t0) * 1000.0, n_out
        except Exception:
            return None, 0

    tasks = []
    t_start = loop.time()
    connector = aiohttp.TCPConnector(limit=512)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(total):
            tasks.append((i, asyncio.create_task(
                do_one(session, pick(), prompts[i % len(prompts)])
            )))
            if i < total - 1:
                await asyncio.sleep(interval)
        raw_all = [(idx, await t) for idx, t in tasks]
    t_end = loop.time()

    # Discard warmup window; measure only post-warmup requests
    t_warmup_end = warmup_prompts * interval
    duration = t_end - t_start - t_warmup_end
    for idx, r in raw_all:
        if idx >= warmup_prompts:
            if isinstance(r, tuple) and r[0] is not None and r[1] > 0:
                lats.append(r[0])
                toks.append(r[1])
    if not lats:
        return 0.0, 0.0, 0.0, 0
    sl = sorted(lats)
    n = len(sl)
    return (round(sum(toks) / max(duration, 1.0), 1),
            round(sl[n // 2], 1),
            round(sl[min(n-1, int(0.99*n))], 1),
            n)


def run_serving_system(label, model, adapter_dir, K, max_loras, tp_size, port,
                       dataset_path, rate, num_prompts, tau_iter_ms=30.0,
                       use_adapter_slots=False, tmax_ms=2.0,
                       prefetch_policy="none", tau_load_ms=96.3, cold_boost=1.5,
                       max_tokens=128, warmup_prompts=20, pcie_min_deferral=False,
                       pp_size=1):
    prompts = _load_prompts(dataset_path, num_prompts + warmup_prompts + 50)
    print(f"  [{label}] Launching server (K={K}, K_warm={max_loras}, "
          f"TP={tp_size}, PP={pp_size}, "
          f"prefetch={prefetch_policy}, pcie_deferral={pcie_min_deferral})...")
    proc = launch_vllm(model, adapter_dir, K, max_loras, tp_size, port,
                       use_adapter_slots, tau_iter_ms, tmax_ms,
                       prefetch_policy=prefetch_policy,
                       tau_load_ms=tau_load_ms, cold_boost=cold_boost,
                       pcie_min_deferral=pcie_min_deferral,
                       pp_size=pp_size)
    try:
        if not wait_for_server(port):
            print(f"  [{label}] Server did not start!")
            return None
        print(f"  [{label}] Benchmarking rate={rate} num_prompts={num_prompts} "
              f"warmup={warmup_prompts} max_tokens={max_tokens}...")
        tput, p50, p99, n_done = asyncio.run(
            _async_bench(port, K, rate, num_prompts, prompts,
                         max_tokens=max_tokens, warmup_prompts=warmup_prompts)
        )
        print(f"  [{label}] tput={tput:.1f} tok/s  P50={p50:.0f}ms P99={p99:.0f}ms  n={n_done}")
        return dict(tput=tput, e2e_lat_p50=p50, e2e_lat_p99=p99, n_done=n_done)
    finally:
        stop_server(proc)


def run_dp2_serving_system(label, model, adapter_dir, K, max_loras_per_server,
                           port_a, port_b, dataset_path, rate, num_prompts,
                           tau_iter_ms=30.0, use_adapter_slots=False, tmax_ms=2.0,
                           prefetch_policy="none", tau_load_ms=96.3, cold_boost=1.5,
                           max_tokens=128, warmup_prompts=20, pcie_min_deferral=False):
    """Run benchmark with two single-GPU servers (DP=2 mode).

    Interleaved partition: adapter_k → server 0 if k%2==0, server 1 if k%2==1.
    Each server loads K//2 adapters on CUDA_VISIBLE_DEVICES=0 or 1 respectively.
    The benchmark client routes requests using the same interleaving rule.
    Combined throughput = sum(tokens) / wall_time (both servers measured together).
    """
    prompts = _load_prompts(dataset_path, num_prompts + warmup_prompts + 50)
    lora_a = _build_lora_modules_interleaved(adapter_dir, K, server_idx=0, n_servers=2)
    lora_b = _build_lora_modules_interleaved(adapter_dir, K, server_idx=1, n_servers=2)

    print(f"  [{label}] Launching DP=2 servers "
          f"(K={K}, {len(lora_a)} adapters/server, τ_iter={tau_iter_ms}ms, "
          f"prefetch={prefetch_policy}, pcie_deferral={pcie_min_deferral})...")

    proc_a = launch_vllm(model, adapter_dir, K, max_loras_per_server, 1, port_a,
                         use_adapter_slots, tau_iter_ms, tmax_ms,
                         prefetch_policy=prefetch_policy,
                         tau_load_ms=tau_load_ms, cold_boost=cold_boost,
                         pcie_min_deferral=pcie_min_deferral,
                         cuda_devices="0", lora_modules_override=lora_a)
    proc_b = launch_vllm(model, adapter_dir, K, max_loras_per_server, 1, port_b,
                         use_adapter_slots, tau_iter_ms, tmax_ms,
                         prefetch_policy=prefetch_policy,
                         tau_load_ms=tau_load_ms, cold_boost=cold_boost,
                         pcie_min_deferral=pcie_min_deferral,
                         cuda_devices="1", lora_modules_override=lora_b)
    try:
        ok_a = wait_for_server(port_a)
        ok_b = wait_for_server(port_b)
        if not ok_a or not ok_b:
            print(f"  [{label}] One or both servers did not start!")
            return None

        def router(adapter_name):
            k = int(adapter_name.split("_")[1])
            return port_a if k % 2 == 0 else port_b

        print(f"  [{label}] Benchmarking DP=2 rate={rate} num_prompts={num_prompts} "
              f"warmup={warmup_prompts} max_tokens={max_tokens}...")
        tput, p50, p99, n_done = asyncio.run(
            _async_bench(port_a, K, rate, num_prompts, prompts,
                         max_tokens=max_tokens, warmup_prompts=warmup_prompts,
                         port_router=router)
        )
        print(f"  [{label}] tput={tput:.1f} tok/s  P50={p50:.0f}ms P99={p99:.0f}ms  n={n_done}")
        return dict(tput=tput, e2e_lat_p50=p50, e2e_lat_p99=p99, n_done=n_done)
    finally:
        stop_server(proc_a)
        stop_server(proc_b)


# Cache simulation (policy ablation w/o re-running full server)

def simulate_policy(policy_name, K, k_warm, tau_load_ms, lambda_total):
    """
    Simulate cache hit rate for a given policy over N_SIM_REQUESTS Zipf arrivals.
    Returns (hit_rate, cold_fraction, throughput_loss_estimate).
    """
    # default_rate=0.0: unseen adapters start at rate 0, not 1.0.
    # Using 1.0 (the EWMA default for serving) biases TopK/Pred at high K because
    # unseen adapters appear "valuable" until their EWMA converges to their true rate.
    estimator = ArrivalRateEstimator(alpha=0.1, default_rate=0.0, enforce_rank0=False)
    cache = WarmCacheManager(k_warm, tau_load_ms=tau_load_ms, policy=policy_name)
    pick = _zipf_adapter_picker(K, seed=42)
    interval = 1.0 / lambda_total

    for i in range(N_SIM_REQUESTS):
        adapter = pick()
        t_sim = i * interval
        estimator.update(adapter, t_sim)
        rates = estimator.get_all_rates()

        if policy_name == "predictive":
            # Predictive policy: also run prefetch schedule
            predictor = PredictivePrefetcher(tau_load_ms=tau_load_ms, p_thresh=0.3)
            to_prefetch, _ = predictor.prefetch_schedule(rates, cache.warm_set, k_warm)
            for a in to_prefetch:
                cache.prefetch(a, rate_estimates=rates)

        cache.request(adapter, rate_estimates=rates)

    loss = cache.throughput_loss_estimate(lambda_total)
    return cache.hit_rate, cache.cold_fraction, loss


# Policy ablation (2x2 factorial)
#
# Conditions: {WAR alignment} x {PredictiveLFU cold-boost}, all at K_warm=K//2
#   C0: vLLM-LRU      WAR=off, PredLFU=off  baseline
#   C1: AdapterSlots-WAR      WAR=on,  PredLFU=off  WAR gain only
#   C2: AdapterSlots-PredLFU  WAR=off, PredLFU=on   cold-boost only
#   C3: AdapterSlots-Combined WAR=on,  PredLFU=on   full novel system

def run_policy_ablation(model, adapter_dir, K, k_warm, tp_size, hardware_label,
                        dataset_path, output_dir, port, lambda_total, num_prompts,
                        tau_load_ms, tau_iter_ms, max_tokens=128, warmup_prompts=20,
                        cold_boost=1.5, pcie_min_deferral=False, pp_size=1,
                        dp_mode=False):
    """Run 2×2 factorial ablation.

    pp_size: int -- pipeline parallel degree passed to launch_vllm (PP=2 mode).
        τ_iter_ms should reflect the measured PP=2 iteration time (≈45ms for A6000).
    dp_mode: bool -- use two single-GPU servers with interleaved adapter routing.
        When True, tp_size must be 1 (each DP server uses one GPU).
        Port allocation: C0→port, C1→port+2, C2→port+4, C3→port+6
        (each condition uses two consecutive ports for its two servers).
    """
    os.makedirs(output_dir, exist_ok=True)

    parallelism = ("DP=2" if dp_mode else
                   f"PP={pp_size}" if pp_size > 1 else f"TP={tp_size}")
    print(f"\nPrefetch Policy Ablation [{hardware_label}]")
    print(f"  K={K}, K_warm={k_warm}, λ={lambda_total} req/s, τ_load={tau_load_ms}ms")
    print(f"  parallelism={parallelism}  τ_iter={tau_iter_ms}ms")
    print(f"  max_tokens={max_tokens}  warmup={warmup_prompts}  cold_boost={cold_boost}")
    print(f"  pcie_min_deferral={pcie_min_deferral}  (S-LoRA §4.3 hard-block for τ_load={tau_load_ms}ms)")

    # Simulation: cache hit rate under each policy
    print(f"  Cache simulation (N={N_SIM_REQUESTS} requests, Zipf α={ZIPF_ALPHA}):")
    sim_lru  = dict(zip(("hit_rate","cold_fraction","throughput_loss"),
                        simulate_policy("lru",       K, k_warm, tau_load_ms, lambda_total)))
    sim_pred = dict(zip(("hit_rate","cold_fraction","throughput_loss"),
                        simulate_policy("predictive",K, k_warm, tau_load_ms, lambda_total)))
    print(f"    LRU         hit={sim_lru['hit_rate']:.3f}  cold={sim_lru['cold_fraction']:.3f}  "
          f"loss_est={sim_lru['throughput_loss']:.3f}")
    print(f"    PredLFU     hit={sim_pred['hit_rate']:.3f}  cold={sim_pred['cold_fraction']:.3f}  "
          f"loss_est={sim_pred['throughput_loss']:.3f}  "
          f"Δhit={sim_pred['hit_rate']-sim_lru['hit_rate']:+.3f}")

    # Real GPU: 2×2 factorial
    print(f"\n  Real GPU benchmarks (2×2 factorial, all K_warm={k_warm}):")

    k_warm_per_server = max(1, k_warm // 2) if dp_mode else k_warm
    _run = run_dp2_serving_system if dp_mode else run_serving_system

    def _kwargs_base(dp_port_a, use_as, prefetch, tmax=2.0):
        kw = dict(
            tau_iter_ms=tau_iter_ms, use_adapter_slots=use_as, tmax_ms=tmax,
            prefetch_policy=prefetch, tau_load_ms=tau_load_ms, cold_boost=cold_boost,
            max_tokens=max_tokens, warmup_prompts=warmup_prompts,
            pcie_min_deferral=pcie_min_deferral,
        )
        if dp_mode:
            kw["port_a"] = dp_port_a
            kw["port_b"] = dp_port_a + 1
            kw["max_loras_per_server"] = k_warm_per_server
        else:
            kw["tp_size"] = tp_size
            kw["port"] = dp_port_a
            kw["max_loras"] = k_warm
            kw["pp_size"] = pp_size
        return kw

    # Port allocation: DP=2 uses pairs (port, port+1) per condition; TP/PP uses single port
    port_stride = 2 if dp_mode else 1

    # C0: vLLM LRU baseline
    res_c0 = _run(
        "C0:vLLM-LRU", model, adapter_dir, K,
        **_kwargs_base(port, use_as=False, prefetch="none"),
        dataset_path=dataset_path, rate=lambda_total, num_prompts=num_prompts,
    )

    # C1: AdapterSlots WAR only (Whittle, no prefetch boost)
    res_c1 = _run(
        "C1:AdapterSlots-WAR-only", model, adapter_dir, K,
        **_kwargs_base(port + port_stride, use_as=True, prefetch="none"),
        dataset_path=dataset_path, rate=lambda_total, num_prompts=num_prompts,
    )

    # C2: PredictiveLFU cold-boost only
    res_c2 = _run(
        "C2:AdapterSlots-PredLFU-only", model, adapter_dir, K,
        **_kwargs_base(port + 2 * port_stride, use_as=True, prefetch="predictive"),
        dataset_path=dataset_path, rate=lambda_total, num_prompts=num_prompts,
    )

    # C3: WAR + PredictiveLFU -- full novel system
    res_c3 = _run(
        "C3:AdapterSlots-Combined", model, adapter_dir, K,
        **_kwargs_base(port + 3 * port_stride, use_as=True, prefetch="predictive"),
        dataset_path=dataset_path, rate=lambda_total, num_prompts=num_prompts,
    )

    # Gain decomposition
    def _t(res): return res["tput"] if res else 0.0

    t0, t1, t2, t3 = _t(res_c0), _t(res_c1), _t(res_c2), _t(res_c3)
    base = max(t0, 1.0)
    gain_war   = round((t1 - t0) / base, 4)
    gain_pred  = round((t2 - t0) / base, 4)
    gain_comb  = round((t3 - t0) / base, 4)
    interaction = round(gain_comb - gain_war - gain_pred, 4)

    print(f"\n  === Policy Ablation Results [{hardware_label}] ===")
    print(f"  C0 vLLM-LRU           : {t0:.1f} tok/s  (baseline)")
    print(f"  C1 AdapterSlots WAR only      : {t1:.1f} tok/s  Δ={gain_war:+.3f} ({gain_war*100:+.1f}%)")
    print(f"  C2 AdapterSlots PredLFU only  : {t2:.1f} tok/s  Δ={gain_pred:+.3f} ({gain_pred*100:+.1f}%)")
    print(f"  C3 AdapterSlots Combined      : {t3:.1f} tok/s  Δ={gain_comb:+.3f} ({gain_comb*100:+.1f}%)")
    print(f"  Interaction (super-add): {interaction:+.4f}  "
          f"({'super-additive' if interaction > 0 else 'sub-additive'})")

    sim_delta_hit = sim_pred["hit_rate"] - sim_lru["hit_rate"]
    ec12_2a = sim_pred["hit_rate"] >= 0.80
    ec12_2b = sim_delta_hit >= 0.02
    ec12_3  = gain_comb >= 0.05
    print(f"\n  EC 12.2a: sim hit(PredLFU) ≥ 0.80 → {sim_pred['hit_rate']:.3f}  "
          f"{'PASS' if ec12_2a else 'FAIL'}")
    print(f"  EC 12.2b: Δhit ≥ 2pp → {sim_delta_hit:+.3f}  {'PASS' if ec12_2b else 'FAIL'}")
    print(f"  EC 12.3:  combined gain ≥ 5% → {gain_comb:+.3f}  {'PASS' if ec12_3 else 'FAIL'}")

    # Write results CSV
    # Columns labeled e2e_lat_* (end-to-end request latency including generation).
    # Note: "TTFT" = end-to-end latency here (includes decode of max_tokens tokens).
    # For max_tokens=128 at τ_iter=100ms, pure decode ≈ 12.8s.
    # True first-token latency would require streaming TTFT (not yet implemented).
    def _row(label, policy, res, hit_s, cold_s, loss_s, gain):
        return dict(
            hardware_label=hardware_label, K=K, k_warm=k_warm,
            tp_size=tp_size, pp_size=pp_size,
            dp_mode=int(dp_mode), tau_iter_ms=tau_iter_ms,
            lambda_total=lambda_total, tau_load_ms=tau_load_ms,
            label=label, policy=policy,
            hit_rate_sim=round(hit_s, 4),
            cold_fraction_sim=round(cold_s, 4),
            throughput_loss_sim=round(loss_s, 4),
            tput_tok_s=res["tput"] if res else 0.0,
            e2e_lat_p50_ms=res.get("e2e_lat_p50", res.get("ttft_p50", 0)) if res else 0,
            e2e_lat_p99_ms=res.get("e2e_lat_p99", res.get("ttft_p99", 0)) if res else 0,
            gain_vs_baseline=round(gain, 4),
            gain_pct=round(gain * 100, 2),
        )

    rows = [
        _row("C0:vLLM-LRU",        "lru",       res_c0,
             sim_lru["hit_rate"],  sim_lru["cold_fraction"],  sim_lru["throughput_loss"],  0.0),
        _row("C1:AdapterSlots-WAR",         "war_only",  res_c1,
             sim_lru["hit_rate"],  sim_lru["cold_fraction"],  sim_lru["throughput_loss"],  gain_war),
        _row("C2:AdapterSlots-PredLFU",     "predictive",res_c2,
             sim_pred["hit_rate"], sim_pred["cold_fraction"], sim_pred["throughput_loss"], gain_pred),
        _row("C3:AdapterSlots-Combined",    "combined",  res_c3,
             sim_pred["hit_rate"], sim_pred["cold_fraction"], sim_pred["throughput_loss"], gain_comb),
    ]

    out_path = os.path.join(output_dir, f"e12_prefetch_ablation_{hardware_label}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  → {out_path}")
    return rows


# CLI

def main():
    ap = argparse.ArgumentParser(description="Prefetch policy ablation")
    ap.add_argument("--mode", choices=["policy-ablation", "combined"],
                    default="policy-ablation")
    ap.add_argument("--model", default="./models/llama-7b")
    ap.add_argument("--adapter-dir", default="./adapters")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--K-warm", type=int, default=25)
    ap.add_argument("--lambda-total", type=float, default=7.0)
    ap.add_argument("--hardware-label", default="two_a6000_pcie")
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--dataset-path", default="./data/sharegpt/sharegpt.jsonl")
    ap.add_argument("--output-dir", default="results/adapter_prefetching/prefetch_ablation/")
    ap.add_argument("--port", type=int, default=8220)
    ap.add_argument("--tau-load-ms", type=float, default=96.3,
                    help="Measured adapter cold-start time (96.3ms on Two A6000 PCIe)")
    ap.add_argument("--tau-iter-ms", type=float, default=100.0,
                    help="τ_iter for this hardware (AdapterSlots Whittle delta_t)")
    ap.add_argument("--num-prompts", type=int, default=200,
                    help="Measured requests per condition (warmup excluded)")
    ap.add_argument("--warmup-prompts", type=int, default=20,
                    help="Warmup requests sent before measurement window (following S-LoRA/CaraServe practice)")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="Max output tokens per request (128 = ~12.8s service time at τ_iter=100ms)")
    ap.add_argument("--cold-boost", type=float, default=1.5,
                    help="Whittle fill_frac penalty divisor for cold adapters")
    ap.add_argument("--pcie-auto-cold-boost", action="store_true",
                    help="Auto-compute cold_boost = ceil(τ_load/τ_iter)+1 per S-LoRA §4.3 "
                         "(overrides --cold-boost; for Two A6000 PCIe gives 2.0)")
    ap.add_argument("--pcie-min-deferral", action="store_true",
                    help="Enable PCIe minimum deferral window (S-LoRA §4.3): cold adapters "
                         "are hard-blocked from dispatch for τ_load seconds after first arrival, "
                         "ensuring PCIe DMA completes before dispatch. Recommended for "
                         "τ_load/τ_iter ≈ 1 (Two A6000 PCIe).")
    ap.add_argument("--pp-size", type=int, default=1,
                    help="Pipeline parallel degree (1=off, 2=PP=2). "
                         "Use with --tp-size 1 --tau-iter-ms 45 for Two A6000 PP=2. "
                         "PP splits model layers across GPUs; one activation transfer "
                         "per forward pass vs TP all-reduce → τ_iter drops to ~45ms on A6000.")
    ap.add_argument("--dp-mode", action="store_true",
                    help="Data parallel mode: launch two single-GPU servers with interleaved "
                         "adapter routing (adapter_k→server k%%2). Each server gets "
                         "CUDA_VISIBLE_DEVICES=0 or 1. Requires --tp-size 1. "
                         "τ_iter≈30ms per server (same as single A6000). "
                         "Port allocation uses pairs: C0→port+{0,1}, C1→port+{2,3}, etc.")
    args = ap.parse_args()

    if args.dp_mode and args.tp_size != 1:
        ap.error("--dp-mode requires --tp-size 1 (each DP server uses one GPU)")
    if args.pp_size > 1 and args.tp_size > 1:
        ap.error("--pp-size > 1 and --tp-size > 1 simultaneously is not supported "
                 "(use --tp-size 1 --pp-size 2 for PP=2 on Two A6000)")

    cold_boost = args.cold_boost
    if args.pcie_auto_cold_boost:
        cold_boost = pcie_calibrated_cold_boost(args.tau_load_ms, args.tau_iter_ms)
        print(f"PCIe auto cold_boost = {cold_boost:.1f} "
              f"(ceil({args.tau_load_ms}/{args.tau_iter_ms})+1 per S-LoRA §4.3)")

    if args.mode in ("policy-ablation", "combined"):
        run_policy_ablation(
            model=args.model, adapter_dir=args.adapter_dir,
            K=args.K, k_warm=args.K_warm, tp_size=args.tp_size,
            hardware_label=args.hardware_label,
            dataset_path=args.dataset_path, output_dir=args.output_dir,
            port=args.port, lambda_total=args.lambda_total,
            num_prompts=args.num_prompts,
            tau_load_ms=args.tau_load_ms, tau_iter_ms=args.tau_iter_ms,
            max_tokens=args.max_tokens, warmup_prompts=args.warmup_prompts,
            cold_boost=cold_boost,
            pcie_min_deferral=args.pcie_min_deferral,
            pp_size=args.pp_size,
            dp_mode=args.dp_mode,
        )


if __name__ == "__main__":
    main()
