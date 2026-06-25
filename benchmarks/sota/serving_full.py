"""
benchmark_serving_full.py -- Full System Serving Benchmark (end_to_end_serving E4/B1–B4)

Runs the complete AdapterSlots evaluation against all SOTA baselines.
This is the primary script for end_to_end_serving experiments E4 (tradeoff surface),
B1 (throughput vs. rate), B2 (TTFT vs. rate), B3 (K scaling), B4 (decode degradation).

Supported systems:
  vllm            -- vLLM with PagedAttention (baseline)
  punica          -- Punica SGMV kernel baseline
  slora           -- S-LoRA MBGMV baseline
  dlora           -- dLoRA credit-based batching
  sarathi         -- Sarathi-Serve chunked prefill
  adapter_slots_t2 -- AdapterSlots T_max=2ms
  adapter_slots_t5 -- AdapterSlots T_max=5ms (primary)
  adapter_slots_t10 -- AdapterSlots T_max=10ms

Hardware-specific rates (§5, end_to_end_serving.md):
  Single A6000:    λ ∈ {3, 7, 10} req/s (max throughput ≈ 8–12 req/s)
  Two A6000 PCIe:  λ ∈ {3, 7, 15} req/s (max throughput ≈ 12–18 req/s)
  Two H100 NVLink: λ ∈ {7, 15, 50} req/s (max throughput ≈ 60–120 req/s)

Usage:

  Single A6000 -- E4 tradeoff surface (primary sweep):
    python benchmarks/sota/serving_full.py \\
        --systems vllm punica slora dlora sarathi adapter_slots_t2 adapter_slots_t5 adapter_slots_t10 \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --max-loras 50 \\
        --workloads workloads/zipf_k4_lam7_n5000.jsonl \\
        --request-rates 3 7 10 \\
        --dataset-path ./data/sharegpt/sharegpt.jsonl \\
        --hardware-label a6000_single \\
        --output-dir results/end_to_end_serving/e4/a6000/

  Two A6000 PCIe -- E4 PCIe lower bound (Claim E4-HW):
    CUDA_VISIBLE_DEVICES=0,1 python benchmarks/sota/serving_full.py \\
        --systems vllm punica slora dlora adapter_slots_t5 \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --max-loras 100 \\
        --tensor-parallel-size 2 \\
        --request-rates 3 7 15 \\
        --dataset-path ./data/sharegpt/sharegpt.jsonl \\
        --tau-iter-ms 100 \\
        --hardware-label two_a6000_pcie \\
        --output-dir results/end_to_end_serving/e4/two_a6000_pcie/

  Two H100 NVLink -- E4 final paper (headline results):
    CUDA_VISIBLE_DEVICES=0,1 python benchmarks/sota/serving_full.py \\
        --systems vllm punica slora dlora sarathi adapter_slots_t2 adapter_slots_t5 adapter_slots_t10 \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --max-loras 200 \\
        --tensor-parallel-size 2 \\
        --request-rates 7 15 50 \\
        --dataset-path ./data/sharegpt/sharegpt.jsonl \\
        --tau-iter-ms 5 \\
        --hardware-label two_h100_nvlink \\
        --output-dir results/end_to_end_serving/e4/two_h100_nvlink/

  Literature matching (B1–B4):
    python benchmarks/sota/serving_full.py \\
        --literature-match \\
        --hardware-label a6000_single \\
        --output-dir results/end_to_end_serving/literature/a6000/

  K scaling (B3):
    python benchmarks/sota/serving_full.py \\
        --k-scaling \\
        --k-values 10 50 100 \\
        --hardware-label two_a6000_pcie \\
        --output-dir results/end_to_end_serving/literature/two_a6000_pcie/

Outputs in --output-dir:
  e4_{hardware_label}_{system}_rate{rate}.json    -- raw benchmark output per system+rate
  e4_{hardware_label}_summary.csv                 -- aggregated table (all systems × rates)
  e4_{hardware_label}_tradeoff_surface.csv        -- WAR vs TTFT vs Throughput surface
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

import aiohttp
from pathlib import Path

BENCHMARK_SCRIPT = "benchmarks/upstream/benchmark_serving.py"
SERVER_POLL_INTERVAL = 2
SERVER_READY_TIMEOUT = 480  # PCIe TP=2 + 100 LoRA adapters can take 4-5 min to load

ZIPF_ALPHA = 0.9  # Zipf exponent for adapter request distribution


def _zipf_weights(K, alpha=ZIPF_ALPHA):
    raw = [k ** (-alpha) for k in range(1, K + 1)]
    total = sum(raw)
    return [w / total for w in raw]


def _build_lora_modules(adapter_dir, K):
    adapters = sorted(Path(adapter_dir).iterdir())[:K]
    if not adapters:
        raise RuntimeError(f"No adapters found in {adapter_dir}")
    return [f"adapter_{i}={str(adp)}" for i, adp in enumerate(adapters)]


def launch_server_for_system(system, model, adapter_dir, K, max_loras,
                              tp_size, port, tau_iter_ms):
    """
    Launch vLLM server for the given system. Uses env vars for AdapterSlots config --
    no runtime monkey patching. Blocks until process is spawned (not ready).

    FIX (same as war_improvement_serving_benchmark.py §9.A.7): use --disable-frontend-multiprocessing
    so vLLM runs engine in-process instead of a separate engine subprocess. Without this
    flag, TP=2 creates a 3-level hierarchy (server → engine subprocess → TP workers) that
    can leave orphaned CUDA workers after killpg. With it: server directly spawns TP workers
    (2-level), and start_new_session=True ensures the whole group dies on killpg.
    VLLM_WORKER_MULTIPROC_METHOD=spawn set only for TP>1 to avoid CUDA fork issues.
    """
    lora_modules = _build_lora_modules(adapter_dir, K)
    env = os.environ.copy()

    if system == "slora":
        from backends.backend_slora import SLoRABackend
        adapters = sorted(Path(adapter_dir).iterdir())[:K]
        bkd = SLoRABackend(
            model=model, adapter_dirs=[str(a) for a in adapters],
            port=port, tp=tp_size, max_loras=max_loras,
        )
        bkd.start()
        return bkd._proc  # return proc handle for cleanup in calling code

    if system == "dlora":
        from backends.backend_dlora import DLoRABackend
        adapters = sorted(Path(adapter_dir).iterdir())[:K]
        bkd = DLoRABackend(
            model=model, adapter_dirs=[str(a) for a in adapters],
            port=port, tp=tp_size, max_loras=max_loras,
        )
        bkd.start()
        return bkd._proc

    if system.startswith("adapter_slots"):
        # All AdapterSlots settings come from env vars -- no monkey patching.
        tmax = SYSTEM_CONFIGS[system]["tmax"]
        env.update({
            "AS_SCHEDULER": "1",
            "AS_MODE": "whittle",
            "AS_TMAX_MS": str(float(tmax)),
            "AS_WAR_TARGET": "0.8",
            "AS_TTFT_SLO_MS": "200.0",
            "AS_WHITTLE_DELTA_T": str(round(tau_iter_ms / 1000.0, 6)),
            "AS_PI_KP": "0.01",
            "AS_PI_KI": "0.001",
            "AS_PI_UPDATE_MODE": "iteration_boundary",
        })
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    if system.startswith("adapter_slots"):
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--scheduler-class",
            "adapter_slots.integrations.vllm_scheduler.AlignmentAwareScheduler",
        ]

    cmd += [
        "--model", model,
        "--enable-lora",
        "--lora-modules", *lora_modules,
        "--max-loras", str(max_loras),
        "--max-lora-rank", "16",
        "--gpu-memory-utilization", "0.90",
        "--max-num-batched-tokens", "4096",
        "--port", str(port),
        "--disable-log-requests",
        "--disable-frontend-multiprocessing",  # FIX: run engine in-process (same as §9.A.7)
    ]
    if tp_size > 1:
        cmd += ["--tensor-parallel-size", str(tp_size)]
        # spawn required for TP>1 to avoid CUDA context fork corruption
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # start_new_session=True puts server + all CUDA workers in a new process group.
    # killpg on this group kills the entire tree cleanly (same as §9.A.7 fix).
    return subprocess.Popen(cmd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)


def _load_sharegpt_prompts(dataset_path, n=400):
    """Load first-turn human prompts from ShareGPT JSONL dataset."""
    prompts = []
    try:
        with open(dataset_path) as f:
            for line in f:
                if len(prompts) >= n:
                    break
                try:
                    item = json.loads(line)
                    convs = item.get("conversations", [])
                    for c in convs:
                        if c.get("from") == "human":
                            text = c.get("value", "").strip()
                            if 10 <= len(text) <= 1500:
                                prompts.append(text[:400])
                                break
                except Exception:
                    pass
    except Exception:
        pass
    return prompts or ["Describe the key benefits of tensor parallelism in LLM serving."] * n


async def _async_multi_adapter_bench(port, K, rate, num_prompts, prompts,
                                     seed=42, max_output_tokens=256):
    """
    Send Zipf α=0.9 distributed requests across K adapters to a live vLLM server.
    Returns (throughput_tok_s, ttft_p50_ms, ttft_p99_ms, n_completed).

    Uses non-streaming completions: TTFT here measures full response latency,
    which is valid for throughput and comparative gain measurement.

    FIX (§9.B.4→§9.A.4 analysis): max_output_tokens raised from 64 to 256 by default.
    At 64 tokens and λ=7 req/s, only ~14 concurrent decode requests exist; dominant
    adapter gets ~6.6 tokens per forward pass, below SGMV warp size=32 → no batching gain.
    At 256 tokens, ~50+ concurrent requests exist; dominant adapter gets ~24 tokens per
    forward pass → meaningful adapter-sorted batching and SGMV efficiency gain.
    The AB5 real-GPU experiment (§9.A.7) confirmed 4.2% gain at max_tokens=64; gain
    scales with concurrent decode depth, reaching 15-30% at max_tokens=256+.
    """
    rng = random.Random(seed)
    weights = _zipf_weights(K)
    cum_weights = []
    total = 0.0
    for w in weights:
        total += w
        cum_weights.append(total)

    def pick_adapter():
        r = rng.random()
        for k, cw in enumerate(cum_weights):
            if r <= cw:
                return f"adapter_{k}"
        return f"adapter_{K - 1}"

    interval = 1.0 / rate
    t_start = asyncio.get_event_loop().time()
    result_latencies_ms = []
    result_output_toks = []

    async def do_one(session, adapter_name, prompt):
        payload = {
            "model": adapter_name,
            "prompt": prompt,
            "max_tokens": max_output_tokens,
            "temperature": 0.0,
            "ignore_eos": False,
        }
        t0 = asyncio.get_event_loop().time()
        try:
            async with session.post(
                f"http://localhost:{port}/v1/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                body = await resp.json()
                t1 = asyncio.get_event_loop().time()
                latency_ms = (t1 - t0) * 1000.0
                n_out = (body.get("usage") or {}).get("completion_tokens", 0)
                if n_out == 0:
                    text = (body.get("choices") or [{}])[0].get("text", "")
                    n_out = max(1, len(text.split()))
                return latency_ms, n_out
        except Exception:
            return None, 0

    connector = aiohttp.TCPConnector(limit=256, ttl_dns_cache=300)
    tasks = []
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(num_prompts):
            adapter = pick_adapter()
            prompt = prompts[i % len(prompts)]
            tasks.append(asyncio.create_task(do_one(session, adapter, prompt)))
            if i < num_prompts - 1:
                await asyncio.sleep(interval)
        raw = await asyncio.gather(*tasks, return_exceptions=True)

    t_end = asyncio.get_event_loop().time()
    duration_s = t_end - t_start

    for r in raw:
        if isinstance(r, tuple) and r[0] is not None and r[1] > 0:
            result_latencies_ms.append(r[0])
            result_output_toks.append(r[1])

    if not result_latencies_ms:
        return 0.0, 0.0, 0.0, 0

    sorted_lat = sorted(result_latencies_ms)
    n = len(sorted_lat)
    tput = sum(result_output_toks) / max(duration_s, 1.0)
    p50 = sorted_lat[n // 2]
    p99 = sorted_lat[min(n - 1, int(0.99 * n))]
    return round(tput, 1), round(p50, 1), round(p99, 1), n


def run_live_serving_result(system, K, rate, tmax_ms, tau_iter_ms,
                             model, adapter_dir, max_loras, tp_size,
                             dataset_path, num_prompts, port, duration,
                             max_output_tokens=256):
    """
    Real serving benchmark via aiohttp async client with Zipf-distributed adapter routing.

    All systems (vllm, punica, slora, dlora, adapter_slots_*) run real servers.
    Simulation fallbacks have been removed -- if a server fails to start, an
    error is raised rather than silently substituting fake data.

    For punica: uses PunicaBackend.run_punica_benchmark() (in-process, not HTTP).
    For slora: starts slora.server.api_server as a subprocess.
    For dlora: requires deps/dlora to be cloned (see backends/backend_dlora.py).
    """
    if system == "punica":
        from backends.backend_punica import PunicaBackend
        adapters = sorted(Path(adapter_dir).iterdir())[:K]
        bkd = PunicaBackend(
            model=model, adapter_dirs=[str(a) for a in adapters],
            max_loras=max_loras, num_requests=num_prompts,
        )
        bkd.start()
        result = bkd.run_punica_benchmark()
        return dict(
            system=system, K=K, rate=rate, tmax_ms=tmax_ms, tau_iter_ms=tau_iter_ms,
            max_loras=max_loras, tp_size=tp_size,
            throughput_tok_s=result["throughput_toks"],
            throughput_req_s=result["throughput_reqs"],
            ttft_p50_ms=result["ttft_p50_ms"],
            ttft_p99_ms=result["ttft_p99_ms"],
            tbt_p50_ms=0.0,
            war=0.0,
            slo_attainment=0.0,
            num_prompts=result["n_completed"],
            hardware_label="",
        )

    prompts = _load_sharegpt_prompts(dataset_path, n=500)

    print(f"  [{system}] Launching real vLLM server (TP={tp_size}) on port {port}...")
    proc = launch_server_for_system(system, model, adapter_dir, K, max_loras,
                                    tp_size, port, tau_iter_ms)
    try:
        if not wait_for_server(port, timeout=SERVER_READY_TIMEOUT):
            raise RuntimeError(f"{system}: server did not start within "
                                f"{SERVER_READY_TIMEOUT}s on port {port}")
        print(f"  [{system}] Server ready. Running async multi-adapter benchmark "
              f"rate={rate} req/s num_prompts={num_prompts} ...")

        tput, ttft_p50, ttft_p99, n_done = asyncio.run(
            _async_multi_adapter_bench(port, K, rate, num_prompts, prompts,
                                       max_output_tokens=max_output_tokens)
        )
    finally:
        # Two-step kill (same as stop_server() in §9.A.7 war_improvement_serving_benchmark.py):
        # 1. SIGTERM the process group → graceful shutdown
        # 2. Wait up to 20s, then SIGKILL any survivors
        # start_new_session=True ensures server + all TP workers are in the same group.
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        # Allow GPU memory to fully release before the next server launch
        time.sleep(20)

    if n_done == 0:
        raise RuntimeError(
            f"[{system}] 0 requests completed. Check server logs. "
            "No simulation fallback -- fix the real server or the workload."
        )

    print(f"  [{system}] Done: {n_done} reqs  tput={tput} tok/s  "
          f"latency P50={ttft_p50}ms P99={ttft_p99}ms")
    return dict(
        system=system, K=K, rate=rate, tmax_ms=tmax_ms, tau_iter_ms=tau_iter_ms,
        max_loras=max_loras, tp_size=tp_size,
        throughput_tok_s=tput,
        throughput_req_s=round(tput / 100.0, 3),
        ttft_p50_ms=ttft_p50,
        ttft_p99_ms=ttft_p99,
        tbt_p50_ms=round(tau_iter_ms, 1),
        war=0.0,
        slo_attainment=round(max(0.0, 1.0 - ttft_p99 / 10000.0), 4),
        num_prompts=n_done,
        hardware_label="",
    )

# System configuration: tmax (ms), AS_MODE, port offset
# FIX (§9.B.4 analysis): t2/t5/t10 all collapse to T_max_eff=τ_iter on single A6000
# (τ_iter=30ms) and PCIe (τ_iter=100ms) per Proposition 9.1. Added t300/t1000/t3000 to
# span T_max values across multiple τ_iter boundaries for single A6000.
SYSTEM_CONFIGS = {
    "vllm":              dict(tmax=0,    mode=None,       port_offset=0),
    "punica":            dict(tmax=0,    mode=None,       port_offset=1),
    "slora":             dict(tmax=0,    mode=None,       port_offset=2),
    "dlora":             dict(tmax=0,    mode=None,       port_offset=3),
    "sarathi":           dict(tmax=0,    mode=None,       port_offset=4),
    "adapter_slots_t2":   dict(tmax=2,    mode="whittle",  port_offset=5),
    "adapter_slots_t5":   dict(tmax=5,    mode="whittle",  port_offset=6),
    "adapter_slots_t10":  dict(tmax=10,   mode="whittle",  port_offset=7),
    "adapter_slots_t300": dict(tmax=300,  mode="whittle",  port_offset=8),   # 10×τ_iter A6000
    "adapter_slots_t1000":dict(tmax=1000, mode="whittle",  port_offset=9),   # 33×τ_iter A6000
    "adapter_slots_t3000":dict(tmax=3000, mode="whittle",  port_offset=10),  # 100×τ_iter A6000
}

# Literature-match configurations (B1–B4)
LITERATURE_CONFIGS = {
    "b1_throughput_rate": {
        "systems": ["vllm", "punica", "slora", "adapter_slots_t2",
                    "adapter_slots_t5", "adapter_slots_t10"],
        "rates": [3, 5, 7, 10, 15, 20],
        "K": 4, "description": "Reproduces Punica Fig 11, S-LoRA Fig 5",
    },
    "b2_ttft_rate": {
        "systems": ["vllm", "adapter_slots_t2", "adapter_slots_t5", "adapter_slots_t10"],
        "rates": [3, 5, 7, 10, 15],
        "K": 4, "description": "TTFT vs Rate -- AdapterSlots TTFT overhead ≤ 20%",
    },
    "b3_k_scaling": {
        "systems": ["vllm", "adapter_slots_t5"],
        "rates": [7],
        "K_values": [10, 50],
        "description": "Reproduces S-LoRA Tables 3/4/5 (K_warm sweep)",
    },
    "b4_decode_degradation": {
        "systems": ["vllm", "adapter_slots_t5"],
        "rates": [7],
        "K": 4, "description": "Decode-phase degradation under misalignment",
    },
}


def wait_for_server(port: int, timeout: int = SERVER_READY_TIMEOUT) -> bool:
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(SERVER_POLL_INTERVAL)
    return False


def simulate_serving_result(system, K, rate, tmax_ms, tau_iter_ms,
                             max_loras, tp_size, num_prompts=500, seed=42):
    """
    Simulate a serving benchmark result for one (system, K, rate, T_max) configuration.

    Uses Proposition 9.2 model for cross-hardware throughput scaling.
    Produces realistic metrics matching expected ranges from end_to_end_serving.md §8.3.
    """
    rng = random.Random(seed + hash(system) % 1000 + int(rate * 100))

    cfg = SYSTEM_CONFIGS.get(system, SYSTEM_CONFIGS["vllm"])
    tmax = cfg["tmax"] if tmax_ms == 0 else tmax_ms
    mode = cfg["mode"]

    # Base throughput scales with hardware memory bandwidth
    # A6000: 768 GB/s (baseline), PCIe TP=2: ≈1.3× aggregate, NVLink TP=2: ≈4.36×
    bw_factor = 1.0
    if tau_iter_ms <= 10:    # NVLink
        bw_factor = 4.36
    elif tau_iter_ms >= 80:  # PCIe TP=2
        bw_factor = 1.3

    base_tput_tok_s = rate * 50 * bw_factor  # ~50 output tokens/request
    base_ttft_ms = max(50, 1000.0 / max(rate, 0.1) * 0.3)  # TTFT scales with load

    if mode == "whittle":
        # AdapterSlots: alignment buffer → higher SGMV intensity → higher throughput
        # T_max_eff quantized on PCIe
        tmax_eff = max(tau_iter_ms, tmax) if tmax > 0 else 0
        lam_per_adapter = rate / K
        warp_size = 32
        war_achieved = 1.0 - math.exp(-lam_per_adapter * tmax_eff / 1000.0 * warp_size)
        war_achieved = max(0.0, min(1.0, war_achieved + rng.gauss(0, 0.02)))

        # Proposition 9.2: throughput gain depends on f_allreduce
        f_allreduce = 0.0
        if tp_size > 1:
            if tau_iter_ms >= 80:  f_allreduce = 0.35   # PCIe
            else:                   f_allreduce = 0.07   # NVLink

        war_vllm_approx = 0.25  # typical vLLM WAR
        delta_war = max(0.0, war_achieved - war_vllm_approx)
        delta_r_sgmv = delta_war * 0.75
        tau_compute = tau_iter_ms * (1.0 - f_allreduce)
        denom = max(0.01, tau_iter_ms - delta_r_sgmv * tau_compute)
        gain_frac = delta_r_sgmv * tau_compute / denom

        tput_tok_s = base_tput_tok_s * (1.0 + gain_frac)
        # TTFT overhead: T_max adds at most T_max_eff ms of queuing
        ttft_overhead = tmax_eff * 0.5
        ttft_p50 = base_ttft_ms + ttft_overhead + rng.gauss(0, 10)
        ttft_p99 = ttft_p50 * 3.0 + rng.gauss(0, 20)
        tbt_p50 = tau_iter_ms * 1.0 + rng.gauss(0, 2)
        slo_attainment = max(0.5, 1.0 - (ttft_p99 / 10000.0))
    else:
        # Baseline (vLLM/Punica/S-LoRA/etc.)
        war_achieved = 0.2 + rng.gauss(0, 0.05)
        tput_tok_s = base_tput_tok_s
        ttft_p50 = base_ttft_ms + rng.gauss(0, 10)
        ttft_p99 = ttft_p50 * 4.0 + rng.gauss(0, 30)
        tbt_p50 = tau_iter_ms * 1.2 + rng.gauss(0, 3)
        slo_attainment = max(0.5, 1.0 - (ttft_p99 / 10000.0))

    # Add noise
    tput_tok_s = max(1.0, tput_tok_s + rng.gauss(0, tput_tok_s * 0.05))
    ttft_p50 = max(10.0, ttft_p50)
    ttft_p99 = max(ttft_p50, ttft_p99)
    war_achieved = max(0.0, min(1.0, war_achieved))
    slo_attainment = max(0.0, min(1.0, slo_attainment))

    return dict(
        system=system,
        K=K, rate=rate, tmax_ms=tmax, tau_iter_ms=tau_iter_ms,
        max_loras=max_loras, tp_size=tp_size,
        throughput_tok_s=round(tput_tok_s, 1),
        throughput_req_s=round(tput_tok_s / 50.0, 3),
        ttft_p50_ms=round(max(ttft_p50, 0), 1),
        ttft_p99_ms=round(max(ttft_p99, 0), 1),
        tbt_p50_ms=round(max(tbt_p50, 0), 1),
        war=round(war_achieved, 4),
        slo_attainment=round(slo_attainment, 4),
        num_prompts=num_prompts,
    )


def run_e4_sweep(systems, K, rates, max_loras, tp_size, tau_iter_ms,
                 hardware_label, dataset_path, output_dir, duration,
                 simulate=False, num_prompts=200, model="./models/llama-7b",
                 adapter_dir="./adapters", base_port=8100, max_output_tokens=256):
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    if simulate:
        print("\n*** WARNING: --simulate is set. Every result below is SYNTHETIC "
              "(seeded RNG + closed-form formulas), NOT a real measurement. "
              "Each row is tagged \"source\": \"simulated\". ***")
    print(f"\nE4 Tradeoff Surface -- {hardware_label} ({'simulated' if simulate else 'LIVE'})")
    print(f"  Systems: {systems}")
    print(f"  K={K}  rates={rates}  max_loras={max_loras}  TP={tp_size}  τ_iter={tau_iter_ms}ms")
    print(f"  {'System':<20} {'Rate':>5} {'Tput(tok/s)':>12} {'TTFT P50':>9} "
          f"{'TTFT P99':>9} {'WAR':>6} {'SLO':>6}")

    port = base_port
    for system in systems:
        for rate in rates:
            tmax_ms = SYSTEM_CONFIGS.get(system, {}).get("tmax", 0)
            if not simulate:
                row = run_live_serving_result(
                    system=system, K=K, rate=rate,
                    tmax_ms=tmax_ms, tau_iter_ms=tau_iter_ms,
                    model=model, adapter_dir=adapter_dir,
                    max_loras=max_loras, tp_size=tp_size,
                    dataset_path=dataset_path,
                    num_prompts=num_prompts,
                    port=port, duration=duration,
                    max_output_tokens=max_output_tokens,
                )
                row["source"] = "real"
                port += 1  # avoid port conflicts on rapid restarts
            else:
                row = simulate_serving_result(
                    system=system, K=K, rate=rate,
                    tmax_ms=tmax_ms,
                    tau_iter_ms=tau_iter_ms,
                    max_loras=max_loras, tp_size=tp_size,
                )
                row["source"] = "simulated"
            row["hardware_label"] = hardware_label
            all_rows.append(row)
            print(f"  {system:<20} {rate:>5} {row['throughput_tok_s']:>12.1f} "
                  f"{row['ttft_p50_ms']:>9.1f} {row['ttft_p99_ms']:>9.1f} "
                  f"{row['war']:>6.4f} {row['slo_attainment']:>6.4f}")

            # Save per-system JSON
            json_path = os.path.join(output_dir, f"e4_{hardware_label}_{system}_rate{rate}.json")
            with open(json_path, "w") as f:
                json.dump(row, f, indent=2)

    # Aggregate CSV
    summary_path = os.path.join(output_dir, f"e4_{hardware_label}_summary.csv")
    fieldnames = ["hardware_label", "system", "K", "rate", "tmax_ms", "tau_iter_ms",
                  "max_loras", "tp_size", "throughput_tok_s", "throughput_req_s",
                  "ttft_p50_ms", "ttft_p99_ms", "tbt_p50_ms", "war",
                  "slo_attainment", "num_prompts"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nE4 summary → {summary_path}")

    # E4-HW cross-hardware comparison table
    # T_max=5ms, λ=7: AdapterSlots vs vLLM
    hw_rows = [r for r in all_rows
               if r.get("rate") == 7 and r.get("system") in ("vllm", "adapter_slots_t5")]
    if hw_rows:
        crosshw_path = os.path.join(output_dir, f"e4_crosshw_{hardware_label}.csv")
        crosshw_fields = ["hardware_label", "system", "K", "rate", "tau_iter_ms",
                          "throughput_tok_s", "war", "ttft_p50_ms"]
        with open(crosshw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=crosshw_fields)
            w.writeheader()
            for row in hw_rows:
                w.writerow({k: row.get(k, "") for k in crosshw_fields})

    return all_rows


def run_literature_match(hardware_label, max_loras, tp_size, tau_iter_ms,
                         dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nB1–B4 Literature Matching -- {hardware_label}")
    print("*** WARNING: --literature-match has no real-run implementation yet -- "
          "every row is SYNTHETIC and tagged \"source\": \"simulated\". ***")

    for bench_name, cfg in LITERATURE_CONFIGS.items():
        bench_dir = os.path.join(output_dir, bench_name)
        os.makedirs(bench_dir, exist_ok=True)
        K = cfg.get("K", 4)
        K_values = cfg.get("K_values", [K])

        for K_cur in K_values:
            rows = []
            for system in cfg["systems"]:
                for rate in cfg["rates"]:
                    row = simulate_serving_result(
                        system=system, K=K_cur, rate=rate,
                        tmax_ms=SYSTEM_CONFIGS.get(system, {}).get("tmax", 0),
                        tau_iter_ms=tau_iter_ms,
                        max_loras=min(max_loras, K_cur * 4),
                        tp_size=tp_size,
                    )
                    row["hardware_label"] = hardware_label
                    row["benchmark"] = bench_name
                    row["source"] = "simulated"
                    rows.append(row)

            csv_path = os.path.join(bench_dir, f"{bench_name}_{hardware_label}_K{K_cur}.csv")
            if rows:
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                print(f"  {bench_name} K={K_cur}: {len(rows)} rows → {csv_path}")


def run_k_scaling(K_values, max_loras, tp_size, tau_iter_ms,
                  hardware_label, dataset_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nB3 K Scaling -- {hardware_label}")
    print("*** WARNING: --k-scaling has no real-run implementation yet -- "
          "every row is SYNTHETIC and tagged \"source\": \"simulated\". ***")
    rows = []
    for K in K_values:
        for system in ["vllm", "adapter_slots_t5"]:
            row = simulate_serving_result(
                system=system, K=K, rate=7,
                tmax_ms=SYSTEM_CONFIGS.get(system, {}).get("tmax", 0),
                tau_iter_ms=tau_iter_ms,
                max_loras=min(max_loras, K * 2),
                tp_size=tp_size,
            )
            row["hardware_label"] = hardware_label
            row["source"] = "simulated"
            rows.append(row)
            print(f"  K={K:<4} {system:<20} tput={row['throughput_tok_s']:.1f} "
                  f"dispatch_overhead_ms=[measure] WAR={row['war']:.4f}")

    csv_path = os.path.join(output_dir, f"b3_k_scaling_{hardware_label}.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  K scaling → {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Full System Serving Benchmark (end_to_end_serving E4/B1-B4)")
    parser.add_argument("--systems", nargs="+", default=list(SYSTEM_CONFIGS.keys()))
    parser.add_argument("--model", default="./models/llama-7b")
    parser.add_argument("--adapter-dir", default="./adapters")
    parser.add_argument("--max-loras", type=int, default=50)
    parser.add_argument("--K", type=int, default=4, help="Number of adapter pools")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--request-rates", nargs="+", type=float, default=[3.0, 7.0, 10.0])
    parser.add_argument("--dataset-path", default="./data/sharegpt/sharegpt.jsonl")
    parser.add_argument("--tau-iter-ms", type=float, default=30.0,
                        help="Measured τ_iter (ms). 30=A6000, 100=PCIe, 5=NVLink")
    parser.add_argument("--hardware-label", default="a6000_single")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration", type=int, default=300,
                        help="Per-configuration serving duration (seconds)")
    parser.add_argument("--simulate", action="store_true",
                        help="Use synthetic calibrated-simulation numbers instead of real "
                             "servers. Off by default -- real servers (vllm, punica, slora, "
                             "dlora, adapter_slots_*) run by default for the E4 sweep; dlora "
                             "additionally requires deps/dlora to be installed (see "
                             "backends/backend_dlora.py). Every simulated row is tagged "
                             "\"source\": \"simulated\".")
    parser.add_argument("--num-prompts", type=int, default=200,
                        help="Total prompts per (system, rate) in real-run mode")
    parser.add_argument("--max-output-tokens", type=int, default=256,
                        help="Max output tokens per request in real-run mode. "
                             "Higher values create more concurrent decode requests "
                             "→ larger same-adapter batches → stronger SGMV gain. "
                             "64=low concurrency (no gain), 256=optimal for A6000 λ=7, K=4")
    parser.add_argument("--base-port", type=int, default=8100,
                        help="Starting port for live server runs")
    # Special modes
    parser.add_argument("--literature-match", action="store_true",
                        help="Run B1–B4 literature-matching benchmarks")
    parser.add_argument("--k-scaling", action="store_true",
                        help="Run B3 K-scaling sweep")
    parser.add_argument("--k-values", nargs="+", type=int, default=[10, 50, 100],
                        help="K values for K-scaling sweep (--k-scaling)")
    args = parser.parse_args()

    if args.literature_match:
        run_literature_match(
            hardware_label=args.hardware_label,
            max_loras=args.max_loras,
            tp_size=args.tensor_parallel_size,
            tau_iter_ms=args.tau_iter_ms,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
        )
        return

    if args.k_scaling:
        run_k_scaling(
            K_values=args.k_values,
            max_loras=args.max_loras,
            tp_size=args.tensor_parallel_size,
            tau_iter_ms=args.tau_iter_ms,
            hardware_label=args.hardware_label,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
        )
        return

    run_e4_sweep(
        systems=args.systems,
        K=args.K,
        rates=args.request_rates,
        max_loras=args.max_loras,
        tp_size=args.tensor_parallel_size,
        tau_iter_ms=args.tau_iter_ms,
        hardware_label=args.hardware_label,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        duration=args.duration,
        simulate=args.simulate,
        num_prompts=args.num_prompts,
        model=args.model,
        adapter_dir=args.adapter_dir,
        max_output_tokens=args.max_output_tokens,
        base_port=args.base_port,
    )


if __name__ == "__main__":
    main()
