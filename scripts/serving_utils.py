"""
serving_utils.py -- Shared utilities for real vLLM server management.

Used by all multi_gpu_correctness and flashinfer_composition live-mode experiment scripts.
Provides:
  - launch_server()        -- start vLLM or AdapterSlots server subprocess
  - wait_for_server()      -- poll /health until ready or timeout
  - kill_server()          -- graceful SIGTERM then SIGKILL with process group
  - async_bench()          -- async aiohttp multi-adapter request sender
  - read_war_from_jsonl()  -- parse AS_METRICS_PATH JSONL for WAR/TTFT stats
  - build_lora_modules()   -- construct --lora-modules argument list
  - load_sharegpt_prompts()-- load prompts from ShareGPT dataset
"""

import asyncio
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


# Constants

SERVER_POLL_INTERVAL = 3          # seconds between /health polls
SERVER_READY_TIMEOUT = 480        # max seconds to wait for server start (TP=2 loads slow)
POST_KILL_SLEEP      = 15         # seconds after kill to let GPU memory release
ZIPF_ALPHA           = 0.9


# Server helpers

def build_lora_modules(adapter_dir: str, K: int) -> List[str]:
    """Return ['adapter_0=path/k0', 'adapter_1=path/k1', ...] for K adapters."""
    adapter_dir = Path(adapter_dir)
    adapters = sorted(adapter_dir.iterdir())[:K]
    if not adapters:
        # Fallback: expect adapter_r16_k{i}_s{42+i} naming
        return [f"adapter_{i}={adapter_dir}/adapter_r16_k{i}_s{42+i}" for i in range(K)]
    return [f"adapter_{i}={str(a)}" for i, a in enumerate(adapters)]


def load_sharegpt_prompts(dataset_path: str, n: int = 500) -> List[str]:
    """Load first-turn human prompts from ShareGPT JSONL dataset."""
    prompts = []
    try:
        with open(dataset_path) as f:
            for line in f:
                if len(prompts) >= n:
                    break
                try:
                    item = json.loads(line)
                    for c in item.get("conversations", []):
                        if c.get("from") == "human":
                            text = c.get("value", "").strip()
                            if 10 <= len(text) <= 1500:
                                prompts.append(text[:500])
                                break
                except Exception:
                    pass
    except Exception:
        pass
    if not prompts:
        prompts = [
            "Explain the key differences between tensor parallelism and pipeline parallelism.",
            "What are the main challenges in serving multiple LoRA adapters simultaneously?",
            "Describe how warp-aligned batching improves SGMV kernel efficiency.",
            "What is the role of the alignment buffer in AdapterSlots?",
        ] * (n // 4 + 1)
    return prompts[:n]


def launch_server(
    mode: str,              # 'vllm' | 'flashinfer' | 'adapterslots' | 'combined'
    model: str,
    adapter_dir: str,
    K: int,
    max_loras: int,
    tp_size: int,
    port: int,
    tau_iter_ms: float,
    tmax_ms: float = 5.0,
    war_target: float = 0.8,
    metrics_path: Optional[str] = None,
    extra_vllm_args: Optional[List[str]] = None,
) -> subprocess.Popen:
    """
    Launch a vLLM server for the given mode.

    mode:
      'vllm'       -- plain vLLM, no AdapterSlots, no FlashInfer
      'flashinfer' -- plain vLLM with --attention-backend flashinfer
      'adapterslots'       -- AdapterSlots alignment scheduler, no FlashInfer
      'combined'   -- AdapterSlots alignment scheduler + FlashInfer backend

    Process management:
      start_new_session=True puts server + all TP workers in a new process group.
      Use kill_server(proc) to clean up the entire group.
      VLLM_WORKER_MULTIPROC_METHOD=spawn required for TP>1 (avoids CUDA fork corruption).
      --disable-frontend-multiprocessing runs engine in-process (avoids orphaned workers).
    """
    lora_modules = build_lora_modules(adapter_dir, K)
    env = os.environ.copy()

    use_adapterslots = mode in ("adapterslots", "combined")
    use_flashinfer = mode in ("flashinfer", "combined")

    if use_adapterslots:
        env.update({
            "AS_SCHEDULER": "1",
            "AS_MODE": "whittle",
            "AS_TMAX_MS": str(float(tmax_ms)),
            "AS_WAR_TARGET": str(float(war_target)),
            "AS_TTFT_SLO_MS": "2000.0",
            "AS_WHITTLE_DELTA_T": str(round(tau_iter_ms / 1000.0, 6)),
            "AS_PI_KP": "0.01",
            "AS_PI_KI": "0.001",
            "AS_PI_UPDATE_MODE": "iteration_boundary",
        })
        if metrics_path:
            env["AS_METRICS_PATH"] = metrics_path
        cmd = [sys.executable, "scripts/vllm_serve_adapter_slots.py"]
    else:
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]

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
        "--disable-frontend-multiprocessing",
    ]
    if tp_size > 1:
        cmd += ["--tensor-parallel-size", str(tp_size)]
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    if use_flashinfer:
        cmd += ["--attention-backend", "flashinfer"]
    if extra_vllm_args:
        cmd += extra_vllm_args

    print(f"  [launch_server] mode={mode} port={port} TP={tp_size}")
    print(f"  cmd: {' '.join(cmd[:6])} ...")
    return subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )


def wait_for_server(port: int, timeout: int = SERVER_READY_TIMEOUT) -> bool:
    """Poll GET /health until 200 or timeout. Returns True if ready."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(SERVER_POLL_INTERVAL)
        print(f"  [wait_for_server] port={port} waiting... ({int(deadline - time.time())}s left)")
    return False


def kill_server(proc: subprocess.Popen, grace_seconds: int = 20):
    """
    Kill the server process group cleanly.
    Step 1: SIGTERM the process group (graceful shutdown of all workers).
    Step 2: Wait up to grace_seconds.
    Step 3: SIGKILL any survivors.
    Step 4: Sleep POST_KILL_SLEEP to let GPU memory fully release.
    """
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
        proc.wait(timeout=grace_seconds)
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
    time.sleep(POST_KILL_SLEEP)


# Async benchmark client

async def _async_multi_adapter_bench(
    port: int,
    K: int,
    rate: float,
    num_prompts: int,
    prompts: List[str],
    max_output_tokens: int = 256,
    seed: int = 42,
) -> Tuple[float, float, float, int]:
    """
    Send num_prompts requests at `rate` req/s to port, Zipf-distributed over K adapters.

    Returns: (throughput_tok_s, ttft_p50_ms, ttft_p99_ms, n_completed)

    Note: 'TTFT' here is end-to-end latency (non-streaming). Use for
    comparative analysis, not absolute TTFT values.
    max_output_tokens=256 ensures enough concurrent decodes for SGMV batching
    to matter (see §9.B.4 analysis in benchmark_serving_full.py).
    """
    try:
        import aiohttp
    except ImportError:
        print("  [bench] aiohttp not installed -- install with: pip install aiohttp")
        return 0.0, 0.0, 0.0, 0

    rng = random.Random(seed)
    raw = [k ** (-ZIPF_ALPHA) for k in range(1, K + 1)]
    total_w = sum(raw)
    cum_weights = []
    cum = 0.0
    for w in raw:
        cum += w / total_w
        cum_weights.append(cum)

    def pick_adapter():
        r = rng.random()
        for k, cw in enumerate(cum_weights):
            if r <= cw:
                return f"adapter_{k}"
        return f"adapter_{K - 1}"

    interval = 1.0 / rate
    result_latencies_ms: List[float] = []
    result_output_toks: List[int] = []

    async def do_one(session, adapter_name: str, prompt: str):
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
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                body = await resp.json()
                t1 = asyncio.get_event_loop().time()
                latency_ms = (t1 - t0) * 1000.0
                n_out = (body.get("usage") or {}).get("completion_tokens", 0)
                if n_out == 0:
                    text = ((body.get("choices") or [{}])[0]).get("text", "")
                    n_out = max(1, len(text.split()))
                return latency_ms, n_out
        except Exception as e:
            return None, 0

    connector = aiohttp.TCPConnector(limit=256, ttl_dns_cache=300)
    t_start = asyncio.get_event_loop().time()
    tasks = []
    async with aiohttp.ClientSession(connector=connector) as session:
        for i in range(num_prompts):
            adapter = pick_adapter()
            prompt = prompts[i % len(prompts)]
            tasks.append(asyncio.create_task(do_one(session, adapter, prompt)))
            if i < num_prompts - 1:
                await asyncio.sleep(interval)
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    t_end = asyncio.get_event_loop().time()
    duration_s = max(t_end - t_start, 1.0)

    for r in raw_results:
        if isinstance(r, tuple) and r[0] is not None and r[1] > 0:
            result_latencies_ms.append(r[0])
            result_output_toks.append(r[1])

    if not result_latencies_ms:
        return 0.0, 0.0, 0.0, 0

    sorted_lat = sorted(result_latencies_ms)
    n = len(sorted_lat)
    tput = sum(result_output_toks) / duration_s
    p50 = sorted_lat[n // 2]
    p99 = sorted_lat[min(n - 1, int(0.99 * n))]
    return round(tput, 1), round(p50, 1), round(p99, 1), n


def run_bench(
    port: int,
    K: int,
    rate: float,
    num_prompts: int,
    prompts: List[str],
    max_output_tokens: int = 256,
    seed: int = 42,
) -> Tuple[float, float, float, int]:
    """Synchronous wrapper around _async_multi_adapter_bench."""
    return asyncio.run(_async_multi_adapter_bench(
        port, K, rate, num_prompts, prompts, max_output_tokens, seed))


# WAR extraction from batch_logger JSONL

def read_war_from_jsonl(path: str, warmup_frac: float = 0.1) -> Dict[str, float]:
    """
    Parse AS_METRICS_PATH JSONL file and compute WAR statistics.

    Each line is a BatchEvent:
      {"tick_id": 42, "timestamp_ms": ..., "war": 0.750, "wartau_ms": ...,
       "halign": ..., "batch_size": 128, ...}

    Returns dict with: war_mean, war_p10, war_p50, war_p90, halign_mean,
                        n_batches, batch_size_mean
    """
    if not path or not os.path.exists(path):
        return {"war_mean": 0.0, "war_p10": 0.0, "war_p50": 0.0,
                "war_p90": 0.0, "halign_mean": 0.0, "n_batches": 0, "batch_size_mean": 0}

    wars = []
    haligns = []
    batch_sizes = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                # Support both JSONL format and CSV format
                war = float(ev.get("war", ev.get("WAR", 0)))
                halign = float(ev.get("halign", ev.get("h_align", 0)))
                bsz = int(ev.get("batch_size", ev.get("n_tokens", 0)))
                if 0 <= war <= 1.0:
                    wars.append(war)
                    haligns.append(halign)
                    batch_sizes.append(bsz)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

    if not wars:
        return {"war_mean": 0.0, "war_p10": 0.0, "war_p50": 0.0,
                "war_p90": 0.0, "halign_mean": 0.0, "n_batches": 0, "batch_size_mean": 0}

    # Drop warmup (first warmup_frac fraction)
    n_warmup = max(0, int(len(wars) * warmup_frac))
    wars = wars[n_warmup:]
    haligns = haligns[n_warmup:]
    batch_sizes = batch_sizes[n_warmup:]

    s = sorted(wars)
    n = len(s)
    return {
        "war_mean": round(sum(wars) / n, 4),
        "war_p10": round(s[max(0, int(0.1 * n))], 4),
        "war_p50": round(s[n // 2], 4),
        "war_p90": round(s[min(n-1, int(0.9 * n))], 4),
        "halign_mean": round(sum(haligns) / max(1, len(haligns)), 4),
        "n_batches": n,
        "batch_size_mean": round(sum(batch_sizes) / max(1, len(batch_sizes)), 1),
    }


# Hardware parameter lookup

HW_PARAMS = {
    "a6000_single": {
        "tau_iter_ms": 30.0,
        "tp_size": 1,
        "label": "a6000_single",
        "cuda_devices": "0",
        "default_rate": 7.0,
        "default_num_prompts": 500,
    },
    "two_a6000_pcie": {
        "tau_iter_ms": 100.0,
        "tp_size": 2,
        "label": "two_a6000_pcie",
        "cuda_devices": "0,1",
        "default_rate": 7.0,
        "default_num_prompts": 500,
    },
    "two_h100_nvlink": {
        "tau_iter_ms": 5.0,
        "tp_size": 2,
        "label": "two_h100_nvlink",
        "cuda_devices": "0,1",
        "default_rate": 15.0,
        "default_num_prompts": 1000,
    },
}
