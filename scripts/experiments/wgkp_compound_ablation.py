#!/usr/bin/env python3
"""
wgkp_compound_ablation.py -- E13.8 through E13.12 ablation + headline evaluation.

Experiment map:
  E13.8  Baseline: K=10, vanilla vLLM (no AdapterSlots)
  E13.9  WGKP compound stack: Whittle + WGKP + MWC + fused kernel + macro-batching
  E13.10 APIS two-GPU partition validation
  E13.11 Headline 1.5× throughput number
  E13.12 WAR/GWAR timeline during sustained load

All experiments write CSVs and a summary JSON to --output-dir.

Usage (single A6000, TP=1):
    export CUDA_VISIBLE_DEVICES=0
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    # E13.8 -- baseline
    python scripts/experiments/wgkp_compound_ablation.py \\
        --experiment e13_8 \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --K 10 \\
        --lambda-total 7.0 \\
        --hardware-label a6000_tp1 \\
        --tp-size 1 \\
        --dataset-path ./data/sharegpt/sharegpt.jsonl \\
        --port 8300 \\
        --output-dir results/kernel_promotion/e13_8/

    # E13.9 -- WGKP compound
    python scripts/experiments/wgkp_compound_ablation.py \\
        --experiment e13_9 \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --K 10 \\
        --lambda-total 7.0 \\
        --hardware-label a6000_tp1 \\
        --tp-size 1 \\
        --wgkp-threshold 16 \\
        --macro-n-accum 2 \\
        --mwc-k-hot 5 \\
        --fused-kernel \\
        --dataset-path ./data/sharegpt/sharegpt.jsonl \\
        --port 8301 \\
        --output-dir results/kernel_promotion/e13_9/

    # E13.10 -- APIS two-GPU partition
    python scripts/experiments/wgkp_compound_ablation.py \\
        --experiment e13_10 \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --K 10 \\
        --lambda-total 14.0 \\
        --hardware-label a6000_tp2_apis \\
        --tp-size 1 \\
        --apis-n-gpus 2 \\
        --apis-upstream-urls "http://127.0.0.1:8310,http://127.0.0.1:8311" \\
        --dataset-path ./data/sharegpt/sharegpt.jsonl \\
        --port 8312 \\
        --output-dir results/kernel_promotion/e13_10/

    # E13.11 -- headline 1.5x result
    python scripts/experiments/wgkp_compound_ablation.py \\
        --experiment e13_11 \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --K 10 \\
        --lambda-total 7.0 \\
        --hardware-label a6000_tp1 \\
        --tp-size 1 \\
        --wgkp-threshold 16 \\
        --macro-n-accum 2 \\
        --mwc-k-hot 5 \\
        --fused-kernel \\
        --n-prompts 1000 \\
        --dataset-path ./data/sharegpt/sharegpt.jsonl \\
        --port 8303 \\
        --output-dir results/kernel_promotion/e13_11/

    # E13.12 -- WAR/GWAR timeline
    python scripts/experiments/wgkp_compound_ablation.py \\
        --experiment e13_12 \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --K 10 \\
        --lambda-total 7.0 \\
        --hardware-label a6000_tp1 \\
        --tp-size 1 \\
        --wgkp-threshold 16 \\
        --dataset-path ./data/sharegpt/sharegpt.jsonl \\
        --port 8304 \\
        --output-dir results/kernel_promotion/e13_12/

Exit conditions:
    EC 13.8:  Baseline tput documented; mean_WAR(baseline) < 0.15
    EC 13.9:  compound_gain ≥ 1.30× vs baseline; WAR(WGKP) ≥ 0.60
    EC 13.10: APIS tput ≥ 1.8× TP=2 PCIe (eliminates allreduce bottleneck)
    EC 13.11: headline_gain ≥ 1.45× (target 1.53×); WAR(wgkp) ≥ 0.65
    EC 13.12: GWAR(n*=16) ≥ 0.40; promotion_fraction ≥ 0.30
"""

import argparse
import csv
import json
import os
import pathlib
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Dict, List, Optional, Tuple
import urllib.error


# Constants

EXPERIMENTS = ["e13_8", "e13_9", "e13_10", "e13_11", "e13_12"]
SERVED_MODEL_NAME = "default_model"

# Base env vars shared by all serving calls
_BASE_ENV = {
    "AS_TTFT_SLO_MS": "2000.0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}


# Port utilities (ported from war_improvement_serving_benchmark.py)

def _kill_port_holders(port: int) -> None:
    """Kill every process holding a TCP socket on port using psutil."""
    try:
        import psutil
    except ImportError:
        return
    killed = []
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr.port == port and conn.pid:
            try:
                psutil.Process(conn.pid).kill()
                killed.append(conn.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    if killed:
        time.sleep(1)


def _wait_for_port_free(port: int, timeout: int = 30) -> None:
    """Block until the TCP port is no longer bound."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return
            except OSError:
                time.sleep(1)
    print(f"  [warn] Port {port} still held after {timeout}s -- force-killing holder")
    _kill_port_holders(port)
    time.sleep(2)


# Server lifecycle

def _list_adapters(adapter_dir: str, K: int) -> List[str]:
    """Return up to K unique-k-index adapter paths (one per k, sorted by k index)."""
    base = pathlib.Path(adapter_dir)
    if not base.exists():
        print(f"WARNING: adapter_dir {base} does not exist. Using synthetic paths.")
        return [str(base / f"adapter_r32_k{i}_s{42+i}") for i in range(K)]
    # Prefer rank-32 adapters; fall back to rank-16.
    for pattern in ("adapter_r32_k*_s*", "adapter_r16_k*_s*", "adapter_r16_k*"):
        candidates = sorted(base.glob(pattern))
        if candidates:
            break
    # Pick one path per unique k index to avoid duplicate adapters.
    seen_k: set = set()
    paths: List[str] = []
    for d in candidates:
        try:
            k_idx = int(d.name.split("_k")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        if k_idx in seen_k:
            continue
        seen_k.add(k_idx)
        paths.append(str(d))
        if len(paths) >= K:
            break
    if not paths:
        print(f"WARNING: no adapters found in {base}")
    return paths


def _build_serve_cmd(
    args: argparse.Namespace,
    extra_env: Dict[str, str],
    log_path: str,
) -> Tuple[List[str], Dict[str, str]]:
    """Build vllm serve command (via vllm_serve_adapter_slots.py, which runs
    AlignmentAwareAsyncEngine) and merged environment dict."""
    env = os.environ.copy()
    env.update(_BASE_ENV)
    env.update(extra_env)

    adapters = _list_adapters(args.adapter_dir, args.K)
    lora_args = ["--lora-modules"] + [f"adapter_{i}={p}" for i, p in enumerate(adapters)]

    cmd = [
        sys.executable, "scripts/vllm_serve_adapter_slots.py",
        "--model", args.model,
        "--enable-lora",
        *lora_args,
        "--max-loras", str(args.K),
        "--max-lora-rank", "32",
        "--tensor-parallel-size", str(args.tp_size),
        "--gpu-memory-utilization", "0.90",
        "--max-num-batched-tokens", "4096",
        "--disable-frontend-multiprocessing",
        "--served-model-name", SERVED_MODEL_NAME,
        "--port", str(args.port),
    ]

    if args.tp_size > 1:
        cmd += ["--worker-cls", "adapter_slots.kernel.model_runner.AlignmentAwareWorker"]

    return cmd, env


def _wait_for_server(port: int, proc: subprocess.Popen, timeout_s: float = 180.0, log_path: Optional[str] = None) -> bool:
    """Poll /health until ready, checking for early process exit."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            print(f"  [launch] ERROR: server exited early (rc={rc})")
            _print_log_tail(log_path)
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    print(f"  [launch] ERROR: server did not respond within {timeout_s:.0f}s")
    _print_log_tail(log_path)
    return False


def _print_log_tail(log_path: Optional[str], lines: int = 60) -> None:
    if not log_path:
        return
    try:
        with open(log_path, errors="replace") as f:
            tail = f.readlines()[-lines:]
        print(f"  [diag] Last {len(tail)} lines of {log_path}:")
        print("  " + "-" * 60)
        for line in tail:
            print("  " + line, end="")
        print()
    except FileNotFoundError:
        pass


def _launch_server(cmd: List[str], env: Dict[str, str], port: int, log_path: str) -> Optional[subprocess.Popen]:
    """Launch vllm server subprocess and wait for readiness."""
    _kill_port_holders(port)
    print(f"  [launch] {' '.join(cmd[:6])} ... (port {port})")
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=log_fh, stderr=log_fh,
        start_new_session=True,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    print(f"  [launch] PID={proc.pid}, waiting for server (log: {log_path}) ...")
    if not _wait_for_server(port, proc, timeout_s=240.0, log_path=log_path):
        _stop_server(proc, port)
        return None
    print(f"  [launch] Server ready on port {port}")
    return proc


def _stop_server(proc: Optional[subprocess.Popen], port: Optional[int] = None) -> None:
    """Kill server: SIGTERM process group → SIGKILL survivors → free port."""
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None

    if proc.poll() is None:
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
    if proc.poll() is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    if port is not None:
        _kill_port_holders(port)
        _wait_for_port_free(port, timeout=20)

    fh = getattr(proc, "_log_fh", None)
    if fh is not None:
        fh.close()


# Benchmark runner (inline HTTP -- no subprocess, proper adapter routing)

def _run_benchmark(
    port: int,
    dataset_path: str,
    n_prompts: int,
    lambda_total: float,
    K: int,
    output_json: str,
    extra_label: str = "",
    model: str = "./models/llama-7b",
) -> Dict:
    """Send n_prompts requests to vLLM at lambda_total req/s, cycling K adapters.

    Uses inline threading (no subprocess), routing each request to a specific
    LoRA adapter by name -- the same approach proven in war_improvement_serving_benchmark.py.
    Returns dict with: tput_tok_s, p50_ttft_ms, p99_ttft_ms, label, elapsed_s.
    """
    # Load ShareGPT prompts.
    prompts: List[Tuple[str, int]] = []
    try:
        with open(dataset_path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                items = item if isinstance(item, list) else [item]
                for entry in items:
                    convs = entry.get("conversations", [])
                    if len(convs) >= 2:
                        human = convs[0].get("value", "")
                        gpt   = convs[1].get("value", "")
                        # Skip prompts that are too large for a 2048-token model
                        if len(human.split()) > 400:
                            continue
                        est_prompt = len(human.split()) * 2
                        max_tok = min(
                            128,
                            max(16, 1900 - est_prompt)
                        )
                        prompts.append((human, max_tok))
                    if len(prompts) >= n_prompts:
                        break
                if len(prompts) >= n_prompts:
                    break
    except FileNotFoundError:
        print(f"  [benchmark] ERROR: dataset not found: {dataset_path}")
        return _synthetic_benchmark_result(extra_label, n_prompts, lambda_total)

    if not prompts:
        print("  [benchmark] WARNING: no prompts loaded")
        return _synthetic_benchmark_result(extra_label, n_prompts, lambda_total)

    adapter_names = [f"adapter_{i}" for i in range(K)]
    url = f"http://127.0.0.1:{port}/v1/completions"
    _lock = threading.Lock()
    ttft_ms_list: List[float] = []
    output_tok_sum: List[int] = [0]
    error_count: List[int] = [0]

    def _send(prompt: str, max_tokens: int, adapter: str) -> None:
        payload = json.dumps({
            "model": adapter,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }).encode()
        t0 = time.perf_counter()
        try:
            print(
                f"[DEBUG] Sending request: "
                f"adapter={adapter} "
                f"prompt_words={len(prompt.split())} "
                f"max_tokens={max_tokens}"
            )

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()

                print(
                    f"[DEBUG] Received response: "
                    f"status={resp.status} "
                    f"adapter={adapter}"
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            try:
                data = json.loads(body)

                print("[DEBUG] Response JSON:")
                print(json.dumps(data, indent=2)[:5000])

                n_out = (
                    (data.get("usage") or {})
                    .get("completion_tokens", max_tokens)
                )

            except Exception as parse_exc:
                n_out = max_tokens

                print(
                    f"[DEBUG] Failed to parse response JSON: "
                    f"{type(parse_exc).__name__}: {parse_exc}"
                )

                try:
                    print("[DEBUG] Raw response body:")
                    print(body.decode("utf-8", errors="replace")[:5000])
                except Exception:
                    pass

            with _lock:
                ttft_ms_list.append(elapsed_ms)
                output_tok_sum[0] += n_out

        except urllib.error.HTTPError as exc:
            with _lock:
                error_count[0] += 1

            print("\n" + "=" * 100)
            print("[HTTP ERROR]")
            print(f"Status Code : {exc.code}")
            print(f"Reason      : {exc.reason}")
            print(f"Adapter     : {adapter}")
            print(f"Prompt Words: {len(prompt.split())}")
            print(f"Max Tokens  : {max_tokens}")

            try:
                error_body = exc.read().decode("utf-8", errors="replace")
                print("\nResponse Body:")
                print(error_body[:10000])
            except Exception as body_exc:
                print(f"\nFailed to read error body: {body_exc}")

            print("=" * 100 + "\n")

        except Exception as exc:
            with _lock:
                error_count[0] += 1

            print("\n" + "=" * 100)
            print("[GENERIC ERROR]")
            print(f"Type        : {type(exc).__name__}")
            print(f"Message     : {exc}")
            print(f"Adapter     : {adapter}")
            print(f"Prompt Words: {len(prompt.split())}")
            print(f"Max Tokens  : {max_tokens}")
            print("=" * 100 + "\n")

    interval = 1.0 / lambda_total
    threads: List[threading.Thread] = []
    t_start = time.perf_counter()
    for i, (prompt, max_tok) in enumerate(prompts):
        adapter = adapter_names[i % len(adapter_names)]
        t = threading.Thread(target=_send, args=(prompt, max_tok, adapter), daemon=True)
        t.start()
        threads.append(t)
        next_send = t_start + (i + 1) * interval
        wait = next_send - time.perf_counter()
        if wait > 0:
            time.sleep(wait)

    for t in threads:
        t.join(timeout=300)

    elapsed_total = time.perf_counter() - t_start

    if not ttft_ms_list:
        print(f"  [benchmark] WARNING: 0 successful requests ({error_count[0]} errors)")
        return _synthetic_benchmark_result(extra_label, n_prompts, lambda_total)

    ttft_sorted = sorted(ttft_ms_list)
    n = len(ttft_sorted)
    result = {
        "label": extra_label,
        "tput_tok_s": output_tok_sum[0] / max(elapsed_total, 1e-6),
        "p50_ttft_ms": statistics.mean(ttft_ms_list),
        "p99_ttft_ms": ttft_sorted[min(int(0.99 * n), n - 1)],
        "elapsed_s": round(elapsed_total, 2),
        "n_completed": n,
        "n_errors": error_count[0],
    }
    pathlib.Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [benchmark] {n} completed, {error_count[0]} errors, "
          f"tput={result['tput_tok_s']:.1f} tok/s, "
          f"p99_ttft={result['p99_ttft_ms']:.0f}ms")
    return result


def _parse_metrics_jsonl(metrics_path: str) -> Dict:
    """Parse AS_METRICS_PATH JSONL to extract mean WAR, GWAR, promotion_fraction."""
    war_values, gwar_values, promo_values = [], [], []
    try:
        with open(metrics_path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if "war" in rec:
                        war_values.append(float(rec["war"]))
                    if "gwar" in rec:
                        gwar_values.append(float(rec["gwar"]))
                    if "promotion_fraction" in rec:
                        promo_values.append(float(rec["promotion_fraction"]))
                except (json.JSONDecodeError, ValueError):
                    continue
    except FileNotFoundError:
        pass
    return {
        "mean_war": sum(war_values) / len(war_values) if war_values else 0.0,
        "mean_gwar": sum(gwar_values) / len(gwar_values) if gwar_values else 0.0,
        "mean_promotion_fraction": sum(promo_values) / len(promo_values) if promo_values else 0.0,
        "n_batches": len(war_values),
    }


def _synthetic_benchmark_result(label: str, n_prompts: int, lambda_total: float) -> Dict:
    """Placeholder result when server is not available (for dry-run / CI testing)."""
    return {
        "label": label,
        "tput_tok_s": 0.0,
        "p50_ttft_ms": 0.0,
        "p99_ttft_ms": 0.0,
        "elapsed_s": 0.0,
        "synthetic": True,
    }


# Per-experiment runners

def run_e13_8(args: argparse.Namespace) -> Dict:
    """E13.8 -- Baseline K=10 vanilla vLLM (no AdapterSlots).

    Establishes baseline throughput and WAR floor.
    EC 13.8: mean_WAR(baseline) < 0.15 (confirms alignment problem exists).
    """
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = str(out_dir / "e13_8_metrics.jsonl")

    log_path = str(out_dir / "e13_8_server.log")
    extra_env = {
        "AS_MODE": "disabled",
        "AS_METRICS_PATH": metrics_path,
    }
    cmd, env = _build_serve_cmd(args, extra_env, log_path)
    proc = _launch_server(cmd, env, args.port, log_path)

    try:
        result = _run_benchmark(
            port=args.port,
            dataset_path=args.dataset_path,
            n_prompts=args.n_prompts,
            lambda_total=args.lambda_total,
            K=args.K,
            output_json=str(out_dir / "e13_8_benchmark.json"),
            extra_label="baseline_vllm",
            model=args.model,
        )
    finally:
        _stop_server(proc, args.port)

    metrics = _parse_metrics_jsonl(metrics_path)
    result.update(metrics)

    summary = {
        "experiment": "E13.8",
        "hardware": args.hardware_label,
        "K": args.K,
        "lambda_total": args.lambda_total,
        **result,
    }
    _write_summary(out_dir / "e13_8_summary.json", summary)
    _print_ec(summary, {
        "EC 13.8a (mean_WAR < 0.15)": lambda s: s.get("mean_war", 1.0) < 0.15,
        "EC 13.8b (tput_tok_s > 0)":  lambda s: s.get("tput_tok_s", 0) > 0,
    })
    return summary


def run_e13_9(args: argparse.Namespace) -> Dict:
    """E13.9 -- WGKP compound stack ablation.

    Runs WGKP mode with Whittle ranking, MWC, fused kernel, macro-batching.
    Requires baseline result from E13.8 to compute compound_gain.
    EC 13.9: compound_gain ≥ 1.30×; WAR(WGKP) ≥ 0.60.
    """
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = str(out_dir / "e13_9_metrics.jsonl")

    log_path = str(out_dir / "e13_9_server.log")
    tmax_ms = "90.0" if args.tp_size == 1 else "300.0"
    extra_env = {
        "AS_MODE": "wgkp",
        "AS_WAR_TARGET": "0.80",
        "AS_TMAX_MS": tmax_ms,
        "AS_LOG_WAR": "1",
        "AS_WGKP_THRESHOLD": str(args.wgkp_threshold),
        "AS_WGKP_APT": "1" if args.wgkp_apt else "0",
        "AS_MWC_K_HOT": str(args.mwc_k_hot),
        "AS_MWC_MEMORY_GB": "10.0",
        "AS_FUSED_KERNEL": "1" if args.fused_kernel else "0",
        "AS_MACRO_N_ACCUM": str(args.macro_n_accum),
        "AS_WHITTLE_DELTA_T": "0.030" if args.tp_size == 1 else "0.100",
        "AS_WGKP_LOG": str(out_dir / "e13_9_wgkp.jsonl"),
        "AS_METRICS_PATH": metrics_path,
    }
    cmd, env = _build_serve_cmd(args, extra_env, log_path)
    proc = _launch_server(cmd, env, args.port, log_path)

    try:
        result = _run_benchmark(
            port=args.port,
            dataset_path=args.dataset_path,
            n_prompts=args.n_prompts,
            lambda_total=args.lambda_total,
            K=args.K,
            output_json=str(out_dir / "e13_9_benchmark.json"),
            extra_label="wgkp_compound",
            model=args.model,
        )
    finally:
        _stop_server(proc, args.port)

    metrics = _parse_metrics_jsonl(metrics_path)
    result.update(metrics)

    # Load baseline tput from E13.8 if available
    baseline_json = pathlib.Path(args.output_dir).parent / "e13_8" / "e13_8_summary.json"
    baseline_tput = 0.0
    try:
        with open(baseline_json) as f:
            baseline_tput = float(json.load(f).get("tput_tok_s", 0))
    except FileNotFoundError:
        pass

    compound_gain = (result["tput_tok_s"] / baseline_tput) if baseline_tput > 0 else 0.0
    result["compound_gain"] = round(compound_gain, 4)
    result["baseline_tput_tok_s"] = baseline_tput

    summary = {
        "experiment": "E13.9",
        "hardware": args.hardware_label,
        "K": args.K,
        "lambda_total": args.lambda_total,
        "wgkp_threshold": args.wgkp_threshold,
        "macro_n_accum": args.macro_n_accum,
        "mwc_k_hot": args.mwc_k_hot,
        **result,
    }
    _write_summary(out_dir / "e13_9_summary.json", summary)
    _print_ec(summary, {
        "EC 13.9a (compound_gain ≥ 1.30)": lambda s: s.get("compound_gain", 0) >= 1.30,
        "EC 13.9b (mean_WAR ≥ 0.60)":      lambda s: s.get("mean_war", 0) >= 0.60,
        "EC 13.9c (mean_gwar ≥ 0.30)":     lambda s: s.get("mean_gwar", 0) >= 0.30,
    })
    return summary


def run_e13_10(args: argparse.Namespace) -> Dict:
    """E13.10 -- APIS two-GPU partition validation.

    Launches two independent TP=1 vLLM servers and routes through APISRouter.
    Validates that APIS eliminates PCIe allreduce, achieving τ_iter~30ms on each
    partition (vs ~100ms for TP=2 PCIe).
    EC 13.10: APIS tput ≥ 1.8× TP=2 PCIe baseline.
    """
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.apis_n_gpus < 2:
        print("E13.10 requires --apis-n-gpus ≥ 2. Skipping.")
        return {"experiment": "E13.10", "skipped": True}

    # Split adapters across GPU partitions
    adapters = _list_adapters(args.adapter_dir, args.K)
    mid = len(adapters) // 2
    partitions = [adapters[:mid], adapters[mid:]]
    procs = []
    upstream_ports = []

    for gpu_idx, partition_adapters in enumerate(partitions):
        port = args.port + gpu_idx
        upstream_ports.append(port)
        env = os.environ.copy()
        env.update(_BASE_ENV)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        env["AS_MODE"] = "wgkp"
        env["AS_WAR_TARGET"] = "0.80"
        env["AS_WGKP_THRESHOLD"] = str(args.wgkp_threshold)
        env["AS_MWC_K_HOT"] = str(args.mwc_k_hot)
        env["AS_WHITTLE_DELTA_T"] = "0.030"
        env["AS_METRICS_PATH"] = str(out_dir / f"e13_10_gpu{gpu_idx}_metrics.jsonl")

        cmd = [
            sys.executable,
            "scripts/vllm_serve_adapter_slots.py",
            "--model", args.model,
            "--enable-lora",
            "--lora-modules", *[f"adapter_{i}={p}" for i, p in enumerate(partition_adapters)],
            "--max-loras", str(len(partition_adapters) + 1),
            "--max-lora-rank", "32",
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", "0.90",
            "--max-num-batched-tokens", "4096",
            "--disable-frontend-multiprocessing",
            "--port", str(port),
        ]
        log_path = str(out_dir / f"e13_10_gpu{gpu_idx}_server.log")
        proc = _launch_server(cmd, env, port, log_path)
        procs.append((proc, port))

    # Launch APIS router on --port + 2
    router_port = args.port + 2
    upstream_urls = ",".join(f"http://127.0.0.1:{p}" for p in upstream_ports)
    router_log = str(out_dir / "e13_10_router.log")
    router_log_fh = open(router_log, "wb")
    router_cmd = [
        sys.executable, "-m", "adapter_slots.integrations.apis_router",
        "--port", str(router_port),
        "--upstream-urls", upstream_urls,
        "--n-gpus", str(args.apis_n_gpus),
        "--rebalance-interval", "30",
    ]
    router_proc = subprocess.Popen(
        router_cmd, stdout=router_log_fh, stderr=router_log_fh,
        start_new_session=True,
    )
    router_proc._log_fh = router_log_fh  # type: ignore[attr-defined]
    time.sleep(3)

    try:
        result = _run_benchmark(
            port=router_port,
            dataset_path=args.dataset_path,
            n_prompts=args.n_prompts,
            lambda_total=args.lambda_total,
            K=args.K,
            output_json=str(out_dir / "e13_10_benchmark.json"),
            extra_label="apis_two_gpu",
            model=args.model,
        )
    finally:
        _stop_server(router_proc, router_port)
        for proc, port in procs:
            _stop_server(proc, port)

    # Load TP=2 PCIe baseline tput if available
    pcie_baseline = pathlib.Path(args.output_dir).parent.parent / \
        "adapter_prefetching" / "combined_pcie_fix" / "e12_prefetch_ablation_two_a6000_pcie_k100_pcie_fix.csv"
    pcie_tput = 0.0
    try:
        rows = list(csv.DictReader(open(pcie_baseline)))
        baseline_row = next((r for r in rows if r.get("policy") == "lru"), None)
        if baseline_row:
            pcie_tput = float(baseline_row.get("tput_tok_s", 0))
    except FileNotFoundError:
        pass

    apis_gain = (result["tput_tok_s"] / pcie_tput) if pcie_tput > 0 else 0.0
    result["apis_gain_vs_tp2_pcie"] = round(apis_gain, 4)
    result["pcie_baseline_tput"] = pcie_tput

    summary = {"experiment": "E13.10", "hardware": args.hardware_label, **result}
    _write_summary(out_dir / "e13_10_summary.json", summary)
    _print_ec(summary, {
        "EC 13.10 (apis_gain ≥ 1.80× vs TP=2 PCIe)":
            lambda s: s.get("apis_gain_vs_tp2_pcie", 0) >= 1.80,
    })
    return summary


def run_e13_11(args: argparse.Namespace) -> Dict:
    """E13.11 -- Headline 1.5× compound result.

    Full compound stack on n=1000 prompts. This is the primary paper result.
    EC 13.11: headline_gain ≥ 1.45×; WAR(wgkp) ≥ 0.65; p99_TTFT ≤ SLO.
    """
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = str(out_dir / "e13_11_metrics.jsonl")

    tmax_ms_11 = "90.0" if args.tp_size == 1 else "300.0"
    extra_env = {
        "AS_MODE": "wgkp",
        "AS_WAR_TARGET": "0.80",
        "AS_TMAX_MS": tmax_ms_11,
        "AS_LOG_WAR": "1",
        "AS_WGKP_THRESHOLD": str(args.wgkp_threshold),
        "AS_WGKP_APT": "1",
        "AS_MWC_K_HOT": str(args.mwc_k_hot),
        "AS_MWC_MEMORY_GB": "22.0",
        "AS_FUSED_KERNEL": "1" if args.fused_kernel else "0",
        "AS_MACRO_N_ACCUM": str(args.macro_n_accum),
        "AS_WHITTLE_DELTA_T": "0.030" if args.tp_size == 1 else "0.100",
        "AS_WGKP_LOG": str(out_dir / "e13_11_wgkp.jsonl"),
        "AS_METRICS_PATH": metrics_path,
        "AS_TTFT_SLO_MS": "2000.0",
    }
    log_path = str(out_dir / "e13_11_server.log")
    cmd, env = _build_serve_cmd(args, extra_env, log_path)
    proc = _launch_server(cmd, env, args.port, log_path)

    try:
        result = _run_benchmark(
            port=args.port,
            dataset_path=args.dataset_path,
            n_prompts=args.n_prompts,
            lambda_total=args.lambda_total,
            K=args.K,
            output_json=str(out_dir / "e13_11_benchmark.json"),
            extra_label="wgkp_headline",
            model=args.model,
        )
    finally:
        _stop_server(proc, args.port)

    metrics = _parse_metrics_jsonl(metrics_path)
    result.update(metrics)

    # Compare to E13.8 baseline
    baseline_json = pathlib.Path(args.output_dir).parent / "e13_8" / "e13_8_summary.json"
    baseline_tput = 0.0
    try:
        with open(baseline_json) as f:
            baseline_tput = float(json.load(f).get("tput_tok_s", 0))
    except FileNotFoundError:
        pass

    headline_gain = (result["tput_tok_s"] / baseline_tput) if baseline_tput > 0 else 0.0
    result["headline_gain"] = round(headline_gain, 4)
    result["baseline_tput_tok_s"] = baseline_tput
    slo_ms = 2000.0
    result["slo_met"] = result.get("p99_ttft_ms", 0) <= slo_ms

    summary = {
        "experiment": "E13.11",
        "hardware": args.hardware_label,
        "K": args.K,
        "lambda_total": args.lambda_total,
        **result,
    }
    _write_summary(out_dir / "e13_11_summary.json", summary)
    _print_ec(summary, {
        "EC 13.11a (headline_gain ≥ 1.45)": lambda s: s.get("headline_gain", 0) >= 1.45,
        "EC 13.11b (mean_WAR ≥ 0.65)":      lambda s: s.get("mean_war", 0) >= 0.65,
        "EC 13.11c (p99_TTFT ≤ 2000ms)":    lambda s: s.get("slo_met", False),
    })
    return summary


def run_e13_12(args: argparse.Namespace) -> Dict:
    """E13.12 -- WAR/GWAR timeline during sustained load.

    Runs WGKP mode and collects per-batch WAR, GWAR(n*), and promotion_fraction
    from the AS_WGKP_LOG JSONL. Writes timeline CSV for plotting.
    EC 13.12: GWAR(n*=16) ≥ 0.40; promotion_fraction ≥ 0.30.
    """
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = str(out_dir / "e13_12_metrics.jsonl")
    wgkp_log_path = str(out_dir / "e13_12_wgkp.jsonl")

    tmax_ms_12 = "90.0" if args.tp_size == 1 else "300.0"
    extra_env = {
        "AS_MODE": "wgkp",
        "AS_WAR_TARGET": "0.80",
        "AS_TMAX_MS": tmax_ms_12,
        "AS_LOG_WAR": "1",
        "AS_WGKP_THRESHOLD": str(args.wgkp_threshold),
        "AS_MWC_K_HOT": str(args.mwc_k_hot),
        "AS_WHITTLE_DELTA_T": "0.030" if args.tp_size == 1 else "0.100",
        "AS_WGKP_LOG": wgkp_log_path,
        "AS_METRICS_PATH": metrics_path,
    }
    log_path = str(out_dir / "e13_12_server.log")
    cmd, env = _build_serve_cmd(args, extra_env, log_path)
    proc = _launch_server(cmd, env, args.port, log_path)

    try:
        result = _run_benchmark(
            port=args.port,
            dataset_path=args.dataset_path,
            n_prompts=args.n_prompts,
            lambda_total=args.lambda_total,
            K=args.K,
            output_json=str(out_dir / "e13_12_benchmark.json"),
            extra_label="wgkp_timeline",
            model=args.model,
        )
    finally:
        _stop_server(proc, args.port)

    metrics = _parse_metrics_jsonl(metrics_path)
    result.update(metrics)

    # Write timeline CSV from WGKP log
    timeline_path = out_dir / "e13_12_timeline.csv"
    _write_timeline_csv(wgkp_log_path, str(timeline_path))

    summary = {
        "experiment": "E13.12",
        "hardware": args.hardware_label,
        "timeline_csv": str(timeline_path),
        **result,
    }
    _write_summary(out_dir / "e13_12_summary.json", summary)
    _print_ec(summary, {
        "EC 13.12a (mean_gwar ≥ 0.40)":            lambda s: s.get("mean_gwar", 0) >= 0.40,
        "EC 13.12b (promotion_fraction ≥ 0.30)":    lambda s: s.get("mean_promotion_fraction", 0) >= 0.30,
    })
    return summary


# Helper utilities

def _write_summary(path: pathlib.Path, summary: Dict) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → Summary: {path}")


def _print_ec(summary: Dict, conditions: Dict) -> None:
    print("\n  Exit Conditions:")
    for label, check_fn in conditions.items():
        passed = check_fn(summary)
        print(f"    {'PASS' if passed else 'FAIL'} | {label}")


def _write_timeline_csv(jsonl_path: str, csv_path: str) -> None:
    """Parse WGKP log JSONL and write timeline CSV."""
    rows = []
    try:
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                try:
                    rec = json.loads(line.strip())
                    rows.append({
                        "batch_idx": i,
                        "timestamp_s": rec.get("timestamp_s", 0),
                        "war": rec.get("war", 0),
                        "gwar": rec.get("gwar", 0),
                        "promotion_fraction": rec.get("promotion_fraction", 0),
                        "n_star": rec.get("n_star", 0),
                        "n_tokens": rec.get("n_tokens", 0),
                    })
                except (json.JSONDecodeError, ValueError):
                    continue
    except FileNotFoundError:
        pass

    if not rows:
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Timeline CSV: {csv_path} ({len(rows)} batches)")


# Main

def main():
    parser = argparse.ArgumentParser(description="E13.8–E13.12 WGKP ablation + headline eval")
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--model", default="./models/llama-7b")
    parser.add_argument("--adapter-dir", default="./adapters")
    parser.add_argument("--K", type=int, default=10, help="Number of adapters")
    parser.add_argument("--lambda-total", type=float, default=7.0,
                        help="Total request rate (req/s)")
    parser.add_argument("--hardware-label", default="a6000_tp1")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dataset-path", default="./data/sharegpt/sharegpt.jsonl")
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--output-dir", default="results/kernel_promotion/")
    parser.add_argument("--n-prompts", type=int, default=500)
    # WGKP-specific
    parser.add_argument("--wgkp-threshold", type=int, default=16)
    parser.add_argument("--wgkp-apt", action="store_true", help="Enable APT")
    parser.add_argument("--macro-n-accum", type=int, default=2)
    parser.add_argument("--mwc-k-hot", type=int, default=5)
    parser.add_argument("--fused-kernel", action="store_true", help="Enable fused Triton kernel")
    # APIS-specific
    parser.add_argument("--apis-n-gpus", type=int, default=2)
    parser.add_argument("--apis-upstream-urls", default="",
                        help="Comma-separated upstream server URLs for APIS")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"AdapterSlots kernel_promotion: {args.experiment.upper()}")
    print(f"hardware={args.hardware_label}  K={args.K}  λ={args.lambda_total}  TP={args.tp_size}")
    print(f"{'='*60}\n")

    runners = {
        "e13_8":  run_e13_8,
        "e13_9":  run_e13_9,
        "e13_10": run_e13_10,
        "e13_11": run_e13_11,
        "e13_12": run_e13_12,
    }
    runners[args.experiment](args)


if __name__ == "__main__":
    main()
