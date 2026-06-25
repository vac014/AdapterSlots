"""
war_improvement_serving_benchmark.py -- WAR improvement experiment for AlignmentBuffer serving.

Implements experiments §5.1 (Single A6000), §5.3 (Queue Growth), §5.5a/b (Two A6000 PCIe),
§5.6a/b (Two H100 NVLink) from alignment_buffer.md.

This script:
1. Starts a vLLM server with AlignmentAwareScheduler via a subprocess.
2. Sends requests using the vLLM benchmark_serving.py client.
3. Collects WAR metrics from the server's batch log.
4. Sweeps T_max ∈ {0, 1, 2, 5, 10, 20} ms and writes a summary CSV.

Prerequisites:
    - conda activate adapter_env
    - vLLM installed with --scheduler-class support (vLLM 0.6+)
    - Adapters generated (./adapters/adapter_r16_k{0..K-1}_s*)
    - Model downloaded (./models/llama-7b)

Usage:
    # Single A6000 WAR improvement sweep (§5.1):
    python scripts/experiments/war_improvement_serving_benchmark.py \
        --model ./models/llama-7b \
        --adapter-dir ./adapters \
        --K 4 \
        --tmax-values 0 1 2 5 10 20 \
        --request-rates 3 7 10 \
        --dataset-path ./data/sharegpt/sharegpt.jsonl \
        --output-dir results/alignment_buffer/a6000_single \
        --duration 300

    # Two A6000 PCIe TP=2 WAR transparency (§5.5a):
    python scripts/experiments/war_improvement_serving_benchmark.py \
        --model ./models/llama-7b \
        --adapter-dir ./adapters \
        --K 4 \
        --tensor-parallel-size 2 \
        --tmax-values 0 1 2 5 10 20 \
        --request-rates 7 \
        --dataset-path ./data/sharegpt/sharegpt.jsonl \
        --output-dir results/alignment_buffer/two_a6000_pcie \
        --label "Two A6000 PCIe (TP=2)" \
        --duration 300

    # Two H100 NVLink T_max recalibration -- single H100 (§5.6a):
    CUDA_VISIBLE_DEVICES=0 python scripts/experiments/war_improvement_serving_benchmark.py \
        --model ./models/llama-7b \
        --adapter-dir ./adapters \
        --K 4 \
        --tmax-values 0.5 1 2 5 10 20 \
        --request-rates 7 \
        --dataset-path ./data/sharegpt/sharegpt.jsonl \
        --output-dir results/alignment_buffer/h100_single \
        --label "Single H100 (Hopper)" \
        --duration 300
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Constants

BENCHMARK_SCRIPT = "benchmarks/upstream/benchmark_serving.py"
VLLM_SERVER_READY_TIMEOUT = 180  # seconds
SERVER_POLL_INTERVAL = 2         # seconds


def kill_port_holders(port: int) -> None:
    """Kill every process that has any TCP socket bound to *port* using psutil.

    Covers LISTEN, ESTABLISHED, CLOSE_WAIT, and all other active states.
    TIME_WAIT sockets have pid=None (kernel-owned) and are skipped.
    """
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
        time.sleep(1)  # give the kernel a moment to reclaim the socket


# Server management

def wait_for_server(
    port: int,
    proc: "subprocess.Popen",
    timeout: int = VLLM_SERVER_READY_TIMEOUT,
    log_path: Optional[str] = None,
) -> bool:
    """Poll /health until ready.  Returns False and prints diagnosis on failure."""
    import urllib.request
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Check for early process exit before hitting the timeout.
        rc = proc.poll()
        if rc is not None:
            print(f"  [ERROR] Server process exited early (returncode={rc})")
            _print_server_log_tail(log_path, lines=60)
            return False
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(SERVER_POLL_INTERVAL)
    print(f"  [ERROR] Server did not respond within {timeout}s "
          f"(process still running: {proc.poll() is None})")
    _print_server_log_tail(log_path, lines=60)
    return False


def _print_server_log_tail(log_path: Optional[str], lines: int = 60) -> None:
    """Print the last N lines of the server log file for diagnosis."""
    if not log_path:
        print("  [DIAG] No server log path available.")
        return
    try:
        with open(log_path, "r", errors="replace") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        print(f"  [DIAG] Last {len(tail)} lines of {log_path}:")
        print("  " + "-" * 60)
        for line in tail:
            print("  " + line, end="")
        print()
        print("  " + "-" * 60)
    except FileNotFoundError:
        print(f"  [DIAG] Server log not found: {log_path} "
              "(server may have crashed before writing anything)")


def build_lora_modules_args(adapter_dir: str, K: int) -> List[str]:
    """Build --lora-modules arguments for vLLM."""
    base = Path(adapter_dir)
    args = []
    k_found = 0
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if k_found >= K:
            break
        # Adapter directories are named adapter_r{rank}_k{idx}_s{seed}
        if d.name.startswith("adapter_") and "k" in d.name:
            # lora_name = adapter_id used in requests
            lora_name = f"adapter_{k_found}"
            args.extend([f"{lora_name}={str(d)}"])
            k_found += 1
    if k_found < K:
        print(f"[WARN] Found only {k_found} adapter directories, expected {K}")
    return args


def start_vllm_server(
    model: str,
    adapter_dir: str,
    K: int,
    tmax_ms: float,
    ttft_slo_ms: float = 200.0,
    tensor_parallel_size: int = 1,
    port: int = 8000,
    max_num_batched_tokens: int = 4096,
    gpu_memory_utilization: float = 0.88,
    batch_log_path: Optional[str] = None,
    stderr_log_path: Optional[str] = None,
    max_lora_rank: int = 16,
    served_model_name: str = "default_model",
) -> subprocess.Popen:
    """Start vLLM server with AlignmentAwareScheduler in a subprocess."""
    lora_modules = build_lora_modules_args(adapter_dir, K)
    if not lora_modules:
        raise RuntimeError(f"No adapters found in {adapter_dir}")

    env = os.environ.copy()
    env["AS_TMAX_MS"] = str(tmax_ms)
    env["AS_TTFT_SLO_MS"] = str(ttft_slo_ms)
    env["AS_MODE"] = "threshold"
    env["AS_LOG_WAR"] = "1"
    if batch_log_path:
        env["AS_METRICS_PATH"] = batch_log_path

    # vllm_serve_adapter_slots.py runs AlignmentAwareAsyncEngine (see
    # adapter_slots/integrations/aligned_engine.py) -- a real AsyncLLMEngine
    # subclass, not a runtime monkey-patch of vLLM internals.
    env["AS_SCHEDULER"] = "1"
    cmd = [
        sys.executable, "scripts/vllm_serve_adapter_slots.py",
        "--model", model,
        "--enable-lora",
        "--lora-modules", *lora_modules,
        "--max-loras", str(K),
        "--max-lora-rank", str(max_lora_rank),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--port", str(port),
        "--served-model-name", served_model_name,
        "--disable-frontend-multiprocessing",
    ]
    if tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    print(f"  Starting vLLM (T_max={tmax_ms}ms, TP={tensor_parallel_size})...")
    # Evict any leftover process from a prior run before binding the port.
    kill_port_holders(port)

    # Redirect stderr to a file so we can parse WAR logs later without filling
    # the pipe buffer (vLLM is very verbose during model loading -- a 64 KB pipe
    # would block the subprocess before the /health endpoint comes up).
    # Redirect both stdout AND stderr to the log file.
    # vLLM's logger.info (including AdapterSlots WAR tick lines) goes to stdout;
    # our explicit print(..., file=sys.stderr) calls also need capture.
    # Using the same file handle for both avoids interleaving issues.
    stderr_fh = (open(stderr_log_path, "wb")  # noqa: WPS515
                 if stderr_log_path else None)
    # start_new_session=True puts the server and all its CUDA worker children
    # into a new process group so stop_server() can kill the whole tree.
    proc = subprocess.Popen(cmd, env=env, stdout=stderr_fh, stderr=stderr_fh,
                            start_new_session=True)
    proc._stderr_fh = stderr_fh  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen, port: Optional[int] = None) -> None:
    """Kill the vLLM server and ALL its descendant processes.

    vLLM forks GPU worker processes via multiprocessing.fork; those workers
    inherit the listening socket FD and hold it open after the main process
    dies.  Strategy:
      1. SIGTERM the whole process group (graceful shutdown).
      2. Wait for the main process.
      3. SIGKILL any process group survivors.
      4. If a port is given, psutil-kill any remaining holder of that socket.
    """
    import os

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None

    # Step 1 & 2: graceful SIGTERM
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

    # Step 3: SIGKILL any survivors in the process group
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # group already gone
    if proc.poll() is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    # Step 4: psutil-kill any process still holding the port
    if port is not None:
        kill_port_holders(port)

    fh = getattr(proc, "_stderr_fh", None)
    if fh is not None:
        fh.close()


def wait_for_port_free(port: int, timeout: int = 30) -> None:
    """Block until the TCP port is no longer bound, or timeout expires."""
    import socket
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return  # port is free
            except OSError:
                time.sleep(1)
    # Last resort: psutil-kill whatever is still holding the port.
    print(f"  [WARN] Port {port} still held after {timeout}s -- force-killing holder")
    kill_port_holders(port)
    time.sleep(2)


# Benchmark runner

def run_benchmark(
    model: str,
    dataset_path: str,
    request_rate: float,
    num_prompts: int,
    port: int = 8000,
    result_path: Optional[str] = None,
    served_model_name: str = "default_model",
    adapter_names: Optional[List[str]] = None,
) -> Optional[dict]:
    """Send num_prompts requests to vLLM, cycling through adapter_names.

    Replaces the benchmark_serving.py subprocess so that:
      1. Requests are tagged with LoRA adapter names (required for WAR measurement).
      2. No tokenizer path resolution or HuggingFace lookup is needed.
    """
    import threading
    import statistics as _stats
    import urllib.request as _urllib

    # Load ShareGPT prompts
    prompts: List[tuple] = []   # (prompt_text, max_tokens)
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
                # Support both a single dict per line and a JSON array per line.
                items = item if isinstance(item, list) else [item]
                for entry in items:
                    convs = entry.get("conversations", [])
                    if len(convs) >= 2:
                        human_text = convs[0].get("value", "")
                        gpt_text   = convs[1].get("value", "")
                        est_prompt_tok = len(human_text) // 4
                        est_out_tok = max(16, int(len(gpt_text.split()) * 1.3))
                        max_tok = min(est_out_tok, max(16, 1900 - est_prompt_tok))
                        prompts.append((human_text, max_tok))
                    if len(prompts) >= num_prompts:
                        break
                if len(prompts) >= num_prompts:
                    break
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return None

    if not prompts:
        print("[WARN] No prompts loaded from dataset")
        return None
    if len(prompts) < num_prompts:
        print(f"[WARN] Only {len(prompts)} prompts loaded (requested {num_prompts})")

    # Cycle adapters: adapter_0, adapter_1, … round-robin across requests.
    if not adapter_names:
        adapter_names = [served_model_name]

    # Threaded request sender
    url   = f"http://localhost:{port}/v1/completions"
    _lock = threading.Lock()
    ttft_ms_list:   List[float] = []
    output_tok_sum: List[int]   = [0]
    error_count:    List[int]   = [0]

    def _send(prompt: str, max_tokens: int, model_name: str) -> None:
        payload = json.dumps({
            "model":       model_name,
            "prompt":      prompt,
            "max_tokens":  max_tokens,
            "temperature": 0.0,
            "stream":      False,
        }).encode()
        t0 = time.perf_counter()
        try:
            req = _urllib.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _urllib.urlopen(req, timeout=120) as resp:
                body = resp.read()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            try:
                data  = json.loads(body)
                n_out = (data.get("usage") or {}).get("completion_tokens", max_tokens)
            except Exception:
                n_out = max_tokens
            with _lock:
                ttft_ms_list.append(elapsed_ms)
                output_tok_sum[0] += n_out
        except Exception:
            with _lock:
                error_count[0] += 1

    # Rate-controlled dispatch
    interval = 1.0 / request_rate
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

    # Aggregate metrics
    if not ttft_ms_list:
        print(f"[WARN] Benchmark produced 0 results ({error_count[0]} errors)")
        return None

    ttft_sorted = sorted(ttft_ms_list)
    n = len(ttft_sorted)
    result = {
        "output_throughput": output_tok_sum[0] / max(elapsed_total, 1e-6),
        "mean_ttft_ms":      _stats.mean(ttft_ms_list),
        "p99_ttft_ms":       ttft_sorted[min(int(0.99 * n), n - 1)],
        "num_prompts":       n,
        "errors":            error_count[0],
        "elapsed_s":         round(elapsed_total, 2),
    }

    if result_path:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as fh:
            json.dump(result, fh, indent=2)

    return result


# Batch-log parsing

def parse_batch_log(log_path: str, warmup_s: float = 30.0, cooldown_s: float = 30.0):
    """Parse the metrics JSONL written by AlignmentAwareScheduler._log_war_stats.

    Excludes the first warmup_s and last cooldown_s seconds of the run so that
    ramp-up and teardown batches (which have N < K×W and always give WAR=0) do
    not drag down the reported steady-state WAR.

    Returns:
        war_values     – list[float] of per-batch WAR (steady-state window only)
        adapter_counts – dict[str, int] mapping adapter_id → dispatched token count
        war_values_all – list[float] of ALL per-batch WAR (for diagnostics)
    """
    all_records: list = []
    adapter_counts: dict = {}
    try:
        with open(log_path) as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                all_records.append(rec)
                for tok in rec.get("tokens", []):
                    aid = tok.get("adapter_id", "unknown")
                    adapter_counts[aid] = adapter_counts.get(aid, 0) + 1
    except FileNotFoundError:
        return [], {}, []

    if not all_records:
        return [], adapter_counts, []

    # dispatch_time_ms is ms since server start; trim first/last windows.
    t_start = all_records[0].get("dispatch_time_ms", 0.0)
    t_end   = all_records[-1].get("dispatch_time_ms", 0.0)
    t_lo = t_start + warmup_s * 1000.0
    t_hi = t_end   - cooldown_s * 1000.0

    war_values_all = [float(r["war"]) for r in all_records if "war" in r]
    war_values = [
        float(r["war"]) for r in all_records
        if "war" in r and t_lo <= r.get("dispatch_time_ms", 0.0) <= t_hi
    ]
    # Fall back to all records if the window filtered everything out
    # (e.g. very short runs or warmup > run duration).
    if not war_values:
        war_values = war_values_all

    return war_values, adapter_counts, war_values_all


# Main experiment loop

def run_experiment(
    model: str,
    adapter_dir: str,
    dataset_path: str,
    K: int,
    tmax_values_ms: List[float],
    request_rates: List[float],
    output_dir: str,
    tensor_parallel_size: int = 1,
    duration_s: int = 300,
    label: str = "",
    port: int = 8000,
    max_lora_rank: int = 16,
    num_prompts: int = 1000,
    gpu_memory_utilization: float = 0.88,
) -> None:
    """Run the full T_max × request_rate sweep for alignment_buffer §5.1/§5.5a/§5.6a/§5.6b."""
    os.makedirs(output_dir, exist_ok=True)

    results = []
    print(f"\n=== alignment_buffer WAR Improvement Sweep  {label} ===")
    print(f"    Model: {model}  K={K}  TP={tensor_parallel_size}")
    print(f"    T_max values: {tmax_values_ms} ms")
    print(f"    Request rates: {request_rates} req/s")
    print(f"    Output: {output_dir}")

    for tmax_ms in tmax_values_ms:
        for rate in request_rates:
            run_id = f"tmax{tmax_ms:.1f}_rate{rate:.0f}"
            result_path = os.path.join(output_dir, f"bm_{run_id}.json")
            batch_log = os.path.join(output_dir, f"batch_log_{run_id}.jsonl")
            server_log = os.path.join(output_dir, f"server_log_{run_id}.txt")

            print(f"\n  --- T_max={tmax_ms}ms  rate={rate}req/s ---")

            served_model_name = Path(model).name  # e.g. "llama-7b"
            adapter_names = [f"adapter_{k}" for k in range(K)]

            proc = start_vllm_server(
                model=model,
                adapter_dir=adapter_dir,
                K=K,
                tmax_ms=tmax_ms,
                tensor_parallel_size=tensor_parallel_size,
                port=port,
                batch_log_path=batch_log,
                stderr_log_path=server_log,
                max_lora_rank=max_lora_rank,
                served_model_name=served_model_name,
                gpu_memory_utilization=gpu_memory_utilization,
            )

            server_ready = wait_for_server(port, proc=proc, log_path=server_log)
            if not server_ready:
                stop_server(proc, port=port)
                wait_for_port_free(port)
                continue

            print(f"  Server ready. Running benchmark ({num_prompts} prompts)...")
            bm = run_benchmark(
                model=model,
                dataset_path=dataset_path,
                request_rate=rate,
                num_prompts=num_prompts,
                port=port,
                result_path=result_path,
                served_model_name=served_model_name,
                adapter_names=adapter_names,
            )
            stop_server(proc, port=port)
            wait_for_port_free(port)

            # Parse WAR from batch-log; steady-state window trims ramp-up/teardown.
            war_values, adapter_counts, war_all = parse_batch_log(batch_log)

            import statistics
            war_mean = statistics.mean(war_values) if war_values else float("nan")
            war_p10 = (sorted(war_values)[int(0.1 * len(war_values))]
                       if len(war_values) >= 10 else float("nan"))
            war_p90 = (sorted(war_values)[int(0.9 * len(war_values))]
                       if len(war_values) >= 10 else float("nan"))
            war_raw  = statistics.mean(war_all) if war_all else float("nan")

            throughput = bm.get("output_throughput", float("nan")) if bm else float("nan")
            ttft_p50 = bm.get("mean_ttft_ms", float("nan")) if bm else float("nan")
            ttft_p99 = bm.get("p99_ttft_ms", float("nan")) if bm else float("nan")

            row = {
                "tmax_ms": tmax_ms,
                "request_rate": rate,
                "K": K,
                "tensor_parallel_size": tensor_parallel_size,
                "war_mean": war_mean,
                "war_p10": war_p10,
                "war_p90": war_p90,
                "ttft_p50_ms": ttft_p50,
                "ttft_p99_ms": ttft_p99,
                "throughput_tok_s": throughput,
                "duration_s": duration_s,
                "label": label,
                "adapter_counts_json": json.dumps(adapter_counts),
                "war_raw_mean": war_raw,
            }
            results.append(row)
            print(f"    WAR={war_mean:.3f} (raw={war_raw:.3f})  throughput={throughput:.1f} tok/s  "
                  f"TTFT_P50={ttft_p50:.1f}ms  TTFT_P99={ttft_p99:.1f}ms")

    # Write WAR improvement CSV
    output_csv = os.path.join(output_dir, "war_improvement.csv")
    fieldnames = ["tmax_ms", "request_rate", "K", "tensor_parallel_size",
                  "war_mean", "war_raw_mean", "war_p10", "war_p90",
                  "ttft_p50_ms", "ttft_p99_ms",
                  "throughput_tok_s", "duration_s", "label", "adapter_counts_json"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to: {output_csv}")


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="alignment_buffer WAR improvement and throughput sweep experiment"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model path (e.g. ./models/llama-7b)")
    parser.add_argument("--adapter-dir", type=str, default="./adapters",
                        help="Adapter directory")
    parser.add_argument("--dataset-path", type=str,
                        default="./data/sharegpt/sharegpt.jsonl",
                        help="ShareGPT dataset path")
    parser.add_argument("--K", type=int, default=4,
                        help="Number of concurrent adapters (default: 4)")
    parser.add_argument("--tmax-values", type=float, nargs="+",
                        default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
                        help="T_max values in ms to sweep")
    parser.add_argument("--request-rates", type=float, nargs="+",
                        default=[3.0, 7.0, 10.0],
                        help="Request rates in req/s")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="TP degree (1=single GPU, 2=TP=2)")
    parser.add_argument("--output-dir", type=str,
                        default="results/alignment_buffer/a6000_single",
                        help="Output directory for CSVs")
    parser.add_argument("--duration", type=int, default=300,
                        help="Duration per run in seconds (used for rate estimation)")
    parser.add_argument("--label", type=str, default="",
                        help="Hardware label (e.g. 'Two A6000 PCIe')")
    parser.add_argument("--port", type=int, default=8000,
                        help="vLLM server port (default: 8000)")
    parser.add_argument("--max-lora-rank", type=int, default=16,
                        help="Maximum LoRA rank (default: 16)")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts per benchmark run (default: 1000)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88,
                        help="vLLM gpu-memory-utilization (default: 0.88; use 0.82 for K=16 TP=2)")
    args = parser.parse_args()

    run_experiment(
        model=args.model,
        adapter_dir=args.adapter_dir,
        dataset_path=args.dataset_path,
        K=args.K,
        tmax_values_ms=args.tmax_values,
        request_rates=args.request_rates,
        output_dir=args.output_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        duration_s=args.duration,
        label=args.label,
        port=args.port,
        max_lora_rank=args.max_lora_rank,
        num_prompts=args.num_prompts,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
