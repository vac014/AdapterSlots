"""
benchmark_e1.py -- E1 throughput benchmark for the isolation experiment.

Runs all four Bradford Hill conditions (A/B/C/D) or individual conditions,
measures SGMV kernel time (ms) across n_runs, and produces a CSV with
per-run rows (one row per run, column: tok_s) so that benchmark_e1_scale.py
can compute statistics and p-values from raw data.

Usage (CPU-only decomposition timing mode -- no GPU required):
    python benchmarks/isolation/benchmark_e1.py \
        --n-tokens 512 --K 2 --n-runs 100 --warmup 10 \
        --output results/e1/benchmark_e1_cpu.csv

Usage (GPU -- Punica SGMV micro-benchmark, correct E1 path):
    python benchmarks/isolation/benchmark_e1.py \
        --model ./models/llama-7b \
        --condition all \
        --n-tokens 512 --K 2 --n-runs 100 --warmup 10 \
        --output results/e1/throughput_a6000.csv

Usage (single condition, for ncu attach):
    python benchmarks/isolation/benchmark_e1.py --condition D --model ./models/llama-7b \
        --n-runs 50 --output results/e1/throughput_condition_D.csv

Usage (TP=2, Two A6000 PCIe or Two H100 NVLink -- set CUDA_VISIBLE_DEVICES=0,1 first):
    export CUDA_VISIBLE_DEVICES=0,1
    python benchmarks/isolation/benchmark_e1.py \
        --model ./models/llama-7b \
        --condition A \
        --n-tokens 512 --K 2 --n-runs 100 --warmup 10 \
        --tensor-parallel-size 2 \
        --output results/e1/tp2_a6000_pcie/throughput_condition_A_tp2_a6000_pcie.csv

Output CSV columns (per-run rows):
    run, condition, N, K, WAR, elapsed_ms, tok_s
"""

import argparse
import csv
import os
import subprocess
import time
from collections import Counter

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def parse_args():
    p = argparse.ArgumentParser(description="E1 throughput benchmark (Bradford Hill)")
    p.add_argument("--condition", type=str, choices=["A", "B", "C", "D", "all"],
                   default="all",
                   help="Which condition(s) to run. 'all' runs A then B then C then D.")
    p.add_argument("--model", type=str, default=None,
                   help="Path to base model. Config is read for hidden_size/num_layers. "
                        "If None, runs CPU-only decomp timing.")
    p.add_argument("--adapter-dir", type=str, default="./adapters",
                   help="Unused in GPU path; kept for CLI compatibility.")
    p.add_argument("--n-tokens", type=int, default=512,
                   help="Total tokens in the batch (N)")
    p.add_argument("--K", type=int, default=2, help="Number of adapters")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--prompt-len", type=int, default=128,
                   help="Unused in GPU SGMV path; kept for CLI compatibility.")
    p.add_argument("--n-runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flashinfer", action="store_true",
                   help="Enable FlashInfer backend (isolation_experiment cross-condition)")
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="Tensor-parallel degree. 1=single GPU (default). "
                        "2=TP=2: spawns 2 worker processes, each runs SGMV on its GPU "
                        "with full weights, then all-reduces the output to capture "
                        "PCIe/NVLink communication overhead. Requires "
                        "CUDA_VISIBLE_DEVICES=0,1 set before invocation.")
    p.add_argument("--output", type=str, default=None,
                   help="CSV output path (per-run rows with tok_s column)")
    p.add_argument("--skip-statistical-test", action="store_true",
                   help="Skip paired t-test (faster, for quick checks)")
    p.add_argument("--lock-clocks", action="store_true",
                   help="Lock GPU graphics clocks before benchmarking to prevent "
                        "mid-run frequency steps (recommended for TP>1 / PCIe setups). "
                        "Calls nvidia-smi -lgc; requires sufficient permissions.")
    p.add_argument("--clock-freq", type=int, default=None,
                   help="Graphics clock frequency in MHz to lock to. "
                        "Default: auto-query the GPU max supported clock.")
    return p.parse_args()


# GPU clock locking

def _gpu_ids_from_env(tp: int) -> list:
    """Return physical GPU indices to lock based on CUDA_VISIBLE_DEVICES and TP degree."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        ids = [int(x) for x in cvd.split(",") if x.strip().lstrip("-").isdigit()]
        return ids[:tp] if tp > 1 else ids[:1]
    return list(range(tp))


def _query_max_clock(gpu_id: int) -> int:
    """Query the GPU's max supported graphics clock (MHz). Falls back to 1800."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.max.gr",
         "--format=csv,noheader,nounits", "-i", str(gpu_id)],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        try:
            return int(r.stdout.strip())
        except ValueError:
            pass
    return 1800


def _lock_clocks(gpu_ids: list, freq: int) -> bool:
    """Lock graphics clocks on all specified GPUs. Returns True if all succeeded."""
    ok = True
    for gid in gpu_ids:
        r = subprocess.run(
            ["nvidia-smi", "-lgc", str(freq), "-i", str(gid)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  [clock-lock] WARNING: GPU {gid} lock failed "
                  f"(need sudo or nvperfmon group): {r.stderr.strip()[:120]}")
            ok = False
        else:
            print(f"  [clock-lock] GPU {gid} locked to {freq} MHz")
    return ok


def _unlock_clocks(gpu_ids: list) -> None:
    """Reset graphics clocks on all specified GPUs to driver-managed defaults."""
    for gid in gpu_ids:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(gid)],
                       capture_output=True, text=True)
        print(f"  [clock-lock] GPU {gid} clocks reset to default")


# Batch construction

def build_adapter_ids(condition: str, N: int, K: int) -> list:
    if condition == "A":
        return [0] * N
    elif condition == "B":
        half = N // 2
        return [0] * half + [1] * (N - half)
    elif condition == "C":
        block = 16
        ids = []
        adapter = 0
        while len(ids) < N:
            ids.extend([adapter] * min(block, N - len(ids)))
            adapter = 1 - adapter
        return ids[:N]
    elif condition == "D":
        return [i % K for i in range(N)]
    else:
        raise ValueError(f"Unknown condition: {condition}")


def compute_war(adapter_ids: list, warp_size: int = 32) -> float:
    arr = np.array(adapter_ids, dtype=np.int32)
    n = len(arr)
    m = n // warp_size
    if m == 0:
        return 0.0
    warps = arr[: m * warp_size].reshape(m, warp_size)
    return float(np.mean(warps.min(axis=1) == warps.max(axis=1)))


# SGMV segment builder

def build_sgmv_segments(adapter_ids: list):
    """
    Build Punica SGMV segment indptr from a flat adapter_ids list.

    Returns:
        appearing : sorted list of adapter IDs present in the batch
        s_list    : [0, count_0, count_0+count_1, ...] (length S+1)
    """
    counts = Counter(adapter_ids)
    appearing = sorted(counts.keys())
    s = [0]
    for aid in appearing:
        s.append(s[-1] + counts[aid])
    return appearing, s


# Timing

def time_cpu_decomposition(adapter_ids: list, K: int, n_runs: int) -> list:
    """CPU-only: time the O(N) decomposition loop. Returns list of ms values."""
    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        segments: dict = {k: [] for k in range(K)}
        for i, aid in enumerate(adapter_ids):
            segments[aid].append(i)
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1e6)
    return times_ms


def time_sgmv_forward(lora_weights, appearing, s_indptr,
                      N, hidden_size, lora_rank, num_layers,
                      device, dtype, n_runs, warmup) -> list:
    """
    GPU: time Punica SGMV kernel via add_lora_sgmv_custom_cutlass.
    Returns list of elapsed_ms values (one per run, measured with CUDA events).
    """
    import torch
    from punica.ops import add_lora_sgmv_custom_cutlass
    from punica.utils import BatchedLoraWeight

    batched = BatchedLoraWeight([lora_weights[aid] for aid in appearing])
    x = torch.randn(N, hidden_size, dtype=dtype, device=device)
    y = torch.zeros(N, hidden_size, dtype=dtype, device=device)

    for _ in range(warmup):
        y.zero_()
        for layer_idx in range(num_layers):
            add_lora_sgmv_custom_cutlass(
                y, x, batched.wa_ptr, batched.wb_ptr,
                s_indptr, layer_idx, lora_rank,
            )
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_runs):
        y.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for layer_idx in range(num_layers):
            add_lora_sgmv_custom_cutlass(
                y, x, batched.wa_ptr, batched.wb_ptr,
                s_indptr, layer_idx, lora_rank,
            )
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    return times_ms


def _tp_worker(rank: int, world_size: int, tp_args: dict, queue) -> None:
    """
    Worker spawned by torch.multiprocessing.spawn for TP=N runs.

    Each rank binds to cuda:<rank>, initialises an NCCL process group,
    runs SGMV with full weights, and all-reduces the output to model
    the TP communication overhead (PCIe PHB on Two A6000 or NVLink 4.0
    on Two H100).  Rank 0 puts the per-run elapsed_ms list into *queue*
    so the main process can aggregate and write the CSV.

    This is a data-parallel simulation of column-parallel TP:
      - Each rank independently runs SGMV on all N tokens (same work per rank
        as single-GPU, but split across 2 physical devices)
      - all_reduce captures the real cross-GPU communication cost
      - The A→D throughput gap is preserved because both ranks run the same
        adapter-mixing pattern

    Communication volume: N × hidden_size × 2 bytes (fp16) = same as a
    column-parallel all-reduce in a real TP=2 Transformer decode step.
    """
    import torch
    import torch.distributed as dist
    from punica.ops import add_lora_sgmv_custom_cutlass
    from punica.utils import BatchedLoraWeight, LoraWeight

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(tp_args["master_port"]))

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    N = tp_args["N"]
    K = tp_args["K"]
    condition = tp_args["condition"]
    hidden_size = tp_args["hidden_size"]
    num_layers = tp_args["num_layers"]
    lora_rank = tp_args["lora_rank"]
    n_runs = tp_args["n_runs"]
    warmup = tp_args["warmup"]
    dtype = torch.float16

    adapter_ids = build_adapter_ids(condition, N, K)
    appearing, s_list = build_sgmv_segments(adapter_ids)
    s_indptr = torch.tensor(s_list, dtype=torch.int32, device=device)

    lora_weights = [
        LoraWeight(num_layers, hidden_size, hidden_size, lora_rank, dtype, device)
        for _ in range(K)
    ]
    batched = BatchedLoraWeight([lora_weights[aid] for aid in appearing])
    x = torch.randn(N, hidden_size, dtype=dtype, device=device)
    y = torch.zeros(N, hidden_size, dtype=dtype, device=device)

    # Warmup: SGMV + all_reduce
    for _ in range(warmup):
        y.zero_()
        for layer_idx in range(num_layers):
            add_lora_sgmv_custom_cutlass(
                y, x, batched.wa_ptr, batched.wb_ptr,
                s_indptr, layer_idx, lora_rank,
            )
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_runs):
        y.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for layer_idx in range(num_layers):
            add_lora_sgmv_custom_cutlass(
                y, x, batched.wa_ptr, batched.wb_ptr,
                s_indptr, layer_idx, lora_rank,
            )
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    if rank == 0:
        queue.put(times_ms)

    dist.destroy_process_group()


def time_sgmv_tp(condition: str, N: int, K: int, hidden_size: int,
                 num_layers: int, lora_rank: int, n_runs: int, warmup: int,
                 world_size: int) -> list:
    """
    Launch *world_size* worker processes via torch.multiprocessing.spawn.
    Returns the per-run elapsed_ms list from rank 0.
    """
    import torch.multiprocessing as mp

    # Pick an available port to avoid collisions when running multiple sweeps.
    import socket
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]

    tp_args = dict(
        N=N, K=K, condition=condition,
        hidden_size=hidden_size, num_layers=num_layers, lora_rank=lora_rank,
        n_runs=n_runs, warmup=warmup,
        master_port=master_port,
    )

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(_tp_worker, args=(world_size, tp_args, queue),
             nprocs=world_size, join=True)
    return queue.get()


# Statistical helpers

def print_summary(condition: str, times_ms: list, N: int) -> None:
    arr = np.array(times_ms)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    p50_ms = float(np.percentile(arr, 50))
    p99_ms = float(np.percentile(arr, 99))
    throughput = N / (mean_ms / 1000.0) if mean_ms > 0 else 0.0
    print(f"  Cond {condition}:  mean={mean_ms:.2f}ms  std={std_ms:.2f}ms  "
          f"p50={p50_ms:.2f}ms  p99={p99_ms:.2f}ms  "
          f"throughput={throughput:.0f} tok/s")


def paired_ttest(times_a: list, times_d: list) -> dict:
    """Paired t-test: H0 = condition A and D have equal means."""
    if not HAS_SCIPY:
        return {"t_stat": "N/A", "p_value": "N/A", "significant_p001": "N/A"}

    a = np.array(times_a)
    d = np.array(times_d)
    n = min(len(a), len(d))
    t_stat, p_val = scipy_stats.ttest_rel(a[:n], d[:n])
    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_val), 6),
        "significant_p001": p_val < 0.01,
    }


# Per-condition runner

def run_condition(condition: str, N: int, K: int, n_runs: int, warmup: int,
                  sgmv_ctx=None, tp: int = 1) -> list:
    """
    Run one condition and return a list of per-run elapsed_ms values.

    sgmv_ctx: dict with keys lora_weights, hidden_size, num_layers, lora_rank,
              device, dtype -- required for the single-GPU GPU path (tp=1).
              When tp > 1, sgmv_ctx is ignored and time_sgmv_tp() is used.
    """
    import torch

    adapter_ids = build_adapter_ids(condition, N, K)
    war = compute_war(adapter_ids)
    print(f"  Condition {condition}: WAR={war:.4f}", end="")

    if tp > 1:
        # TP=2 path: multiprocessing workers with NCCL all-reduce.
        assert sgmv_ctx is not None, "--model required for TP>1 GPU path"
        print(f"  [TP={tp} SGMV]  launching {tp} workers ...", end="", flush=True)
        times_ms = time_sgmv_tp(
            condition, N, K,
            sgmv_ctx["hidden_size"], sgmv_ctx["num_layers"],
            sgmv_ctx["lora_rank"],
            n_runs, warmup, world_size=tp,
        )
        print(f"  mean={np.mean(times_ms):.2f}ms  "
              f"throughput={N / (np.mean(times_ms) / 1000):.0f} tok/s")
    elif sgmv_ctx is None:
        times_ms = time_cpu_decomposition(adapter_ids, K, n_runs)
        print(f"  [CPU decomp]  mean={np.mean(times_ms):.3f}ms")
    else:
        appearing, s_list = build_sgmv_segments(adapter_ids)
        s_indptr = torch.tensor(s_list, dtype=torch.int32,
                                device=sgmv_ctx["device"])
        times_ms = time_sgmv_forward(
            sgmv_ctx["lora_weights"], appearing, s_indptr,
            N, sgmv_ctx["hidden_size"], sgmv_ctx["lora_rank"],
            sgmv_ctx["num_layers"], sgmv_ctx["device"], sgmv_ctx["dtype"],
            n_runs, warmup,
        )
        print(f"  [GPU SGMV]  mean={np.mean(times_ms):.2f}ms  "
              f"throughput={N / (np.mean(times_ms) / 1000):.0f} tok/s")

    return times_ms


# Main

def main():
    args = parse_args()
    import torch
    torch.manual_seed(args.seed)

    tp = args.tensor_parallel_size
    conditions = (["A", "B", "C", "D"] if args.condition == "all"
                  else [args.condition])

    # GPU context: read config + allocate LoRA weights (single-GPU path)
    # For TP>1, the workers allocate their own weights inside _tp_worker.
    # We still read the model config here to get hidden_size / num_layers.
    sgmv_ctx = None
    if args.model:
        from transformers import AutoConfig

        print(f"Reading model config from {args.model} ...")
        cfg = AutoConfig.from_pretrained(args.model)
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        lora_rank = args.rank
        dtype = torch.float16

        print(f"  hidden_size={hidden_size}, num_layers={num_layers}, rank={lora_rank}")

        if tp == 1:
            from punica.utils import LoraWeight
            device = torch.device("cuda:0")
            print(f"Allocating {args.K} LoRA weight tensors on GPU ...")
            lora_weights = [
                LoraWeight(num_layers, hidden_size, hidden_size, lora_rank, dtype, device)
                for _ in range(args.K)
            ]
            sgmv_ctx = {
                "lora_weights": lora_weights,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "lora_rank": lora_rank,
                "device": device,
                "dtype": dtype,
            }
        else:
            # TP path: workers allocate on their own GPUs -- pass config only.
            sgmv_ctx = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "lora_rank": lora_rank,
            }

    tp_label = f" [TP={tp}]" if tp > 1 else ""
    print(f"\nE1 Benchmark{tp_label}: N={args.n_tokens} K={args.K} "
          f"runs={args.n_runs} warmup={args.warmup}")
    print("-" * 60)

    # Lock GPU clocks before any timing to prevent mid-benchmark frequency steps.
    lock_clocks = args.lock_clocks or (tp > 1)
    gpu_ids = _gpu_ids_from_env(tp) if args.model else []
    if lock_clocks and gpu_ids:
        freq = args.clock_freq or _query_max_clock(gpu_ids[0])
        print(f"\n[clock-lock] Locking GPUs {gpu_ids} to {freq} MHz ...")
        _lock_clocks(gpu_ids, freq)

    # all_times: {condition: [elapsed_ms, ...]}  -- used for t-test
    all_times: dict = {}
    # per_run_rows: written to CSV, one row per run
    per_run_rows = []

    try:
        for cond in conditions:
            times_ms = run_condition(
                condition=cond,
                N=args.n_tokens,
                K=args.K,
                n_runs=args.n_runs,
                warmup=args.warmup,
                sgmv_ctx=sgmv_ctx,
                tp=tp,
            )
            all_times[cond] = times_ms
            print_summary(cond, times_ms, args.n_tokens)

            adapter_ids = build_adapter_ids(cond, args.n_tokens, args.K)
            war = compute_war(adapter_ids)
            for run_i, ms in enumerate(times_ms):
                per_run_rows.append({
                    "run": run_i,
                    "condition": cond,
                    "N": args.n_tokens,
                    "K": args.K,
                    "WAR": round(war, 4),
                    "elapsed_ms": round(ms, 4),
                    "tok_s": round(args.n_tokens / (ms / 1000.0), 2) if ms > 0 else 0.0,
                    "tp": tp,
                })

        # Statistical test A vs D
        if "A" in all_times and "D" in all_times and not args.skip_statistical_test:
            print("\nStatistical test (A vs D):")
            ttest = paired_ttest(all_times["A"], all_times["D"])
            print(f"  t={ttest['t_stat']}  p={ttest['p_value']}  "
                  f"significant(p<0.01)={ttest['significant_p001']}")

        # Print condition summary table
        print(f"\n{'Cond':>5} {'WAR':>6} {'Mean tok/s':>12} {'Std tok/s':>11}")
        print("-" * 40)
        for cond in conditions:
            adapter_ids = build_adapter_ids(cond, args.n_tokens, args.K)
            war = compute_war(adapter_ids)
            toks = [r["tok_s"] for r in per_run_rows if r["condition"] == cond]
            if toks:
                print(f"  {cond:>4} {war:>6.3f} {np.mean(toks):>12.1f} {np.std(toks):>11.1f}")

        # Write per-run CSV (tok_s column for benchmark_e1_scale.py)
        if args.output and per_run_rows:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            fieldnames = ["run", "condition", "N", "K", "WAR",
                          "elapsed_ms", "tok_s", "tp"]
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction="ignore", restval="")
                writer.writeheader()
                writer.writerows(per_run_rows)
            print(f"\nPer-run results ({len(per_run_rows)} rows) -> {args.output}")

    finally:
        if lock_clocks and gpu_ids:
            print("\n[clock-lock] Releasing clock locks ...")
            _unlock_clocks(gpu_ids)


if __name__ == "__main__":
    main()
