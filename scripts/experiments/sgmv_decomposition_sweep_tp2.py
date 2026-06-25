"""
sgmv_decomposition_sweep_tp2.py -- E11 Decomposition Analysis: Multi-GPU TP=2 sweep.

Runs the E11 N×K decomposition timing experiment under TP=2 (tensor-parallel),
capturing:
  - Orchestrator-side batch sort time (Phase 1, runs once in the main process)
  - Per-GPU CTA construction time (Phase 2, runs independently in each worker)
  - GPU SGMV kernel + all-reduce time (Phases 3+4, NCCL all-reduce included)
  - Preprocessing fraction = (sort + cta) / total_latency

This script is the TP=2 counterpart of sgmv_decomposition_sweep.py (single-GPU).

Hardware targets:
  §4.5  -- Two RTX A6000 PCIe  (CUDA_VISIBLE_DEVICES=0,1, PHB topology ~32 GB/s)
  §4.6c -- Two H100 NVLink      (CUDA_VISIBLE_DEVICES=0,1, NVLink 4.0 ~900 GB/s)

Usage (Two A6000 PCIe -- §4.5a):
    export CUDA_VISIBLE_DEVICES=0,1
    python scripts/experiments/sgmv_decomposition_sweep_tp2.py \\
        --output-dir results/e11/two_a6000_pcie \\
        --model ./models/llama-7b \\
        --interconnect pcie \\
        --n-reps 1000

Usage (Two H100 NVLink -- §4.6c, run only after A6000 PCIe validated):
    export CUDA_VISIBLE_DEVICES=0,1
    python scripts/experiments/sgmv_decomposition_sweep_tp2.py \\
        --output-dir results/e11/two_h100_nvlink \\
        --model ./models/llama-7b \\
        --interconnect nvlink \\
        --N-values 64 128 256 512 1024 2048 4096 \\
        --K-values 2 4 8 16 32 \\
        --n-reps 1000

Output files (written to --output-dir):
    timing_sweep.csv           -- per-(N,K,sorted) row with orchestrator+per-GPU phases
    allreduce_masking.csv      -- allreduce_us and preprocessing_fraction per (N,K)
    tp2_decomp_breakdown.csv   -- orchestrator vs. per-GPU decomp time breakdown (§4.5c)
"""

import argparse
import csv
import os
import time
from collections import Counter

import numpy as np


# Grid defaults
# §4.5a -- Two A6000 PCIe: N up to 4096, K up to 16
DEFAULT_N_VALUES_PCIE = [64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_K_VALUES_PCIE = [2, 4, 8, 16]
# §4.6c -- Two H100 NVLink: K up to 32
DEFAULT_N_VALUES_NVLINK = [64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_K_VALUES_NVLINK = [2, 4, 8, 16, 32]

DEFAULT_N_REPS = 1000
DEFAULT_RANK = 16
DEFAULT_WARMUP = 20

_LLAMA7B_DEFAULTS = dict(hidden_size=4096, num_hidden_layers=32)


def parse_args():
    p = argparse.ArgumentParser(
        description="E11 TP=2 decomposition sweep -- orchestrator + per-GPU phases"
    )
    p.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write output CSVs"
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Path to base model dir (reads hidden_size/num_layers). "
             "If None, CPU-only decomposition timing."
    )
    p.add_argument(
        "--interconnect", type=str, choices=["pcie", "nvlink"],
        default="pcie",
        help="Interconnect type label written to output rows (pcie or nvlink)"
    )
    p.add_argument(
        "--N-values", type=int, nargs="+", default=None, metavar="N",
        help="Token-count values to sweep. Default: 64..4096 for both interconnects."
    )
    p.add_argument(
        "--K-values", type=int, nargs="+", default=None, metavar="K",
        help="Adapter-count values to sweep. Default: 2..16 (pcie) or 2..32 (nvlink)."
    )
    p.add_argument(
        "--n-reps", type=int, default=DEFAULT_N_REPS,
        help="Repetitions per (N,K,sorted) combination (default: 1000)"
    )
    p.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help="Warmup repetitions (default: 20)"
    )
    p.add_argument(
        "--rank", type=int, default=DEFAULT_RANK,
        help="LoRA rank for SGMV kernel (default: 16)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for unsorted batch construction"
    )
    p.add_argument(
        "--cpu-only", action="store_true",
        help="Measure CPU decomposition phases only; skip GPU kernel and all-reduce."
    )
    return p.parse_args()


# Batch builders

def build_unsorted_batch(N: int, K: int, rng: np.random.Generator) -> list:
    return rng.integers(0, K, size=N).tolist()


def build_sorted_batch(N: int, K: int) -> list:
    ids = []
    for k in range(K):
        start = k * N // K
        end = (k + 1) * N // K
        ids.extend([k] * (end - start))
    return ids


# CPU Phase timing helpers

def time_sort_us(adapter_ids: list, n_reps: int) -> tuple:
    arr = np.array(adapter_ids, dtype=np.int32)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter_ns()
        np.argsort(arr, kind="stable")
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    return float(np.mean(times)), float(np.std(times))


def time_cta_unsorted_us(adapter_ids: list, K: int, n_reps: int) -> tuple:
    """O(N) scan."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter_ns()
        segs: dict = {k: 0 for k in range(K)}
        for aid in adapter_ids:
            segs[aid] += 1
        indptr = [0]
        for k in range(K):
            indptr.append(indptr[-1] + segs[k])
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    return float(np.mean(times)), float(np.std(times))


def time_cta_sorted_us(adapter_ids: list, K: int, n_reps: int) -> tuple:
    """O(K) scan -- iterates the Python list directly to avoid numpy boxing overhead."""
    n = len(adapter_ids)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter_ns()
        boundaries = [0]
        cur = adapter_ids[0]
        for i in range(1, n):
            if adapter_ids[i] != cur:
                boundaries.append(i)
                cur = adapter_ids[i]
        boundaries.append(n)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    return float(np.mean(times)), float(np.std(times))


# GPU SGMV + all-reduce timing

def time_sgmv_allreduce_ms(
    batched, x_buf, y_buf, s_indptr,
    lora_rank, num_layers, dist_group
):
    """
    Time SGMV kernel + NCCL all-reduce (TP=2 path).

    Returns:
        (kernel_ms, allreduce_ms)
    """
    import torch
    from punica.ops import add_lora_sgmv_custom_cutlass

    y_buf.zero_()
    torch.cuda.synchronize()

    # SGMV kernel (all layers)
    start_kern = torch.cuda.Event(enable_timing=True)
    end_kern = torch.cuda.Event(enable_timing=True)
    start_kern.record()
    for layer_idx in range(num_layers):
        add_lora_sgmv_custom_cutlass(
            y_buf, x_buf,
            batched.wa_ptr, batched.wb_ptr,
            s_indptr, layer_idx, lora_rank,
        )
    end_kern.record()
    torch.cuda.synchronize()
    kernel_ms = start_kern.elapsed_time(end_kern)

    # All-reduce
    start_ar = torch.cuda.Event(enable_timing=True)
    end_ar = torch.cuda.Event(enable_timing=True)
    start_ar.record()
    import torch.distributed as tdist
    tdist.all_reduce(y_buf, op=tdist.ReduceOp.SUM, group=dist_group)
    end_ar.record()
    torch.cuda.synchronize()
    allreduce_ms = start_ar.elapsed_time(end_ar)

    return kernel_ms, allreduce_ms


def allocate_sgmv_buffers(adapter_ids, K, hidden_size, lora_rank, num_layers, device, dtype):
    import torch
    from punica.utils import LoraWeight, BatchedLoraWeight

    counts = Counter(adapter_ids)
    appearing = sorted(counts.keys())
    indptr = [0]
    for aid in appearing:
        indptr.append(indptr[-1] + counts[aid])
    s_indptr = torch.tensor(indptr, dtype=torch.int32, device=device)

    lora_ws = [
        LoraWeight(num_layers, hidden_size, hidden_size, lora_rank, dtype, device)
        for _ in range(K)
    ]
    batched = BatchedLoraWeight([lora_ws[aid] for aid in appearing])
    x_buf = torch.randn(len(adapter_ids), hidden_size, dtype=dtype, device=device)
    y_buf = torch.zeros(len(adapter_ids), hidden_size, dtype=dtype, device=device)
    return batched, x_buf, y_buf, s_indptr


# TP=2 worker function (spawned per GPU via torchrun / mp.spawn)

def _worker_main(local_rank, args, adapter_ids_map, result_dir):
    """
    Runs on each GPU worker. Measures per-GPU CTA construction and SGMV+allreduce.

    adapter_ids_map: dict[(N, K, is_sorted)] -> adapter_ids list
    result_dir: directory where this worker writes rank_{local_rank}.json
    """
    import json
    import torch
    import torch.distributed as dist

    # mp.spawn passes local_rank as an argument but does NOT set env vars;
    # env:// rendezvous requires RANK and LOCAL_RANK to be present.
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # NCCL init is only needed for the GPU all-reduce path; skip entirely for cpu-only.
    if not args.cpu_only:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = None

    dtype = torch.float16

    hidden_size = _LLAMA7B_DEFAULTS["hidden_size"]
    num_layers = _LLAMA7B_DEFAULTS["num_hidden_layers"]

    if args.model:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_hidden_layers

    group = dist.group.WORLD if not args.cpu_only else None
    results = {}

    for (N, K, is_sorted), adapter_ids in adapter_ids_map.items():
        # Per-GPU CTA construction (§4.5c -- each worker times its own CTA build)
        if is_sorted:
            cta_mean_us, cta_std_us = time_cta_sorted_us(adapter_ids, K, args.n_reps)
        else:
            cta_mean_us, cta_std_us = time_cta_unsorted_us(adapter_ids, K, args.n_reps)

        if args.cpu_only:
            results[(N, K, is_sorted, local_rank)] = {
                "cta_mean_us": cta_mean_us,
                "cta_std_us": cta_std_us,
                "kernel_mean_ms": "nan",
                "allreduce_mean_ms": "nan",
            }
            continue

        batched, x_buf, y_buf, s_indptr = allocate_sgmv_buffers(
            adapter_ids, K, hidden_size, args.rank, num_layers, device, dtype
        )
        torch.cuda.synchronize()

        # Warmup
        for _ in range(args.warmup):
            time_sgmv_allreduce_ms(
                batched, x_buf, y_buf, s_indptr, args.rank, num_layers, group
            )

        kern_list, ar_list = [], []
        for _ in range(args.n_reps):
            km, arm = time_sgmv_allreduce_ms(
                batched, x_buf, y_buf, s_indptr, args.rank, num_layers, group
            )
            kern_list.append(km)
            ar_list.append(arm)

        results[(N, K, is_sorted, local_rank)] = {
            "cta_mean_us": cta_mean_us,
            "cta_std_us": cta_std_us,
            "kernel_mean_ms": round(float(np.mean(kern_list)), 6),
            "allreduce_mean_ms": round(float(np.mean(ar_list)), 6),
        }

    # Write results to a file synchronously before process exit.
    # Using a file (not a queue) is safe under os._exit because the write is a
    # kernel-level syscall that completes before the CUDA driver teardown runs.
    serializable = {f"{N},{K},{int(s)},{r}": v for (N, K, s, r), v in results.items()}
    result_file = os.path.join(result_dir, f"rank_{local_rank}.json")
    with open(result_file, "w") as f:
        json.dump(serializable, f)

    if not args.cpu_only:
        dist.destroy_process_group()
        # Skip Python/CUDA shutdown -- NCCL destructors SIGSEGV on exit in some
        # driver versions. Results are already on disk.
        os._exit(0)


# Orchestrator-side sweep (runs in main process)

def run_tp2_sweep(args):
    """
    Orchestrator:
      - builds adapter_ids for each (N,K,sorted) config
      - times Phase 1 (sort) once (orchestrator-side, shared across both workers)
      - spawns two worker processes via torchrun-compatible mp.spawn
      - collects per-GPU CTA times (Phase 2) and SGMV+allreduce times from workers
      - combines all phases into output CSVs
    """
    import torch.multiprocessing as mp

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Resolve grid defaults based on interconnect
    N_values = args.N_values or (
        DEFAULT_N_VALUES_PCIE if args.interconnect == "pcie"
        else DEFAULT_N_VALUES_NVLINK
    )
    K_values = args.K_values or (
        DEFAULT_K_VALUES_PCIE if args.interconnect == "pcie"
        else DEFAULT_K_VALUES_NVLINK
    )

    configs = [
        (N, K, is_sorted)
        for N in N_values
        for K in K_values
        if K <= N
        for is_sorted in [False, True]
    ]

    # Build all adapter_ids maps (orchestrator side)
    adapter_ids_map = {}
    for (N, K, is_sorted) in configs:
        adapter_ids = (
            build_sorted_batch(N, K)
            if is_sorted
            else build_unsorted_batch(N, K, rng)
        )
        adapter_ids_map[(N, K, is_sorted)] = adapter_ids

    # Orchestrator Phase 1: adapter sort timing
    # Sort runs once in orchestrator per kernel_decomposition.md §3.1 (shared across workers)
    orch_sort_us = {}
    for (N, K, is_sorted), adapter_ids in adapter_ids_map.items():
        if not is_sorted:
            mean_us, _ = time_sort_us(adapter_ids, args.n_reps)
            orch_sort_us[(N, K)] = mean_us
        else:
            orch_sort_us.setdefault((N, K), 0.0)

    # Run two workers (GPU via mp.spawn, CPU-only runs inline)
    # Workers write results to JSON files in result_dir so the data lands on disk
    # before os._exit(0), surviving the NCCL-destructor SIGSEGV that occurs on
    # some driver versions during process cleanup.
    import json, shutil, tempfile
    result_dir = tempfile.mkdtemp(prefix="e11_tp2_")
    num_gpus = 2

    print(f"Spawning {num_gpus} GPU workers (TP=2, {args.interconnect}) ...")

    if args.cpu_only:
        for local_rank in range(num_gpus):
            _worker_main(local_rank, args, adapter_ids_map, result_dir)
    else:
        # Set required NCCL env vars for mp.spawn (torchrun alternative)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29502")
        os.environ["WORLD_SIZE"] = str(num_gpus)

        try:
            mp.spawn(
                _worker_main,
                args=(args, adapter_ids_map, result_dir),
                nprocs=num_gpus,
                join=True,
            )
        except mp.ProcessExitedException:
            # NCCL cleanup SIGSEGV is expected on some driver versions; the result
            # files are written before os._exit(0) so they're already on disk.
            missing = [
                r for r in range(num_gpus)
                if not os.path.exists(os.path.join(result_dir, f"rank_{r}.json"))
            ]
            if missing:
                shutil.rmtree(result_dir, ignore_errors=True)
                raise RuntimeError(f"GPU workers {missing} failed before writing results")

    # Collect results from JSON files
    worker_results = {}
    for rank in range(num_gpus):
        result_file = os.path.join(result_dir, f"rank_{rank}.json")
        with open(result_file) as f:
            data = json.load(f)
        worker_results[rank] = {}
        for key_str, val in data.items():
            n_s, k_s, s_s, r_s = key_str.split(",")
            tkey = (int(n_s), int(k_s), bool(int(s_s)), int(r_s))
            worker_results[rank][tkey] = val
    shutil.rmtree(result_dir, ignore_errors=True)

    # Combine results into output rows
    timing_fields = [
        "N", "K", "sorted", "interconnect", "n_reps",
        "orch_sort_mean_us",        # Phase 1: orchestrator sort (runs once)
        "gpu0_cta_mean_us",         # Phase 2: CTA construction, GPU 0 worker
        "gpu1_cta_mean_us",         # Phase 2: CTA construction, GPU 1 worker
        "wall_cta_mean_us",         # wall-clock CTA (max across workers, parallel)
        "decomp_total_mean_us",     # orch_sort + wall_cta
        "kernel_mean_ms",           # Phase 4: SGMV kernel (GPU 0, representative)
        "allreduce_mean_ms",        # NCCL all-reduce time (PCIe or NVLink)
        "total_latency_ms",         # decomp + kernel + allreduce
        "preprocessing_fraction",   # (sort + cta) / total_latency
        "theoretical_speedup",
        "measured_speedup_decomp",
    ]

    timing_rows = []
    unsorted_decomp: dict = {}

    for (N, K, is_sorted) in configs:
        sort_us = orch_sort_us.get((N, K), 0.0) if not is_sorted else 0.0

        r0 = worker_results.get(0, {}).get((N, K, is_sorted, 0), {})
        r1 = worker_results.get(1, {}).get((N, K, is_sorted, 1), {})

        gpu0_cta = r0.get("cta_mean_us", float("nan"))
        gpu1_cta = r1.get("cta_mean_us", float("nan"))
        wall_cta = max(
            gpu0_cta if not (isinstance(gpu0_cta, float) and gpu0_cta != gpu0_cta) else 0.0,
            gpu1_cta if not (isinstance(gpu1_cta, float) and gpu1_cta != gpu1_cta) else 0.0,
        )
        decomp_total = sort_us + wall_cta

        if not is_sorted:
            unsorted_decomp.setdefault(N, {})[K] = decomp_total

        kernel_ms = r0.get("kernel_mean_ms", "nan")
        allreduce_ms = r0.get("allreduce_mean_ms", "nan")

        if (
            kernel_ms != "nan" and allreduce_ms != "nan"
            and not (isinstance(kernel_ms, float) and kernel_ms != kernel_ms)
        ):
            total_latency_ms = round(
                decomp_total / 1e3 + float(kernel_ms) + float(allreduce_ms), 6
            )
            preprocessing_fraction = round(
                (decomp_total / 1e3) / total_latency_ms, 6
            ) if total_latency_ms > 0 else "nan"
        else:
            total_latency_ms = "nan"
            preprocessing_fraction = "nan"

        theoretical_speedup = round(N / K, 2) if is_sorted else 1.0
        measured_speedup = "nan"
        if is_sorted and N in unsorted_decomp and K in unsorted_decomp.get(N, {}):
            ref = unsorted_decomp[N][K]
            if ref > 0 and decomp_total > 0:
                measured_speedup = round(ref / decomp_total, 4)

        timing_rows.append({
            "N": N,
            "K": K,
            "sorted": 1 if is_sorted else 0,
            "interconnect": args.interconnect,
            "n_reps": args.n_reps,
            "orch_sort_mean_us": round(sort_us, 4),
            "gpu0_cta_mean_us": round(gpu0_cta, 4) if isinstance(gpu0_cta, float) else "nan",
            "gpu1_cta_mean_us": round(gpu1_cta, 4) if isinstance(gpu1_cta, float) else "nan",
            "wall_cta_mean_us": round(wall_cta, 4),
            "decomp_total_mean_us": round(decomp_total, 4),
            "kernel_mean_ms": kernel_ms,
            "allreduce_mean_ms": allreduce_ms,
            "total_latency_ms": total_latency_ms,
            "preprocessing_fraction": preprocessing_fraction,
            "theoretical_speedup": theoretical_speedup,
            "measured_speedup_decomp": measured_speedup,
        })

        print(
            f"N={N} K={K} {'sorted' if is_sorted else 'unsorted'}: "
            f"sort={sort_us:.2f}µs  cta(wall)={wall_cta:.2f}µs  "
            f"kernel={kernel_ms}ms  allreduce={allreduce_ms}ms  "
            f"speedup={measured_speedup}"
        )

    # Write timing_sweep.csv
    timing_path = os.path.join(args.output_dir, "timing_sweep.csv")
    with open(timing_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=timing_fields)
        w.writeheader()
        w.writerows(timing_rows)
    print(f"\nTP=2 timing sweep -> {timing_path}")

    # Write allreduce_masking.csv (§4.5b / §4.6b)
    ar_path = os.path.join(args.output_dir, "allreduce_masking.csv")
    ar_fields = [
        "N", "K", "sorted", "interconnect",
        "decomp_us", "kernel_ms", "allreduce_ms",
        "total_latency_ms", "preprocessing_fraction",
    ]
    ar_rows = [
        {
            "N": r["N"], "K": r["K"], "sorted": r["sorted"],
            "interconnect": r["interconnect"],
            "decomp_us": r["decomp_total_mean_us"],
            "kernel_ms": r["kernel_mean_ms"],
            "allreduce_ms": r["allreduce_mean_ms"],
            "total_latency_ms": r["total_latency_ms"],
            "preprocessing_fraction": r["preprocessing_fraction"],
        }
        for r in timing_rows
    ]
    with open(ar_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ar_fields)
        w.writeheader()
        w.writerows(ar_rows)
    print(f"All-reduce masking -> {ar_path}")

    # Write tp2_decomp_breakdown.csv (§4.5c orchestrator vs. per-GPU)
    breakdown_path = os.path.join(args.output_dir, "tp2_decomp_breakdown.csv")
    bd_fields = [
        "N", "K", "sorted", "interconnect",
        "orch_sort_us", "gpu0_cta_us", "gpu1_cta_us",
        "wall_cta_us", "wall_decomp_us",
        "tp_invariant_vs_single"
    ]
    bd_rows = [
        {
            "N": r["N"], "K": r["K"], "sorted": r["sorted"],
            "interconnect": r["interconnect"],
            "orch_sort_us": r["orch_sort_mean_us"],
            "gpu0_cta_us": r["gpu0_cta_mean_us"],
            "gpu1_cta_us": r["gpu1_cta_mean_us"],
            "wall_cta_us": r["wall_cta_mean_us"],
            "wall_decomp_us": r["decomp_total_mean_us"],
            # Placeholder -- filled by regression analysis after single-GPU results available
            "tp_invariant_vs_single": "TBD",
        }
        for r in timing_rows
    ]
    with open(breakdown_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bd_fields)
        w.writeheader()
        w.writerows(bd_rows)
    print(f"TP=2 decomp breakdown -> {breakdown_path}")


def main():
    args = parse_args()
    run_tp2_sweep(args)


if __name__ == "__main__":
    main()
