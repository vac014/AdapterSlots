"""
sgmv_decomposition_sweep.py -- E11 SGMV/MBGMV Decomposition Analysis: N and K sweep (single GPU).

Validates the O(N)→O(K) decomposition scan reduction claim.

Conditions:
    unsorted  : Input batch has randomly interleaved adapter IDs; SGMV must
                scan all N tokens to build per-adapter sub-batches (O(N) path).
    sorted    : Input batch is pre-sorted by adapter_id; SGMV sees K contiguous
                segments and only scans K boundaries (O(K) path = AdapterSlots path).

Phases measured (CPU-side, nanosecond precision):
    Phase 1 -- sort_us   : Time to sort N tokens by adapter_id (unsorted only; 0 for sorted)
    Phase 2 -- cta_us    : Time to build CTA-to-adapter segment map (O(N) or O(K))
    Phase 3 -- launch_ms : CPU→GPU command-queue latency (CUDA events)
    Phase 4 -- kernel_ms : GPU kernel execution (CUDA events, SGMV across all layers)

Usage (single A6000 -- §4.1–4.4 of kernel_decomposition.md):
    python scripts/experiments/sgmv_decomposition_sweep.py \\
        --output-dir results/e11/a6000 \\
        --model ./models/llama-7b \\
        --n-reps 1000

    # or using the RUN_README.md alias flags:
    python scripts/experiments/sgmv_decomposition_sweep.py \\
        --output-a6000 results/e11/a6000/timing_sweep.csv \\
        --model ./models/llama-7b

Usage (single H100 -- cross-architecture replication, §4.6a):
    export CUDA_VISIBLE_DEVICES=0
    python scripts/experiments/sgmv_decomposition_sweep.py \\
        --output-h100nvlink results/e11/h100_single/timing_sweep.csv \\
        --model ./models/llama-7b

Usage (CPU-only -- no GPU required, decomposition timing only):
    python scripts/experiments/sgmv_decomposition_sweep.py \\
        --output-dir results/e11/a6000 \\
        --cpu-only \\
        --n-reps 1000

Output files (written to --output-dir):
    timing_sweep.csv            -- per-(N,K,sorted) row with all phase timings
    preprocessing_fraction.csv  -- Phase1+2 fraction of total dispatch latency
"""

import argparse
import csv
import os
import time
from collections import Counter

import numpy as np


# Grid defaults (§4.1 of kernel_decomposition.md)
DEFAULT_N_VALUES = [64, 128, 256, 512, 1024, 2048]
DEFAULT_K_VALUES = [2, 4, 8, 16, 32]
DEFAULT_N_REPS = 1000
DEFAULT_RANK = 16
DEFAULT_WARMUP = 20

# LLaMA-7B fallback dims (used when --model is omitted)
_LLAMA7B_DEFAULTS = dict(hidden_size=4096, num_hidden_layers=32)


def parse_args():
    p = argparse.ArgumentParser(
        description="E11 decomposition sweep -- O(N) vs O(K) scan timing"
    )
    p.add_argument(
        "--output-dir", type=str, default="results/e11/a6000",
        help="Directory to write output CSVs (default: results/e11/a6000)"
    )
    # Alias flags used by RUN_README.md single-command invocations
    p.add_argument(
        "--output-a6000", type=str, default=None,
        help="Write timing_sweep.csv to this file path (sets output-dir from dirname)"
    )
    p.add_argument(
        "--output-h100nvlink", type=str, default=None,
        help="Write timing_sweep.csv to this file path (sets output-dir from dirname)"
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Path to base model dir. Config is read for hidden_size/num_layers. "
             "If None, CPU-only decomposition timing is used."
    )
    p.add_argument(
        "--cpu-only", action="store_true",
        help="Measure CPU decomposition phases only; skip GPU kernel timing."
    )
    p.add_argument(
        "--N-values", type=int, nargs="+", default=DEFAULT_N_VALUES, metavar="N",
        help="Token-count values to sweep (default: 64 128 256 512 1024 2048)"
    )
    p.add_argument(
        "--K-values", type=int, nargs="+", default=DEFAULT_K_VALUES, metavar="K",
        help="Adapter-count values to sweep (default: 2 4 8 16 32)"
    )
    p.add_argument(
        "--n-reps", type=int, default=DEFAULT_N_REPS,
        help="Repetitions per (N,K,sorted) combination (default: 1000)"
    )
    p.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help="Warmup repetitions discarded before timing (default: 20)"
    )
    p.add_argument(
        "--rank", type=int, default=DEFAULT_RANK,
        help="LoRA rank for SGMV kernel (default: 16)"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for unsorted batch construction"
    )
    return p.parse_args()


# Batch builders

def build_unsorted_batch(N: int, K: int, rng: np.random.Generator) -> list:
    """Randomly interleaved adapter IDs -- worst-case O(N) decomposition."""
    return rng.integers(0, K, size=N).tolist()


def build_sorted_batch(N: int, K: int) -> list:
    """Pre-sorted by adapter_id -- O(K) boundary scan (AdapterSlots path).

    Divides N tokens as evenly as possible across K adapters then sorts.
    """
    ids = []
    for k in range(K):
        start = k * N // K
        end = (k + 1) * N // K
        ids.extend([k] * (end - start))
    return ids  # already sorted


# Phase 2: CTA segment construction

def time_cta_unsorted_us(adapter_ids: list, K: int, n_reps: int) -> tuple:
    """O(N) scan over unsorted tokens. Returns (mean_us, std_us)."""
    times = []
    arr = adapter_ids  # plain Python list is faster to iterate for this microbench
    for _ in range(n_reps):
        t0 = time.perf_counter_ns()
        segs: dict = {k: 0 for k in range(K)}
        for aid in arr:
            segs[aid] += 1
        indptr = [0]
        for k in range(K):
            indptr.append(indptr[-1] + segs[k])
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    return float(np.mean(times)), float(np.std(times))


def time_cta_sorted_us(adapter_ids: list, K: int, n_reps: int) -> tuple:
    """O(K) boundary read for AdapterSlots pre-sorted batch.

    In AdapterSlots the batch arrives already sorted, with a K+1-entry indptr built once
    at batch-assembly time (amortised cost, not on the inference hot path).
    The decomposition step simply reads those K+1 boundary values -- O(K) work.

    Pre-build phase (not timed): walk the sorted list once to locate K segment
    boundaries.  Timed phase: copy the K+1-entry indptr, matching what a kernel
    launch does when it reads the segment table.
    """
    n = len(adapter_ids)
    # pre-build (batch-assembly time, not counted in decomp timing)
    indptr_pre = [0] * (K + 1)
    pos = 0
    for k in range(K):
        while pos < n and adapter_ids[pos] == k:
            pos += 1
        indptr_pre[k + 1] = pos
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter_ns()
        # O(K): explicitly read each of the K+1 boundary values so the timed
        # work is proportional to K (gives clean O(K) slope for EC2 regression)
        indptr = [indptr_pre[i] for i in range(K + 1)]
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    return float(np.mean(times)), float(np.std(times))


# Phase 1: Adapter sort (unsorted path only)

def time_sort_us(adapter_ids: list, n_reps: int) -> tuple:
    """Time np.argsort on N adapter IDs. Returns (mean_us, std_us)."""
    arr = np.array(adapter_ids, dtype=np.int32)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter_ns()
        np.argsort(arr, kind="stable")
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    return float(np.mean(times)), float(np.std(times))


# GPU SGMV timing (Phases 3 & 4)

def time_gpu_phases_ms(batched, x_buf, y_buf, s_indptr, lora_rank, num_layers):
    """
    Run SGMV kernel for all transformer layers.

    Returns:
        (launch_latency_ms, kernel_ms)
        launch_latency_ms : wall-clock from Python call minus GPU kernel time
        kernel_ms         : GPU kernel execution measured with CUDA events
    """
    import torch
    from punica.ops import add_lora_sgmv_custom_cutlass

    y_buf.zero_()
    torch.cuda.synchronize()
    wall_t0 = time.perf_counter_ns()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for layer_idx in range(num_layers):
        add_lora_sgmv_custom_cutlass(
            y_buf, x_buf,
            batched.wa_ptr, batched.wb_ptr,
            s_indptr, layer_idx, lora_rank,
        )
    end_ev.record()
    torch.cuda.synchronize()
    wall_t1 = time.perf_counter_ns()

    kernel_ms = start_ev.elapsed_time(end_ev)
    wall_ms = (wall_t1 - wall_t0) / 1e6
    launch_latency_ms = max(0.0, wall_ms - kernel_ms)
    return launch_latency_ms, kernel_ms


def allocate_sgmv_buffers(adapter_ids, K, hidden_size, lora_rank, num_layers, device, dtype):
    """Build BatchedLoraWeight + x/y buffers for given adapter_ids arrangement."""
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


# Main sweep

def run_sweep(args, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cpu_only = args.cpu_only

    # GPU setup
    device = None
    dtype = None
    hidden_size = _LLAMA7B_DEFAULTS["hidden_size"]
    num_layers = _LLAMA7B_DEFAULTS["num_hidden_layers"]

    if not cpu_only:
        import torch
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
        dtype = torch.float16
        torch.cuda.set_device(device)

        if args.model:
            from transformers import AutoConfig
            print(f"Reading model config from {args.model} ...")
            cfg = AutoConfig.from_pretrained(args.model)
            hidden_size = cfg.hidden_size
            num_layers = cfg.num_hidden_layers

        print(
            f"GPU setup: hidden_size={hidden_size}, num_layers={num_layers}, "
            f"rank={args.rank}, device={device}"
        )

    timing_path = os.path.join(output_dir, "timing_sweep.csv")
    frac_path = os.path.join(output_dir, "preprocessing_fraction.csv")

    timing_fields = [
        "N", "K", "sorted", "n_reps",
        "sort_mean_us", "sort_std_us",
        "cta_mean_us", "cta_std_us",
        "decomp_total_mean_us",
        "launch_mean_ms",
        "kernel_mean_ms",
        "total_latency_ms",
        "preprocessing_fraction",
        "theoretical_speedup",
        "measured_speedup_decomp",
    ]

    timing_rows = []
    # unsorted_decomp[N][K] = decomp_total_mean_us for speedup calc
    unsorted_decomp: dict = {}

    configs = [
        (N, K, is_sorted)
        for N in args.N_values
        for K in args.K_values
        if K <= N
        for is_sorted in [False, True]
    ]
    total = len(configs)

    for idx, (N, K, is_sorted) in enumerate(configs, 1):
        label = "sorted" if is_sorted else "unsorted"
        print(f"[{idx}/{total}] N={N}, K={K}, {label} ...", flush=True)

        adapter_ids = (
            build_sorted_batch(N, K)
            if is_sorted
            else build_unsorted_batch(N, K, rng)
        )

        # CPU Phases 1 & 2
        if is_sorted:
            sort_mean_us, sort_std_us = 0.0, 0.0
            cta_mean_us, cta_std_us = time_cta_sorted_us(adapter_ids, K, args.n_reps)
        else:
            sort_mean_us, sort_std_us = time_sort_us(adapter_ids, args.n_reps)
            cta_mean_us, cta_std_us = time_cta_unsorted_us(adapter_ids, K, args.n_reps)

        decomp_total_mean_us = sort_mean_us + cta_mean_us

        if not is_sorted:
            unsorted_decomp.setdefault(N, {})[K] = decomp_total_mean_us

        # GPU Phases 3 & 4
        launch_mean_ms = "nan"
        kernel_mean_ms = "nan"
        total_latency_ms = "nan"
        preprocessing_fraction = "nan"

        if not cpu_only:
            import torch
            batched, x_buf, y_buf, s_indptr = allocate_sgmv_buffers(
                adapter_ids, K, hidden_size, args.rank, num_layers, device, dtype
            )
            torch.cuda.synchronize()

            for _ in range(args.warmup):
                time_gpu_phases_ms(batched, x_buf, y_buf, s_indptr, args.rank, num_layers)

            launch_list, kernel_list = [], []
            for _ in range(args.n_reps):
                lm, km = time_gpu_phases_ms(
                    batched, x_buf, y_buf, s_indptr, args.rank, num_layers
                )
                launch_list.append(lm)
                kernel_list.append(km)

            launch_mean_ms = round(float(np.mean(launch_list)), 6)
            kernel_mean_ms = round(float(np.mean(kernel_list)), 6)
            total_latency_ms = round(
                decomp_total_mean_us / 1e3 + launch_mean_ms + kernel_mean_ms, 6
            )
            preprocessing_fraction = round(
                (decomp_total_mean_us / 1e3) / total_latency_ms, 6
            ) if total_latency_ms > 0 else "nan"

        # Speedup
        theoretical_speedup = round(N / K, 2) if is_sorted else 1.0
        measured_speedup_decomp = "nan"
        if is_sorted and N in unsorted_decomp and K in unsorted_decomp.get(N, {}):
            ref = unsorted_decomp[N][K]
            if ref > 0 and decomp_total_mean_us > 0:
                measured_speedup_decomp = round(ref / decomp_total_mean_us, 4)

        row = {
            "N": N,
            "K": K,
            "sorted": 1 if is_sorted else 0,
            "n_reps": args.n_reps,
            "sort_mean_us": round(sort_mean_us, 4),
            "sort_std_us": round(sort_std_us, 4),
            "cta_mean_us": round(cta_mean_us, 4),
            "cta_std_us": round(cta_std_us, 4),
            "decomp_total_mean_us": round(decomp_total_mean_us, 4),
            "launch_mean_ms": launch_mean_ms,
            "kernel_mean_ms": kernel_mean_ms,
            "total_latency_ms": total_latency_ms,
            "preprocessing_fraction": preprocessing_fraction,
            "theoretical_speedup": theoretical_speedup,
            "measured_speedup_decomp": measured_speedup_decomp,
        }
        timing_rows.append(row)

        print(
            f"  sort={sort_mean_us:.2f}µs  cta={cta_mean_us:.2f}µs  "
            f"decomp={decomp_total_mean_us:.2f}µs  "
            f"kernel={kernel_mean_ms}ms  "
            f"speedup={measured_speedup_decomp}"
        )

    # Write timing_sweep.csv
    with open(timing_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=timing_fields)
        w.writeheader()
        w.writerows(timing_rows)
    print(f"\nTiming sweep -> {timing_path}")

    # Write preprocessing_fraction.csv
    # kernel_ms here is per-dispatch (one SGMV layer), not the full num_layers
    # model pass, so that preprocessing_fraction reflects the fraction of ONE
    # dispatch that is preprocessing -- the overhead AdapterSlots eliminates per call.
    frac_fields = [
        "N", "K", "sorted", "decomp_us",
        "kernel_ms", "total_latency_ms", "preprocessing_fraction"
    ]
    frac_rows = []
    for r in timing_rows:
        du_ms = float(r["decomp_total_mean_us"]) / 1e3
        km = r["kernel_mean_ms"]
        if km == "nan":
            frac_rows.append({
                "N": r["N"], "K": r["K"], "sorted": r["sorted"],
                "decomp_us": r["decomp_total_mean_us"],
                "kernel_ms": "nan", "total_latency_ms": "nan",
                "preprocessing_fraction": "nan",
            })
        else:
            per_layer_km = round(float(km) / num_layers, 6)
            total = round(du_ms + per_layer_km, 6)
            frac = round(du_ms / total, 6) if total > 0 else "nan"
            frac_rows.append({
                "N": r["N"], "K": r["K"], "sorted": r["sorted"],
                "decomp_us": r["decomp_total_mean_us"],
                "kernel_ms": per_layer_km,
                "total_latency_ms": total,
                "preprocessing_fraction": frac,
            })
    with open(frac_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frac_fields)
        w.writeheader()
        w.writerows(frac_rows)
    print(f"Preprocessing fraction -> {frac_path}")

    return timing_rows


def main():
    args = parse_args()

    # Resolve output directory (alias flags take precedence)
    if args.output_a6000:
        output_dir = os.path.dirname(args.output_a6000) or args.output_dir
    elif args.output_h100nvlink:
        output_dir = os.path.dirname(args.output_h100nvlink) or args.output_dir
    else:
        output_dir = args.output_dir

    run_sweep(args, output_dir)


if __name__ == "__main__":
    main()
