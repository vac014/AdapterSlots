"""
isolation_batch_conditions.py -- E1 isolation experiment batch script (Bradford Hill four-condition design).

Conditions:
    A  512 tokens from adapter 0 only             (0 adapter flips, WAR=1.00)
    B  256 tokens A + 256 tokens B contiguous      (1 flip, WAR≈1.00)
    C  Alternating blocks of 16 tokens, A/B        (32 flips, WAR≈0.50)
    D  Fully interleaved ABABAB... token-by-token  (512 flips, WAR≈0.00)

Usage (CPU-only decomposition timing):
    python scripts/experiments/isolation_batch_conditions.py --condition D --n-runs 100 --N 512 --K 2 \
        --output results/e1/timing_condition_D.csv

Usage (GPU SGMV kernel micro-benchmark via Punica -- the correct E1 path):
    python scripts/experiments/isolation_batch_conditions.py --condition D --model ./models/llama-7b \
        --adapter-dir ./adapters --n-runs 100

    The --model arg is used only to read the config (hidden_size, num_hidden_layers).
    No model weights are loaded. K LoRA weight tensors are created randomly on GPU
    and fed to Punica's add_lora_sgmv_custom_cutlass (which calls sgmv_cutlass).

Used by ncu_e1.sh / nsys_e1.sh as the profiling target:
    ncu --kernel-name sgmv_cutlass ... python scripts/experiments/isolation_batch_conditions.py --condition D --model ...
"""

import argparse
import csv
import json
import os
import time
from collections import Counter

# Pin to cuda:0 only when NOT running inside a torch.distributed job.
# torchrun sets LOCAL_RANK/RANK; when absent we are single-GPU.
if "LOCAL_RANK" not in os.environ and "RANK" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch

# LLaMA-3-8B defaults (used when --model is omitted)
_LLAMA3_8B_DEFAULTS = dict(hidden_size=4096, num_hidden_layers=32)

CONDITION_WAR = {"A": 1.00, "B": 1.00, "C": 0.50, "D": 0.00}
CONDITION_FLIPS = {"A": 0, "B": 1, "C": 32, "D": 512}


def parse_args():
    p = argparse.ArgumentParser(description="E1 isolation experiment (Bradford Hill design)")
    p.add_argument("--condition", type=str, choices=["A", "B", "C", "D"], default=None,
                   help="Condition to run (required unless --sweep-conditions is used).")
    p.add_argument("--N", type=int, default=512, help="Total tokens per batch")
    p.add_argument("--K", type=int, default=2,
                   help="Number of adapters to create (must be ≥ 2 for B/C/D)")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--n-runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default=None,
                   help="Model path. Config is read to get hidden_size/num_layers. "
                        "If None, LLaMA-7B defaults are used. No weights are loaded.")
    p.add_argument("--adapter-dir", type=str, default="./adapters",
                   help="Unused in GPU path; kept for CLI compatibility with ncu_e1.sh")
    p.add_argument("--output", type=str, default=None, help="CSV output path")
    p.add_argument("--log-sgmv-intensity", action="store_true",
                   help="Log SGMV operational intensity (tokens/adapter/dispatch)")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor-parallel degree. When >1, script must be launched "
                        "via torchrun --nproc_per_node=<tp>. Each rank inits NCCL, "
                        "runs SGMV on its GPU, and all-reduces output to model TP "
                        "communication overhead (PCIe PHB or NVLink 4.0).")
    p.add_argument("--sweep-conditions", type=str, default=None,
                   help="Comma-separated list of conditions to run sequentially "
                        "(e.g. A,B,C,D). Overrides --condition. Each condition runs "
                        "--n-runs batches. Used with ncu_e1.sh --sweep for batch-level "
                        "WAR vs. hardware-counter correlation (EC3).")
    p.add_argument("--war-map-output", type=str, default=None,
                   help="CSV path to write batch→WAR mapping when --sweep-conditions "
                        "is active. compute_correlations.py uses this to assign WAR "
                        "values to NCU kernel-launch groups.")
    return p.parse_args()


# Batch builders for each condition

def build_condition_A(N: int, K: int) -> list:
    """All tokens from adapter 0."""
    return [0] * N


def build_condition_B(N: int, K: int) -> list:
    """Two contiguous blocks: first N/2 from adapter 0, rest from adapter 1."""
    half = N // 2
    return [0] * half + [1] * (N - half)


def build_condition_C(N: int, K: int, block_size: int = 16) -> list:
    """Alternating blocks of block_size tokens from adapter 0 and 1."""
    ids = []
    adapter = 0
    while len(ids) < N:
        ids.extend([adapter] * min(block_size, N - len(ids)))
        adapter = 1 - adapter
    return ids[:N]


def build_condition_D(N: int, K: int) -> list:
    """Fully interleaved: 0,1,0,1,0,1,..."""
    return [i % K for i in range(N)]


BUILDERS = {
    "A": build_condition_A,
    "B": build_condition_B,
    "C": build_condition_C,
    "D": build_condition_D,
}


def build_batch(condition: str, N: int, K: int) -> list:
    return BUILDERS[condition](N, K)


# WAR

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

    SGMV operates on tokens that are pre-grouped by adapter.  We derive
    segment boundaries from token counts (order within each group does not
    affect kernel performance).

    Returns:
        appearing   : sorted list of adapter IDs that appear in the batch
        s_list      : Python list [0, count_0, count_0+count_1, ...] (length S+1)
    """
    counts = Counter(adapter_ids)
    appearing = sorted(counts.keys())
    s = [0]
    for aid in appearing:
        s.append(s[-1] + counts[aid])
    return appearing, s


# SGMV intensity logging

def log_sgmv_intensity(adapter_ids: list, condition: str, run_idx: int,
                        intensity_log: list):
    counts = Counter(adapter_ids)
    for aid, cnt in counts.items():
        intensity_log.append({
            "condition": condition,
            "run": run_idx,
            "adapter_id": aid,
            "tokens_in_chunk": cnt,
            "dispatch_num": run_idx,
        })


# CPU-only decomposition timing

def time_decomposition_us(adapter_ids: list, K: int, n_reps: int = 100) -> dict:
    """Measure time to decompose N tokens into K adapter segments."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter_ns()
        segments: dict = {k: [] for k in range(K)}
        for i, aid in enumerate(adapter_ids):
            segments[aid].append(i)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    return {
        "mean_us": float(np.mean(times)),
        "std_us": float(np.std(times)),
        "p50_us": float(np.percentile(times, 50)),
        "p99_us": float(np.percentile(times, 99)),
    }


# GPU SGMV micro-benchmark (pure kernel, no model weights loaded)

def sgmv_step_ms(
    batched,            # pre-computed BatchedLoraWeight (allocated once outside timing)
    x_buf,             # pre-allocated [N, hidden_size] input tensor (never reallocated)
    y_buf,             # pre-allocated [N, hidden_size] output tensor (zeroed in-place)
    s_indptr,           # [S+1] int32 GPU tensor, segment boundaries
    lora_rank: int,
    num_layers: int,
) -> float:
    """
    Run add_lora_sgmv_custom_cutlass for every transformer layer and return
    elapsed GPU time in milliseconds (measured with CUDA events).

    x_buf and y_buf must be pre-allocated by the caller (outside warmup/timing
    loops) so that torch.randn / torch.zeros GPU kernels do not appear in the
    NCU profiling trace. y_buf is zeroed in-place with .zero_() before each call.
    """
    from punica.ops import add_lora_sgmv_custom_cutlass

    y_buf.zero_()  # in-place memset; does not fire distribution_elementwise kernel

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for layer_idx in range(num_layers):
        add_lora_sgmv_custom_cutlass(
            y_buf, x_buf,
            batched.wa_ptr, batched.wb_ptr,
            s_indptr,
            layer_idx,
            lora_rank,
        )
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end)


def run_sweep(args):
    """Run all conditions sequentially for batch-level NCU correlation (EC3).

    Each condition's N_RUNS batches appear contiguously in the NCU trace.
    Writes a batch_war_map CSV so compute_correlations.py can assign WAR values
    to each group of kernel launches captured by NCU.
    """
    import torch
    conditions = [c.strip() for c in args.sweep_conditions.split(",")]
    war_map_rows = []
    global_batch_idx = 0

    for condition in conditions:
        adapter_ids = build_batch(condition, args.N, args.K)
        war_val = compute_war(adapter_ids)
        n_flips = sum(1 for i in range(1, len(adapter_ids))
                      if adapter_ids[i] != adapter_ids[i - 1])
        print(f"\n[sweep] Condition {condition}: WAR={war_val:.4f}  flips={n_flips}")

        if args.model is None:
            raise ValueError("--sweep-conditions requires --model (GPU path)")

        from punica.utils import LoraWeight, BatchedLoraWeight
        device = torch.device("cuda:0")
        dtype = torch.float16

        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_hidden_layers

        lora_weights = [
            LoraWeight(num_layers, hidden_size, hidden_size, args.rank, dtype, device)
            for _ in range(args.K)
        ]
        appearing, s_list = build_sgmv_segments(adapter_ids)
        s_indptr = torch.tensor(s_list, dtype=torch.int32, device=device)
        batched = BatchedLoraWeight([lora_weights[aid] for aid in appearing])
        x_buf = torch.randn(args.N, hidden_size, dtype=dtype, device=device)
        y_buf = torch.zeros(args.N, hidden_size, dtype=dtype, device=device)
        torch.cuda.synchronize()

        for _ in range(args.warmup):
            sgmv_step_ms(batched, x_buf, y_buf, s_indptr, args.rank, num_layers)

        for run_i in range(args.n_runs):
            sgmv_step_ms(batched, x_buf, y_buf, s_indptr, args.rank, num_layers)
            war_map_rows.append({
                "global_batch_idx": global_batch_idx,
                "condition": condition,
                "war": round(war_val, 4),
                "n_flips": n_flips,
                "run_within_condition": run_i,
            })
            global_batch_idx += 1

    if args.war_map_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.war_map_output)), exist_ok=True)
        with open(args.war_map_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=war_map_rows[0].keys())
            writer.writeheader()
            writer.writerows(war_map_rows)
        print(f"\nWAR map -> {args.war_map_output}")

    return war_map_rows


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Sweep mode: run all conditions for batch-level EC3 correlation
    if args.sweep_conditions:
        run_sweep(args)
        return

    if args.condition is None:
        raise ValueError("--condition is required unless --sweep-conditions is used.")

    # Distributed init for TP=N runs (launched via torchrun)
    dist = None
    local_rank = 0
    world_size = 1
    if args.tp > 1:
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", args.tp))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    adapter_ids = build_batch(args.condition, args.N, args.K)
    war = compute_war(adapter_ids)
    n_flips = sum(1 for i in range(1, len(adapter_ids))
                  if adapter_ids[i] != adapter_ids[i - 1])

    if local_rank == 0:
        print(f"E1 Condition {args.condition}:")
        print(f"  N={args.N}, K={args.K}, WAR={war:.4f}, adapter_flips={n_flips}")
        print(f"  Expected WAR≈{CONDITION_WAR[args.condition]}")
        if args.tp > 1:
            print(f"  TP={args.tp}  local_rank={local_rank}  world_size={world_size}")

    intensity_log = []
    rows = []

    # CPU-only path: decomposition timing
    if args.model is None:
        print(f"\nRunning CPU decomposition timing ({args.n_runs} runs) ...")
        decomp = time_decomposition_us(adapter_ids, args.K, n_reps=args.n_runs)
        print(f"  mean={decomp['mean_us']:.2f}µs  "
              f"p50={decomp['p50_us']:.2f}µs  p99={decomp['p99_us']:.2f}µs")

        rows.append({
            "condition": args.condition,
            "N": args.N,
            "K": args.K,
            "WAR": round(war, 4),
            "n_flips": n_flips,
            "decomp_mean_us": round(decomp["mean_us"], 3),
            "decomp_std_us": round(decomp["std_us"], 3),
            "decomp_p50_us": round(decomp["p50_us"], 3),
            "decomp_p99_us": round(decomp["p99_us"], 3),
            "elapsed_ms": "N/A",
        })

    # GPU path: Punica SGMV micro-benchmark
    else:
        from punica.utils import LoraWeight

        # When TP>1 (torchrun), each rank binds to its own GPU.
        device = torch.device(f"cuda:{local_rank}")
        dtype = torch.float16

        # Read model config for dimensions; do NOT load weights.
        if args.model:
            from transformers import AutoConfig
            print(f"\nReading model config from {args.model} ...")
            cfg = AutoConfig.from_pretrained(args.model)
            hidden_size = cfg.hidden_size
            num_layers = cfg.num_hidden_layers
        else:
            hidden_size = _LLAMA3_8B_DEFAULTS["hidden_size"]
            num_layers = _LLAMA3_8B_DEFAULTS["num_hidden_layers"]

        lora_rank = args.rank
        print(f"  hidden_size={hidden_size}, num_layers={num_layers}, rank={lora_rank}")

        # Allocate K random LoRA weight tensors on GPU (q-projection shape).
        print(f"Allocating {args.K} LoRA weight tensors on GPU ...")
        lora_weights = [
            LoraWeight(num_layers, hidden_size, hidden_size, lora_rank, dtype, device)
            for _ in range(args.K)
        ]

        # Build SGMV segment indptr from adapter_ids.
        appearing, s_list = build_sgmv_segments(adapter_ids)
        s_indptr = torch.tensor(s_list, dtype=torch.int32, device=device)
        print(f"  Segments: adapters={appearing}, boundaries={s_list}")

        # Print tokens/adapter/dispatch so framing_decision.md §1a can be filled
        # with measured post-sort values rather than theoretical pre-sort block sizes.
        seg_sizes = [s_list[i + 1] - s_list[i] for i in range(len(appearing))]
        print(f"  SGMV tokens/adapter/dispatch (post-sort, measured):")
        for aid, sz in zip(appearing, seg_sizes):
            print(f"    adapter {aid}: {sz} tokens  "
                  f"[use {sz} for condition {args.condition} in framing_decision §1a]")

        # Pre-allocate input/output buffers ONCE outside all loops so that
        # torch.randn / torch.zeros GPU kernels do not pollute NCU traces.
        from punica.utils import BatchedLoraWeight
        batched = BatchedLoraWeight([lora_weights[aid] for aid in appearing])
        x_buf = torch.randn(args.N, hidden_size, dtype=dtype, device=device)
        y_buf = torch.zeros(args.N, hidden_size, dtype=dtype, device=device)
        torch.cuda.synchronize()  # flush allocation kernels before NCU scope

        if local_rank == 0:
            print(f"Warmup ({args.warmup}) ...")
        for _ in range(args.warmup):
            sgmv_step_ms(batched, x_buf, y_buf, s_indptr, lora_rank, num_layers)
            if args.tp > 1:
                dist.all_reduce(y_buf, op=dist.ReduceOp.SUM)

        if local_rank == 0:
            print(f"Timing {args.n_runs} runs ...")
        elapsed_list = []
        for run_i in range(args.n_runs):
            if args.tp > 1:
                # Time SGMV + all_reduce together to capture TP communication cost.
                y_buf.zero_()
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                for layer_idx in range(num_layers):
                    from punica.ops import add_lora_sgmv_custom_cutlass
                    add_lora_sgmv_custom_cutlass(
                        y_buf, x_buf, batched.wa_ptr, batched.wb_ptr,
                        s_indptr, layer_idx, lora_rank,
                    )
                dist.all_reduce(y_buf, op=dist.ReduceOp.SUM)
                end_ev.record()
                torch.cuda.synchronize()
                t = start_ev.elapsed_time(end_ev)
            else:
                t = sgmv_step_ms(batched, x_buf, y_buf, s_indptr, lora_rank, num_layers)
            elapsed_list.append(t)
            if args.log_sgmv_intensity:
                log_sgmv_intensity(adapter_ids, args.condition, run_i, intensity_log)

        mean_ms = np.mean(elapsed_list)
        std_ms = np.std(elapsed_list)
        p50 = np.percentile(elapsed_list, 50)
        p99 = np.percentile(elapsed_list, 99)

        if local_rank == 0:
            print(f"\nGPU SGMV results ({args.n_runs} runs, TP={args.tp}):")
            print(f"  mean={mean_ms:.3f}ms  std={std_ms:.3f}ms  "
                  f"p50={p50:.3f}ms  p99={p99:.3f}ms")

        for run_i, t in enumerate(elapsed_list):
            rows.append({
                "condition": args.condition,
                "run": run_i,
                "N": args.N,
                "K": args.K,
                "WAR": round(war, 4),
                "n_flips": n_flips,
                "n_segments": len(appearing),
                "elapsed_ms": round(t, 4),
            })

    # Write outputs (rank 0 only when TP>1)
    if args.output and rows and local_rank == 0:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults -> {args.output}")

    if intensity_log and local_rank == 0:
        intensity_path = (args.output or "results/e1/sgmv_intensity").replace(
            ".csv", "_intensity.jsonl"
        )
        with open(intensity_path, "w") as f:
            for entry in intensity_log:
                f.write(json.dumps(entry) + "\n")
        print(f"SGMV intensity log -> {intensity_path}")

    # Clean up distributed process group.
    if args.tp > 1 and dist is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
