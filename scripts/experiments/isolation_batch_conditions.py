"""
isolation_batch_conditions.py -- E1 batch layouts for SGMV under adapter mixing.

Conditions (N tokens):
    A  all tokens from adapter 0                          (WAR 1.00)
    B  two contiguous halves, adapters 0 and 1            (WAR ~1.00)
    C  alternating 16-token blocks of adapters 0 and 1    (WAR ~0.50)
    D  token-by-token interleaving of K adapters          (WAR ~0.00)
    S  K contiguous blocks, one per adapter               (WAR ~1.00)

Each run applies Punica's add_lora_sgmv_custom_cutlass to every layer of the
model, on K random LoRA weight sets of the model's width; only the model config
is read. scripts/reproduce/ncu_e1.sh profiles this script with Nsight Compute.

Usage:
    python scripts/experiments/isolation_batch_conditions.py --condition D \\
        --model ./models/llama-7b --N 4096 --K 4 --rank 16 --n-runs 20
"""

import argparse
import csv
import os

# A single-GPU run uses cuda:0; under torchrun each rank picks its own device.
if "LOCAL_RANK" not in os.environ and "RANK" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch

CONDITION_WAR = {"A": 1.00, "B": 1.00, "C": 0.50, "D": 0.00, "S": 1.00}


def parse_args():
    p = argparse.ArgumentParser(description="E1 SGMV batch layouts")
    p.add_argument("--condition", type=str, choices=["A", "B", "C", "D", "S"], required=True)
    p.add_argument("--N", type=int, default=512, help="Total tokens per batch")
    p.add_argument("--K", type=int, default=2,
                   help="Number of adapters (at least 2 for B, C and D)")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--n-runs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, required=True,
                   help="Base model directory; only hidden_size and num_hidden_layers "
                        "are read from its config.")
    p.add_argument("--output", type=str, default=None, help="CSV output path")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor-parallel degree; with --tp > 1 launch under torchrun. "
                        "Each rank runs SGMV and all-reduces the output.")
    return p.parse_args()


# Batch layouts

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
    """Fully interleaved: 0,1,...,K-1,0,1,..."""
    return [i % K for i in range(N)]


def build_condition_S(N: int, K: int) -> list:
    """Adapter-aligned at the same K: K contiguous blocks, one per adapter."""
    return [min(i * K // N, K - 1) for i in range(N)]


BUILDERS = {
    "S": build_condition_S,
    "A": build_condition_A,
    "B": build_condition_B,
    "C": build_condition_C,
    "D": build_condition_D,
}


def build_batch(condition: str, N: int, K: int) -> list:
    return BUILDERS[condition](N, K)


def compute_war(adapter_ids: list, warp_size: int = 32) -> float:
    arr = np.array(adapter_ids, dtype=np.int32)
    m = len(arr) // warp_size
    if m == 0:
        return 0.0
    warps = arr[: m * warp_size].reshape(m, warp_size)
    return float(np.mean(warps.min(axis=1) == warps.max(axis=1)))


def build_sgmv_segments(adapter_ids: list):
    """
    Build Punica SGMV segments from a flat adapter_ids list, in token order.

    Each maximal run of consecutive tokens on one adapter is one segment, which
    is what SGMV executes for the batch as laid out. (Deriving segments from
    per-adapter token counts would sort the batch first, and every condition
    would reach the kernel adapter-aligned.)

    Returns:
        seg_adapters : adapter ID of each segment (repeats when an adapter recurs)
        s_list       : segment boundaries [0, end_0, ..., N] (length S+1)
    """
    seg_adapters, s = [], [0]
    for i, aid in enumerate(adapter_ids):
        if i == 0 or aid != adapter_ids[i - 1]:
            if i:
                s.append(i)
            seg_adapters.append(aid)
    s.append(len(adapter_ids))
    return seg_adapters, s


def sgmv_step_ms(batched, x_buf, y_buf, s_indptr, lora_rank: int, num_layers: int) -> float:
    """
    Run add_lora_sgmv_custom_cutlass for every layer and return the elapsed GPU
    time in milliseconds (CUDA events).

    batched, x_buf, y_buf and s_indptr are allocated once by the caller, outside
    the warmup and timing loops, so no allocation kernels enter the NCU profile;
    y_buf is zeroed in place before each step.
    """
    from punica.ops import add_lora_sgmv_custom_cutlass

    y_buf.zero_()

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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

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
        print(f"  Expected WAR~{CONDITION_WAR[args.condition]}")
        if args.tp > 1:
            print(f"  TP={args.tp}  local_rank={local_rank}  world_size={world_size}")

    from punica.utils import BatchedLoraWeight, LoraWeight
    from transformers import AutoConfig

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float16
    cfg = AutoConfig.from_pretrained(args.model)
    hidden_size = cfg.hidden_size
    num_layers = cfg.num_hidden_layers
    lora_rank = args.rank
    print(f"  hidden_size={hidden_size}, num_layers={num_layers}, rank={lora_rank}")

    lora_weights = [
        LoraWeight(num_layers, hidden_size, hidden_size, lora_rank, dtype, device)
        for _ in range(args.K)
    ]

    seg_adapters, s_list = build_sgmv_segments(adapter_ids)
    s_indptr = torch.tensor(s_list, dtype=torch.int32, device=device)
    seg_sizes = [s_list[i + 1] - s_list[i] for i in range(len(seg_adapters))]
    print(f"  {len(seg_adapters)} segments, sizes {seg_sizes[:8]}"
          f"{' ...' if len(seg_sizes) > 8 else ''}")

    batched = BatchedLoraWeight([lora_weights[aid] for aid in seg_adapters])
    x_buf = torch.randn(args.N, hidden_size, dtype=dtype, device=device)
    y_buf = torch.zeros(args.N, hidden_size, dtype=dtype, device=device)
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        sgmv_step_ms(batched, x_buf, y_buf, s_indptr, lora_rank, num_layers)
        if args.tp > 1:
            dist.all_reduce(y_buf, op=dist.ReduceOp.SUM)

    elapsed_list = []
    for _ in range(args.n_runs):
        if args.tp > 1:
            # SGMV and the all-reduce timed together.
            from punica.ops import add_lora_sgmv_custom_cutlass
            y_buf.zero_()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for layer_idx in range(num_layers):
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

    if local_rank == 0:
        print(f"\nSGMV ({args.n_runs} runs, TP={args.tp}): "
              f"mean={np.mean(elapsed_list):.3f}ms  std={np.std(elapsed_list):.3f}ms  "
              f"p50={np.percentile(elapsed_list, 50):.3f}ms  "
              f"p99={np.percentile(elapsed_list, 99):.3f}ms")

    if args.output and local_rank == 0:
        rows = [{
            "condition": args.condition, "run": i, "N": args.N, "K": args.K,
            "WAR": round(war, 4), "n_flips": n_flips, "n_segments": len(seg_adapters),
            "elapsed_ms": round(t, 4),
        } for i, t in enumerate(elapsed_list)]
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results -> {args.output}")

    if args.tp > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
