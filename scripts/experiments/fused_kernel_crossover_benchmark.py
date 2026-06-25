#!/usr/bin/env python3
"""
fused_kernel_crossover_benchmark.py -- psi_fuse and psi_promote against segment size.

For one adapter segment of n tokens (LLaMA-7B width, d = 4096) it times three
ways to compute Y = X W^T + alpha (X A^T) B^T on the GPU:

    SGMV-style   base GEMM, shrink and expand as separate launches, then add
    fused        FusedLoRAKernel (adapter_slots/kernel/fused_lora_kernel.py)
    merged GEMM  one GEMM on the merged weight W + alpha B A (WGKP L3)

psi_fuse = T_SGMV / T_fused and psi_gemm (psi_promote) = T_SGMV / T_GEMM.

Usage:
    python scripts/experiments/fused_kernel_crossover_benchmark.py \\
        --output-dir results/runs/kernel/fused_crossover --hardware a6000_tp1 \\
        --ranks 16 32 --n-tokens 4 8 16 32 64 128 256 --n-reps 500 --warmup 50

Writes <output-dir>/<hardware>_r<rank>.json per rank and, for the first rank,
<output-dir>/hw_profile_<hardware>.json in the format adapter_slots/kernel/apt.py
reads.
"""

import argparse
import json
import pathlib
import time
from typing import Dict, List

import torch

from adapter_slots.kernel.fused_lora_kernel import FusedLoRAKernel


def _time_us(fn, n_reps: int, warmup: int) -> float:
    """Mean wall-clock time of fn in microseconds, GPU-synchronised."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n_reps


def _find_crossover_n(rank_results: dict) -> int:
    """Smallest n at which the merged GEMM beats SGMV."""
    for n_str in sorted(rank_results, key=int):
        if rank_results[n_str]["gemm_faster"]:
            return int(n_str)
    return 32


def run_crossover(output_dir: str, hardware: str, ranks: List[int], n_token_list: List[int],
                  n_reps: int, warmup: int, d_model: int = 4096) -> Dict:
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev, dt = torch.device("cuda"), torch.float16
    kernel = FusedLoRAKernel()
    all_results = {}

    for rank in ranks:
        rank_results = {}
        print(f"\nhardware={hardware} rank={rank}")
        print(f"{'N':>6} {'T_SGMV(us)':>12} {'T_Fused(us)':>12} {'T_GEMM(us)':>12} "
              f"{'psi_fuse':>9} {'psi_gemm':>9}")
        for n_tokens in n_token_list:
            torch.manual_seed(42)
            X = torch.randn(n_tokens, d_model, dtype=dt, device=dev)
            W = torch.randn(d_model, d_model, dtype=dt, device=dev)
            A = torch.randn(rank, d_model, dtype=dt, device=dev)
            B = torch.randn(d_model, rank, dtype=dt, device=dev)
            alpha = 0.5
            W_k = W + (alpha * torch.matmul(B.float(), A.float())).to(dt)

            def sgmv():
                return torch.matmul(X, W.T) + alpha * torch.matmul(torch.matmul(X, A.T), B.T)

            t_sgmv = _time_us(sgmv, n_reps, warmup)
            t_fused = _time_us(lambda: kernel.forward(X, W, A, B, alpha), n_reps, warmup)
            t_gemm = _time_us(lambda: torch.matmul(X, W_k.T), n_reps, warmup)
            psi_fuse, psi_gemm = t_sgmv / t_fused, t_sgmv / t_gemm
            print(f"{n_tokens:>6d} {t_sgmv:>12.1f} {t_fused:>12.1f} {t_gemm:>12.1f} "
                  f"{psi_fuse:>9.3f} {psi_gemm:>9.3f}")
            rank_results[str(n_tokens)] = {
                "t_sgmv_us": round(t_sgmv, 2), "t_fused_us": round(t_fused, 2),
                "t_gemm_us": round(t_gemm, 2), "psi_fuse": round(psi_fuse, 4),
                "psi_gemm": round(psi_gemm, 4), "gemm_faster": psi_gemm >= 1.05,
            }
        all_results[f"rank{rank}"] = rank_results

        curve = {n: {"psi_fuse": v["psi_fuse"], "psi_gemm": v["psi_gemm"],
                     "gemm_faster": v["gemm_faster"]} for n, v in rank_results.items()}
        path = out / f"{hardware}_r{rank}.json"
        path.write_text(json.dumps({"hardware": hardware, "rank": rank, "n_reps": n_reps,
                                    "warmup": warmup, "crossover_curve": curve}, indent=2))
        print(f"wrote {path}")
        if rank == ranks[0]:
            path = out / f"hw_profile_{hardware}.json"
            path.write_text(json.dumps({"hardware": hardware, "rank": rank, "warp_size": 32,
                                        "gemm_crossover_n": _find_crossover_n(rank_results),
                                        "crossover_curve": curve}, indent=2))
            print(f"wrote {path}")
    return all_results


def main():
    p = argparse.ArgumentParser(description="psi_fuse and psi_promote against segment size")
    p.add_argument("--output-dir", default="results/runs/kernel/fused_crossover")
    p.add_argument("--hardware", default="a6000_tp1")
    p.add_argument("--ranks", type=int, nargs="+", default=[16, 32])
    p.add_argument("--n-tokens", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--n-reps", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--d-model", type=int, default=4096)
    args = p.parse_args()
    if not torch.cuda.is_available() or not FusedLoRAKernel.is_available():
        raise SystemExit("needs a CUDA GPU and Triton (AS_FUSED_KERNEL must not be 0)")
    run_crossover(args.output_dir, args.hardware, args.ranks, args.n_tokens, args.n_reps,
                  args.warmup, args.d_model)


if __name__ == "__main__":
    main()
