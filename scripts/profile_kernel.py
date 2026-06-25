"""
profile_kernel.py -- Wrapper that launches ncu / nsys around a target script.

Usage:
    # Profile with Nsight Compute (hardware counters)
    python scripts/profile_kernel.py ncu \
        --condition D \
        --model ./models/llama-7b \
        --adapter-dir ./adapters \
        --n-kernels 50 \
        --output-dir results/infrastructure/ncu

    # Profile with Nsight Systems (timeline)
    python scripts/profile_kernel.py nsys \
        --condition A \
        --model ./models/llama-7b \
        --output-dir results/infrastructure/nsys

This script builds the correct ncu/nsys command-line and invokes isolation_batch_conditions.py as the target.

Requirements:
    - ncu must be callable (sudo ncu or user in nvperfmon group)
    - nsys must be on PATH
    - CUDA driver ≥ 525
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


NCU_METRICS = ",".join([
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l2tex__t_sector_hit_rate.pct",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__sass_thread_inst_executed_op_ldsm_pred_on.avg",
    "sm__warps_eligible.avg.pct_of_peak_sustained_active",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "launch__grid_size",
    "launch__block_size",
])


def parse_args():
    p = argparse.ArgumentParser(description="ncu/nsys profiling wrapper")
    p.add_argument("profiler", choices=["ncu", "nsys"],
                   help="Which profiler to use")
    p.add_argument("--condition", type=str, choices=["A", "B", "C", "D"], default="D",
                   help="E1 condition to profile")
    p.add_argument("--model", type=str, default=None,
                   help="Model path. If omitted, runs decomp-only (--profile-decomp)")
    p.add_argument("--adapter-dir", type=str, default="./adapters")
    p.add_argument("--N", type=int, default=512)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--n-kernels", type=int, default=50,
                   help="ncu: number of kernel launches to profile")
    p.add_argument("--output-dir", type=str, default="results/infrastructure/ncu",
                   help="Directory to write profiler output")
    p.add_argument("--kernel-name", type=str, default="sgmv_cutlass",
                   help="ncu: kernel name filter (substring match)")
    p.add_argument("--target-script", type=str,
                   default="scripts/experiments/isolation_batch_conditions.py",
                   help="Python script to profile")
    p.add_argument("--sudo", action="store_true",
                   help="Prefix command with sudo (needed for ncu on some systems)")
    return p.parse_args()


def build_ncu_command(args, target_args: list) -> list:
    """Build the ncu command list."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir, f"ncu_condition_{args.condition}_{ts}.csv"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    cmd = []
    if args.sudo:
        cmd.append("sudo")
    cmd += [
        "ncu",
        "--metrics", NCU_METRICS,
        "--kernel-name", args.kernel_name,
        "--launch-count", str(args.n_kernels),
        "--csv",
        "--log-file", output_file,
        sys.executable,
    ] + target_args

    return cmd, output_file


def build_nsys_command(args, target_args: list) -> list:
    """Build the nsys command list."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = os.path.join(
        args.output_dir, f"nsys_condition_{args.condition}_{ts}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx,osrt",
        "--output", output_prefix,
        "--force-overwrite", "true",
        sys.executable,
    ] + target_args

    return cmd, output_prefix + ".nsys-rep"


def build_target_args(args) -> list:
    """Build the arguments to pass to the target script (isolation_batch_conditions.py)."""
    target_args = [
        args.target_script,
        "--condition", args.condition,
        "--N", str(args.N),
        "--K", str(args.K),
        "--n-runs", "50",
    ]
    if args.model:
        target_args += ["--model", args.model, "--adapter-dir", args.adapter_dir]
    else:
        # No model: just time decomposition
        pass

    return target_args


def main():
    args = parse_args()
    target_args = build_target_args(args)

    if args.profiler == "ncu":
        cmd, output_file = build_ncu_command(args, target_args)
        print(f"[profile_kernel] Running ncu ...")
        print(f"  Output -> {output_file}")
    else:
        cmd, output_file = build_nsys_command(args, target_args)
        print(f"[profile_kernel] Running nsys ...")
        print(f"  Output -> {output_file}")

    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Profiler exited with code {result.returncode}")
        print("Common fixes:")
        print("  ncu permission: sudo ncu, or add user to nvperfmon group")
        print("  nsys not found: install Nsight Systems")
        sys.exit(result.returncode)

    print(f"\n[done] Profiler output: {output_file}")


if __name__ == "__main__":
    main()
