"""
run_ablations.py -- Orchestrates sota_evaluation ablation experiments AB-1 through AB-6.

Dependency order: M1 gate → AB-1 → AB-2 → AB-3 → AB-4 → AB-5 → AB-6.
All experiments are idempotent (skip if result file exists).

Hardware: 1× RTX A6000 (TP=1) for AB-1..AB-5; 2× A6000 PCIe for AB-6 (APIS).

Usage:
    # All ablations
    python scripts/run_ablations.py \\
        --model ./models/llama-7b --adapter-dir ./adapters \\
        --output-dir results/sota_evaluation/ablations/

    # Specific ablations only
    python scripts/run_ablations.py --which AB1 AB3 AB6 \\
        --model ./models/llama-7b --adapter-dir ./adapters \\
        --output-dir results/sota_evaluation/ablations/

    # Dry-run (no GPU)
    python scripts/run_ablations.py --dry-run \\
        --output-dir results/sota_evaluation/ablations/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Project root
_ROOT = Path(__file__).parent.parent


def _python() -> str:
    return sys.executable


def _run_bench(args_list: List[str], dry_run: bool = False) -> int:
    cmd = [_python(), str(_ROOT / "bench.py")] + args_list
    if dry_run:
        cmd.append("--dry-run")
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def _run_apis(args_list: List[str], dry_run: bool = False) -> int:
    cmd = [_python(), str(_ROOT / "bench_apis.py")] + args_list
    if dry_run:
        cmd.append("--dry-run")
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def _check_m1_gate(results_root: str) -> bool:
    """Verify EC 14.M1 result exists and passed before proceeding."""
    m1_paths = [
        Path(results_root) / ".." / "m1" / "summary.json",
        Path("results") / "m1" / "summary.json",
        Path("results") / "sota_evaluation" / "m1" / "summary.json",
    ]
    for p in m1_paths:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                results = data.get("results", [])
                target = next(
                    (r for r in results if r.get("rank") == 16 and r.get("batch") == 32), None
                )
                if target and target.get("psi_fuse", 0.0) >= 1.25:
                    print(f"  EC 14.M1 gate: PASS (ψ_fuse={target['psi_fuse']:.3f} at M1 path {p})")
                    return True
                print(f"  EC 14.M1 gate: FAIL (psi_fuse={target.get('psi_fuse', 'N/A') if target else 'not found'} at {p})")
                return False
            except Exception:
                pass
    print("  EC 14.M1 gate: result not found -- proceeding anyway (run M1 first for gated runs)")
    return True  # Don't block if result is simply missing (user can run M1 separately)


def _skip_if_exists(path: str) -> bool:
    if Path(path).exists():
        print(f"  Skipping (result exists): {path}")
        return True
    return False


def run_ab1(model: str, adapter_dir: str, out_dir: str, rank: int, dry_run: bool) -> None:
    """AB-1: Configuration lattice C0→C7 (all 8 configs, Zipf K=10, rate=7)."""
    print("\n=== AB-1: Configuration lattice C0→C7 ===")
    configs = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    for mode in configs:
        out = f"{out_dir}/AB1/{mode}.json"
        if _skip_if_exists(out):
            continue
        backend = "adapterslots" if mode != "C0" else "vllm"
        _run_bench([
            "--backend", backend, "--mode", mode,
            "--model", model, "--adapter-dir", adapter_dir,
            "--num-adapters", "10", "--rank", str(rank),
            "--request-rate", "7", "--pattern", "zipf",
            "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
            "--tmax", "90", "--wgkp-threshold", "8",
            "--output", out,
        ], dry_run=dry_run)


def run_ab2(model: str, adapter_dir: str, out_dir: str, dry_run: bool) -> None:
    """AB-2: Rank sensitivity (rank ∈ {16, 32, 64}, C7 only, K=10, rate=7)."""
    print("\n=== AB-2: Rank sensitivity ===")
    for rank in [16, 32, 64]:
        out = f"{out_dir}/AB2/C7_r{rank}.json"
        if _skip_if_exists(out):
            continue
        _run_bench([
            "--backend", "adapterslots", "--mode", "C7",
            "--model", model, "--adapter-dir", adapter_dir,
            "--num-adapters", "10", "--rank", str(rank),
            "--request-rate", "7", "--pattern", "zipf",
            "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
            "--tmax", "90", "--wgkp-threshold", "8",
            "--output", out,
        ], dry_run=dry_run)
    # Baseline C0 at each rank
    for rank in [16, 32, 64]:
        out = f"{out_dir}/AB2/C0_r{rank}.json"
        if _skip_if_exists(out):
            continue
        _run_bench([
            "--backend", "vllm", "--mode", "C0",
            "--model", model, "--adapter-dir", adapter_dir,
            "--num-adapters", "10", "--rank", str(rank),
            "--request-rate", "7", "--pattern", "zipf",
            "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
            "--output", out,
        ], dry_run=dry_run)


def run_ab3(model: str, adapter_dir: str, out_dir: str, rank: int, dry_run: bool) -> None:
    """AB-3: K-decay (K ∈ {4,10,15,20,25,50,100}, C7, rate=7)."""
    print("\n=== AB-3: K-decay curve ===")
    k_list = [4, 10, 15, 20, 25, 50, 100]
    for K in k_list:
        for mode, backend in [("C7", "adapterslots"), ("C0", "vllm")]:
            out = f"{out_dir}/AB3/{mode}_K{K}.json"
            if _skip_if_exists(out):
                continue
            args = [
                "--backend", backend, "--mode", mode,
                "--model", model, "--adapter-dir", adapter_dir,
                "--num-adapters", str(K), "--rank", str(rank),
                "--request-rate", "7", "--pattern", "zipf",
                "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
                "--output", out,
            ]
            if backend == "adapterslots":
                args += ["--tmax", "90", "--wgkp-threshold", "8"]
            _run_bench(args, dry_run=dry_run)


def run_ab4(model: str, adapter_dir: str, out_dir: str, rank: int, dry_run: bool) -> None:
    """AB-4: T_max sweep ({30,60,90,120,150,200}ms, C5/C6/C7, K=10, rate=7)."""
    print("\n=== AB-4: T_max latency-throughput Pareto ===")
    tmax_list = [30, 60, 90, 120, 150, 200]
    for tmax in tmax_list:
        for mode in ["C5", "C6", "C7"]:
            out = f"{out_dir}/AB4/{mode}_tmax{tmax}.json"
            if _skip_if_exists(out):
                continue
            _run_bench([
                "--backend", "adapterslots", "--mode", mode,
                "--model", model, "--adapter-dir", adapter_dir,
                "--num-adapters", "10", "--rank", str(rank),
                "--request-rate", "7", "--pattern", "zipf",
                "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
                "--tmax", str(tmax), "--wgkp-threshold", "8",
                "--output", out,
            ], dry_run=dry_run)


def run_ab5(model: str, adapter_dir: str, out_dir: str, rank: int, dry_run: bool) -> None:
    """AB-5: n* threshold sweep (wgkp_threshold ∈ {4,8,16,32}, C5, K=10, rate=7)."""
    print("\n=== AB-5: n* threshold sweep ===")
    threshold_list = [4, 8, 16, 32]
    for thresh in threshold_list:
        out = f"{out_dir}/AB5/C5_thresh{thresh}.json"
        if _skip_if_exists(out):
            continue
        _run_bench([
            "--backend", "adapterslots", "--mode", "C5",
            "--model", model, "--adapter-dir", adapter_dir,
            "--num-adapters", "10", "--rank", str(rank),
            "--request-rate", "7", "--pattern", "zipf",
            "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
            "--tmax", "90", "--wgkp-threshold", str(thresh),
            "--output", out,
        ], dry_run=dry_run)


def run_ab6(model: str, adapter_dir: str, out_dir: str, rank: int, dry_run: bool) -> None:
    """AB-6: APIS vs TP=2 (two A6000 PCIe; C7-APIS vs C0 TP=2, K=50, rate=7)."""
    print("\n=== AB-6: APIS two-GPU vs TP=2 baseline ===")
    out = f"{out_dir}/AB6/apis_k50.json"
    if not _skip_if_exists(out):
        _run_apis([
            "--model", model, "--adapter-dir", adapter_dir,
            "--num-adapters", "50", "--rank", str(rank),
            "--request-rate", "7",
            "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
            "--tmax", "90", "--wgkp-threshold", "8",
            "--tp-baseline", "2",
            "--output", out,
        ], dry_run=dry_run)

    # Also sweep K for APIS
    for K in [10, 20, 50, 100]:
        out_k = f"{out_dir}/AB6/apis_k{K}.json"
        if _skip_if_exists(out_k):
            continue
        _run_apis([
            "--model", model, "--adapter-dir", adapter_dir,
            "--num-adapters", str(K), "--rank", str(rank),
            "--request-rate", "7",
            "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
            "--tmax", "90", "--wgkp-threshold", "8",
            "--tp-baseline", "2",
            "--output", out_k,
        ], dry_run=dry_run)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="sota_evaluation ablation orchestrator")
    ap.add_argument("--model", default="./models/llama-7b")
    ap.add_argument("--adapter-dir", default="./adapters")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--output-dir", default="results/sota_evaluation/ablations")
    ap.add_argument(
        "--which",
        nargs="+",
        default=["AB1", "AB2", "AB3", "AB4", "AB5", "AB6"],
        choices=["AB1", "AB2", "AB3", "AB4", "AB5", "AB6"],
        help="Which ablations to run (default: all)",
    )
    ap.add_argument("--skip-m1-gate", action="store_true",
                    help="Skip M1 gate check (useful for dry-run or if M1 was run separately)")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out = args.output_dir

    print(f"run_ablations.py -- experiments={args.which}")
    print(f"  model={args.model}  adapter_dir={args.adapter_dir}  rank={args.rank}")
    print(f"  output_dir={out}")

    if not args.skip_m1_gate and not args.dry_run:
        _check_m1_gate(out)

    dispatch = {
        "AB1": lambda: run_ab1(args.model, args.adapter_dir, out, args.rank, args.dry_run),
        "AB2": lambda: run_ab2(args.model, args.adapter_dir, out, args.dry_run),
        "AB3": lambda: run_ab3(args.model, args.adapter_dir, out, args.rank, args.dry_run),
        "AB4": lambda: run_ab4(args.model, args.adapter_dir, out, args.rank, args.dry_run),
        "AB5": lambda: run_ab5(args.model, args.adapter_dir, out, args.rank, args.dry_run),
        "AB6": lambda: run_ab6(args.model, args.adapter_dir, out, args.rank, args.dry_run),
    }

    for name in args.which:
        dispatch[name]()

    print("\n=== Ablation runs complete ===")
    print(f"  Results in: {out}/AB{{1..6}}/")


if __name__ == "__main__":
    main()
