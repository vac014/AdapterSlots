"""
run_sota.py -- Orchestrates the SOTA comparison experiments B1 through B6.

Compares AdapterSlots C7 against Punica, S-LoRA, dLoRA, and vanilla vLLM across
throughput, latency, K-scaling, adversarial patterns, SLO attainment, and
real traces.

Hardware: 1× RTX A6000 (TP=1) for all SOTA benchmarks.

Usage:
    # All SOTA experiments
    python scripts/run_sota.py \\
        --model ./models/llama-7b --adapter-dir ./adapters \\
        --output-dir results/sota_evaluation/sota/

    # Specific experiments
    python scripts/run_sota.py --which B1 B2 B3 \\
        --model ./models/llama-7b --adapter-dir ./adapters \\
        --output-dir results/sota_evaluation/sota/

    # Dry-run
    python scripts/run_sota.py --dry-run --output-dir results/sota_evaluation/sota/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

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


def _run_decode(args_list: List[str], dry_run: bool = False) -> int:
    cmd = [_python(), str(_ROOT / "bench_decode_only.py")] + args_list
    if dry_run:
        cmd.append("--dry-run")
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def _run_traces(args_list: List[str], dry_run: bool = False) -> int:
    cmd = [_python(), str(_ROOT / "bench_real_traces.py")] + args_list
    if dry_run:
        cmd.append("--dry-run")
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def _skip_if_exists(path: str) -> bool:
    if Path(path).exists():
        print(f"  Skipping (result exists): {path}")
        return True
    return False


# All backends tested in SOTA comparisons
_BACKENDS = [
    ("adapterslots", "C7"),     # AdapterSlots full stack
    ("vllm", "C0"),     # vanilla vLLM baseline
    ("punica", "punica"),
    ("slora", "slora"),
    ("dlora", "dlora"),
]


def run_b1(model: str, adapter_dir: str, out_dir: str, rank: int, dry_run: bool) -> None:
    """B1: Throughput vs request rate (λ ∈ {1,3,5,7,10,15}, K=10, all backends)."""
    print("\n=== B1: Throughput vs rate curve ===")
    for rate in [1, 3, 5, 7, 10, 15]:
        for backend, mode in _BACKENDS:
            out = f"{out_dir}/B1/{backend}_{mode}_r{rate}.json"
            if _skip_if_exists(out):
                continue
            args = [
                "--backend", backend, "--mode", mode,
                "--model", model, "--adapter-dir", adapter_dir,
                "--num-adapters", "10", "--rank", str(rank),
                "--request-rate", str(rate), "--pattern", "zipf",
                "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
                "--output", out,
            ]
            if backend == "adapterslots":
                args += ["--tmax", "90", "--wgkp-threshold", "8"]
            _run_bench(args, dry_run=dry_run)


def run_b2(model: str, adapter_dir: str, out_dir: str, rank: int, dry_run: bool) -> None:
    """B2: TTFT latency distribution (λ=7, K=10, P50/P99 for all backends)."""
    print("\n=== B2: TTFT latency comparison (λ=7) ===")
    for backend, mode in _BACKENDS:
        out = f"{out_dir}/B2/{backend}_{mode}.json"
        if _skip_if_exists(out):
            continue
        args = [
            "--backend", backend, "--mode", mode,
            "--model", model, "--adapter-dir", adapter_dir,
            "--num-adapters", "10", "--rank", str(rank),
            "--request-rate", "7", "--pattern", "zipf",
            "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
            "--output", out,
        ]
        if backend == "adapterslots":
            args += ["--tmax", "90", "--wgkp-threshold", "8"]
        _run_bench(args, dry_run=dry_run)


def run_b3(model: str, adapter_dir: str, out_dir: str, rank: int, dry_run: bool) -> None:
    """B3: K-scale comparison (K ∈ {4,10,20,50}, λ=7, all backends)."""
    print("\n=== B3: K-scale comparison ===")
    for K in [4, 10, 20, 50]:
        for backend, mode in _BACKENDS:
            out = f"{out_dir}/B3/{backend}_{mode}_K{K}.json"
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


def run_b4(model: str, adapter_dir: str, out_dir: str, rank: int, dry_run: bool) -> None:
    """B4: Decode degradation (Distinct adversarial pattern, delegates to bench_decode_only.py)."""
    print("\n=== B4: Decode-phase degradation (Distinct pattern) ===")
    _run_decode([
        "--model", model, "--adapter-dir", adapter_dir,
        "--num-adapters", "4", "10", "20", "50",
        "--request-rates", "3", "7", "10",
        "--rank", str(rank),
        "--modes", "C0", "C3", "C7",
        "--reps", "3",
        "--tmax", "90", "--wgkp-threshold", "8",
        "--output-dir", f"{out_dir}/B4/",
    ], dry_run=dry_run)


def run_b5(model: str, adapter_dir: str, out_dir: str, rank: int, dry_run: bool) -> None:
    """B5: SLO attainment at SLO thresholds {500, 1000, 2000}ms, λ=7, K=10."""
    print("\n=== B5: SLO attainment ===")
    for backend, mode in _BACKENDS:
        out = f"{out_dir}/B5/{backend}_{mode}.json"
        if _skip_if_exists(out):
            continue
        args = [
            "--backend", backend, "--mode", mode,
            "--model", model, "--adapter-dir", adapter_dir,
            "--num-adapters", "10", "--rank", str(rank),
            "--request-rate", "7", "--pattern", "zipf",
            "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
            "--output", out,
        ]
        if backend == "adapterslots":
            args += ["--tmax", "90", "--wgkp-threshold", "8"]
        _run_bench(args, dry_run=dry_run)
    # Note: SLO thresholds (500/1000/2000ms) are applied at analysis time
    # by compute_stats.py filtering on ttft_p99_ms


def run_b6(
    model: str,
    adapter_dir: str,
    out_dir: str,
    rank: int,
    trace_path: Optional[str],
    trace_format: str,
    dry_run: bool,
) -> None:
    """B6: Real trace replay (delegates to bench_real_traces.py)."""
    print("\n=== B6: Real trace replay ===")
    args = [
        "--model", model, "--adapter-dir", adapter_dir,
        "--num-adapters", "10", "--rank", str(rank),
        "--reps", "3",
        "--output", f"{out_dir}/B6/traces.json",
    ]
    if trace_path:
        args += ["--trace-path", trace_path, "--trace-format", trace_format]
    _run_traces(args, dry_run=dry_run)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SOTA comparison orchestrator")
    ap.add_argument("--model", default="./models/llama-7b")
    ap.add_argument("--adapter-dir", default="./adapters")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--output-dir", default="results/sota_evaluation/sota")
    ap.add_argument(
        "--which",
        nargs="+",
        default=["B1", "B2", "B3", "B4", "B5", "B6"],
        choices=["B1", "B2", "B3", "B4", "B5", "B6"],
        help="Which SOTA experiments to run",
    )
    ap.add_argument("--trace-path", default=None,
                    help="Path to real trace for B6 (LMSYS/BurstGPT)")
    ap.add_argument("--trace-format", default="lmsys",
                    choices=["lmsys", "burstgpt"])
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out = args.output_dir

    print(f"run_sota.py -- experiments={args.which}")
    print(f"  model={args.model}  adapter_dir={args.adapter_dir}  rank={args.rank}")

    dispatch = {
        "B1": lambda: run_b1(args.model, args.adapter_dir, out, args.rank, args.dry_run),
        "B2": lambda: run_b2(args.model, args.adapter_dir, out, args.rank, args.dry_run),
        "B3": lambda: run_b3(args.model, args.adapter_dir, out, args.rank, args.dry_run),
        "B4": lambda: run_b4(args.model, args.adapter_dir, out, args.rank, args.dry_run),
        "B5": lambda: run_b5(args.model, args.adapter_dir, out, args.rank, args.dry_run),
        "B6": lambda: run_b6(args.model, args.adapter_dir, out, args.rank,
                              args.trace_path, args.trace_format, args.dry_run),
    }

    for name in args.which:
        dispatch[name]()

    print("\n=== SOTA comparison runs complete ===")
    print(f"  Results in: {out}/B{{1..6}}/")


if __name__ == "__main__":
    main()
