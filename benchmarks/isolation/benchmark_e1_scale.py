#!/usr/bin/env python3
"""
benchmark_e1_scale.py -- N and K sweep for E1 isolation experiment.

Runs conditions A and D at every (N, K) point, measuring throughput and the
A→D gap.  Supports both multi-GPU setups via --tensor-parallel-size 2.

Outputs
-------
Columns: N, K, condition, n_runs, mean_tok_s, std_tok_s, p_ad, gap_pct, wall_s

Usage
-----
    # Two RTX A6000 PCIe (TP=2, CUDA_VISIBLE_DEVICES=0,1):
    python benchmarks/isolation/benchmark_e1_scale.py \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --N-values 512 1024 2048 \\
        --K-values 2 4 8 16 \\
        --n-runs 100 \\
        --warmup 10 \\
        --tensor-parallel-size 2 \\
        --output results/e1/scale/scale_sweep_two_a6000_pcie.csv

    # Two H100 NVLink (TP=2, CUDA_VISIBLE_DEVICES=0,1):
    python benchmarks/isolation/benchmark_e1_scale.py \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --N-values 512 1024 2048 4096 \\
        --K-values 2 4 8 16 \\
        --n-runs 200 \\
        --warmup 20 \\
        --tensor-parallel-size 2 \\
        --output results/e1/scale/scale_sweep_two_h100_nvlink.csv

    # 70B model on Two H100 NVLink (TP=2, 2x80GB=160GB fits 70B FP16 ~140GB):
    python benchmarks/isolation/benchmark_e1_scale.py \\
        --model ./models/llama-70b \\
        --adapter-dir ./adapters \\
        --N-values 512 1024 2048 \\
        --K-values 2 4 8 \\
        --n-runs 100 \\
        --warmup 10 \\
        --tensor-parallel-size 2 \\
        --output results/e1/scale/scale_sweep_70b_h100_nvlink.csv
"""

import argparse
import csv
import itertools
import math
import os
import subprocess
import sys
import time
from pathlib import Path


# GPU clock locking

def _gpu_ids_from_env(tp: int) -> list:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        ids = [int(x) for x in cvd.split(",") if x.strip().lstrip("-").isdigit()]
        return ids[:tp] if tp > 1 else ids[:1]
    return list(range(tp))


def _query_max_clock(gpu_id: int) -> int:
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


def _lock_clocks(gpu_ids: list, freq: int) -> None:
    for gid in gpu_ids:
        r = subprocess.run(
            ["nvidia-smi", "-lgc", str(freq), "-i", str(gid)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  [clock-lock] WARNING: GPU {gid} lock failed: {r.stderr.strip()[:120]}")
        else:
            print(f"  [clock-lock] GPU {gid} locked to {freq} MHz")


def _unlock_clocks(gpu_ids: list) -> None:
    for gid in gpu_ids:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(gid)],
                       capture_output=True, text=True)
        print(f"  [clock-lock] GPU {gid} clocks reset to default")


# Helpers

def _run_one(
    model: str,
    adapter_dir: str,
    condition: str,
    N: int,
    K: int,
    n_runs: int,
    warmup: int,
    tp: int,
    tmp_out: Path,
) -> dict:
    """Invoke benchmark_e1.py for one (condition, N, K) point."""
    cmd = [
        sys.executable, "benchmarks/isolation/benchmark_e1.py",
        "--model", model,
        "--adapter-dir", adapter_dir,
        "--condition", condition,
        "--n-tokens", str(N),
        "--K", str(K),
        "--n-runs", str(n_runs),
        "--warmup", str(warmup),
        "--output", str(tmp_out),
    ]
    if tp > 1:
        cmd += ["--tensor-parallel-size", str(tp)]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[-400:])

    rows = list(csv.DictReader(tmp_out.open()))
    if not rows:
        raise RuntimeError(f"benchmark_e1.py wrote empty CSV to {tmp_out}")

    tok_s = [float(r["tok_s"]) for r in rows]
    mean = sum(tok_s) / len(tok_s)
    variance = sum((x - mean) ** 2 for x in tok_s) / len(tok_s)
    std = math.sqrt(variance)

    return {
        "N": N,
        "K": K,
        "condition": condition,
        "n_runs": len(tok_s),
        "mean_tok_s": round(mean, 2),
        "std_tok_s": round(std, 2),
        "wall_s": round(elapsed, 1),
    }


def _ttest_p(a_vals: list[float], d_vals: list[float]) -> float:
    """Two-sample Welch t-test p-value (approximation without scipy)."""
    if not a_vals or not d_vals:
        return float("nan")
    na, nd = len(a_vals), len(d_vals)
    ma = sum(a_vals) / na
    md = sum(d_vals) / nd
    va = sum((x - ma) ** 2 for x in a_vals) / max(na - 1, 1)
    vd = sum((x - md) ** 2 for x in d_vals) / max(nd - 1, 1)
    se = math.sqrt(va / na + vd / nd)
    if se == 0:
        return 1.0
    t = abs(ma - md) / se
    # Rough two-tailed approximation via normal CDF tail (valid for large n)
    # P(|Z| > t) ≈ 2 * (1 - Φ(t))
    # Use Horner's approximation for standard normal CDF.
    def _norm_cdf(x: float) -> float:
        x = abs(x)
        t_ = 1.0 / (1.0 + 0.2316419 * x)
        poly = t_ * (0.319381530 + t_ * (-0.356563782 + t_ * (
               1.781477937 + t_ * (-1.821255978 + t_ * 1.330274429))))
        return 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return 2.0 * (1.0 - _norm_cdf(t))


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--N-values", nargs="+", type=int,
                        default=[512, 1024, 2048, 4096],
                        help="Total token counts to sweep")
    parser.add_argument("--K-values", nargs="+", type=int,
                        default=[2, 4, 8, 16],
                        help="Number of adapters to sweep")
    parser.add_argument("--conditions", nargs="+", default=["A", "D"],
                        choices=["A", "B", "C", "D"],
                        help="E1 conditions to run (default: A D)")
    parser.add_argument("--n-runs", type=int, default=200,
                        help="Timed runs per (condition, N, K) point")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="GPU tensor-parallel degree (1 = single GPU)")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--lock-clocks", action="store_true",
                        help="Lock GPU graphics clocks for the entire sweep to prevent "
                             "mid-benchmark frequency steps. Automatically enabled when "
                             "--tensor-parallel-size > 1. Calls nvidia-smi -lgc.")
    parser.add_argument("--clock-freq", type=int, default=None,
                        help="Graphics clock frequency in MHz. Default: auto-query GPU max.")
    args = parser.parse_args()

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = outpath.parent / "_tmp_scale"
    tmpdir.mkdir(exist_ok=True)

    combos = list(itertools.product(args.N_values, args.K_values, args.conditions))
    print(
        f"Scale sweep: {len(combos)} points\n"
        f"  N ∈ {args.N_values}\n"
        f"  K ∈ {args.K_values}\n"
        f"  conditions = {args.conditions}\n"
        f"  n_runs={args.n_runs}  warmup={args.warmup}  TP={args.tensor_parallel_size}\n"
    )

    # Lock GPU clocks for the entire sweep (not per-point) so clock state is
    # stable across all (N, K, condition) combinations. benchmark_e1.py is called
    # as a subprocess here, so we do NOT pass --lock-clocks to it -- clocks are
    # already held by this parent process.
    lock_clocks = args.lock_clocks or (args.tensor_parallel_size > 1)
    gpu_ids = _gpu_ids_from_env(args.tensor_parallel_size)
    if lock_clocks:
        freq = args.clock_freq or _query_max_clock(gpu_ids[0])
        print(f"[clock-lock] Locking GPUs {gpu_ids} to {freq} MHz for the sweep ...")
        _lock_clocks(gpu_ids, freq)

    # Collect raw tok_s lists per (N, K, cond) for p-value computation later.
    raw: dict[tuple, list[float]] = {}
    summary: list[dict] = []

    try:
        for i, (N, K, cond) in enumerate(combos, 1):
            tmp = tmpdir / f"tmp_N{N}_K{K}_{cond}.csv"
            print(f"[{i:>3}/{len(combos)}] N={N:>5}  K={K:>2}  cond={cond}  ... ",
                  end="", flush=True)
            try:
                row = _run_one(
                    args.model, args.adapter_dir, cond,
                    N, K, args.n_runs, args.warmup,
                    args.tensor_parallel_size, tmp,
                )
                # Re-read raw values for p-value computation.
                raw[(N, K, cond)] = [float(r["tok_s"])
                                     for r in csv.DictReader(tmp.open())]
                summary.append(row)
                print(f"{row['mean_tok_s']:>8.1f} ± {row['std_tok_s']:.1f} tok/s  "
                      f"({row['wall_s']:.0f}s wall)")
            except Exception as exc:
                print(f"ERROR: {exc}")
                summary.append({
                    "N": N, "K": K, "condition": cond,
                    "n_runs": 0, "mean_tok_s": float("nan"),
                    "std_tok_s": float("nan"), "wall_s": 0,
                })

        # Augment with p-value and gap_pct relative to condition A at same (N, K).
        fieldnames = ["N", "K", "condition", "n_runs",
                      "mean_tok_s", "std_tok_s", "p_ad", "gap_pct", "wall_s"]

        final: list[dict] = []
        for row in summary:
            N, K, cond = row["N"], row["K"], row["condition"]
            a_vals = raw.get((N, K, "A"), [])
            d_vals = raw.get((N, K, "D"), [])
            p_ad = _ttest_p(a_vals, d_vals)
            a_mean = (sum(a_vals) / len(a_vals)) if a_vals else float("nan")
            d_mean = (sum(d_vals) / len(d_vals)) if d_vals else float("nan")
            gap_pct = (
                round(100.0 * (a_mean - d_mean) / a_mean, 2)
                if a_mean and not math.isnan(a_mean)
                else float("nan")
            )
            final.append({**row,
                          "p_ad": round(p_ad, 6) if not math.isnan(p_ad) else "nan",
                          "gap_pct": gap_pct})

        with outpath.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(final)

        print(f"\nWritten: {outpath}  ({len(final)} rows)")

        # Print A→D gap summary table.
        print("\nA→D throughput gap summary:")
        print(f"{'N':>6} {'K':>3}  {'A (tok/s)':>12} {'D (tok/s)':>12} "
              f"{'gap%':>7}  {'p-value':>9}")
        by_nk: dict[tuple, dict] = {}
        for row in final:
            by_nk.setdefault((row["N"], row["K"]), {})[row["condition"]] = row
        for (N, K), conds in sorted(by_nk.items()):
            if "A" in conds and "D" in conds:
                a_r, d_r = conds["A"], conds["D"]
                print(f"{N:>6} {K:>3}  "
                      f"{a_r['mean_tok_s']:>12.1f} {d_r['mean_tok_s']:>12.1f} "
                      f"{a_r['gap_pct']:>7.1f}%  {a_r['p_ad']:>9}")

    finally:
        if lock_clocks:
            print("\n[clock-lock] Releasing clock locks ...")
            _unlock_clocks(gpu_ids)


if __name__ == "__main__":
    main()
