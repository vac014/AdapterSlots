"""
whittle_dispatch_overhead_scaling.py -- K-Scaling Overhead Experiment (whittle_scheduler, §5.2)

Verifies that Whittle dispatcher overhead is O(K) and remains < 0.5 ms at K ≤ 50.

Sweeps K ∈ {4, 8, 16, 32, 64, 128} and measures wall-clock time of:
    - compute_indices() (Whittle)
    - rank_adapters()   (Whittle, includes sort)
    - rank_adapters()   (Threshold, sort-only baseline)

Fits a linear regression (overhead ∝ K) and reports R².

Cross-hardware usage -- same script; hardware determined by tau-iter-ms label:

  Single RTX A6000:
    python scripts/experiments/whittle_dispatch_overhead_scaling.py \\
        --hardware-label a6000_single \\
        --output results/whittle_scheduler/a6000_single/overhead_scaling.csv

  Two RTX A6000 PCIe (TP=2):
    python scripts/experiments/whittle_dispatch_overhead_scaling.py \\
        --hardware-label two_a6000_pcie \\
        --tau-iter-ms 100 \\
        --output results/whittle_scheduler/two_a6000_pcie/overhead_k16_tp2.csv

  Two H100 NVLink (TP=2):
    python scripts/experiments/whittle_dispatch_overhead_scaling.py \\
        --hardware-label two_h100_nvlink \\
        --tau-iter-ms 5 \\
        --output results/whittle_scheduler/two_h100_nvlink/update_rate_experiment.csv

Outputs CSV with columns:
    hardware_label, K, tau_iter_ms, method,
    overhead_mean_ms, overhead_std_ms, overhead_p99_ms,
    lt_0p5ms, r2_linear
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from adapter_slots.dispatch.whittle import WhittleDispatcher


def _benchmark_overhead(
    K: int,
    W: int,
    delta_t_s: float,
    n_reps: int = 500,
    warmup: int = 50,
) -> Dict[str, float]:
    """Measure compute_indices() and rank_adapters() overhead for K adapters."""
    adapters = [str(k) for k in range(K)]
    wd = WhittleDispatcher(adapters, warp_size=W, delta_t=delta_t_s)

    # Synthetic fill fracs and lambda estimates
    rng = np.random.default_rng(42)
    fill_fracs = {k: float(rng.uniform(0.0, 1.0)) for k in adapters}
    lambda_est = {k: float(rng.uniform(0.5, 20.0)) for k in adapters}

    # Warmup
    for _ in range(warmup):
        wd.compute_indices(fill_fracs, lambda_est)

    # Benchmark compute_indices
    ci_times = []
    for _ in range(n_reps):
        t0 = time.monotonic()
        wd.compute_indices(fill_fracs, lambda_est)
        ci_times.append((time.monotonic() - t0) * 1000.0)  # ms

    # Benchmark rank_adapters (includes sort)
    for _ in range(warmup):
        wd.rank_adapters(fill_fracs, lambda_est)

    ra_times = []
    for _ in range(n_reps):
        t0 = time.monotonic()
        wd.rank_adapters(fill_fracs, lambda_est)
        ra_times.append((time.monotonic() - t0) * 1000.0)  # ms

    # Threshold baseline: sort by fill frac (O(K log K), no index computation)
    thresh_times = []
    for _ in range(warmup):
        sorted(adapters, key=lambda k: fill_fracs[k], reverse=True)

    for _ in range(n_reps):
        t0 = time.monotonic()
        sorted(adapters, key=lambda k: fill_fracs[k], reverse=True)
        thresh_times.append((time.monotonic() - t0) * 1000.0)  # ms

    ci_arr = np.array(ci_times)
    ra_arr = np.array(ra_times)
    th_arr = np.array(thresh_times)

    return {
        "compute_indices_mean_ms": float(np.mean(ci_arr)),
        "compute_indices_std_ms": float(np.std(ci_arr)),
        "compute_indices_p99_ms": float(np.percentile(ci_arr, 99)),
        "rank_adapters_mean_ms": float(np.mean(ra_arr)),
        "rank_adapters_std_ms": float(np.std(ra_arr)),
        "rank_adapters_p99_ms": float(np.percentile(ra_arr, 99)),
        "threshold_mean_ms": float(np.mean(th_arr)),
        "threshold_std_ms": float(np.std(th_arr)),
        "threshold_p99_ms": float(np.percentile(th_arr, 99)),
    }


def _fit_linear_r2(x_vals: List[float], y_vals: List[float]) -> float:
    """Fit y = a*x + b and return R²."""
    if len(x_vals) < 2:
        return float("nan")
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-12:
        return 1.0
    return 1.0 - ss_res / ss_tot


def run_overhead_scaling(
    k_values: List[int],
    tau_iter_ms: float,
    hardware_label: str,
    n_reps: int,
    output_path: str,
) -> None:
    W = 32
    delta_t_s = tau_iter_ms / 1000.0

    print(f"\nK-scaling overhead experiment")
    print(f"  Hardware : {hardware_label}")
    print(f"  K values : {k_values}")
    print(f"  τ_iter   : {tau_iter_ms} ms")
    print(f"  Reps     : {n_reps}")
    print()

    rows = []
    k_list: List[float] = []
    whittle_means: List[float] = []
    threshold_means: List[float] = []

    for K in k_values:
        print(f"  Benchmarking K={K} ...", flush=True)
        metrics = _benchmark_overhead(K=K, W=W, delta_t_s=delta_t_s, n_reps=n_reps)

        whittle_ok = metrics["rank_adapters_mean_ms"] < 0.5
        threshold_ok = metrics["threshold_mean_ms"] < 0.5

        k_list.append(float(K))
        whittle_means.append(metrics["rank_adapters_mean_ms"])
        threshold_means.append(metrics["threshold_mean_ms"])

        for method, mean_key, std_key, p99_key in [
            ("whittle_compute_indices", "compute_indices_mean_ms",
             "compute_indices_std_ms", "compute_indices_p99_ms"),
            ("whittle_rank_adapters",   "rank_adapters_mean_ms",
             "rank_adapters_std_ms",   "rank_adapters_p99_ms"),
            ("threshold_sort",          "threshold_mean_ms",
             "threshold_std_ms",        "threshold_p99_ms"),
        ]:
            rows.append({
                "hardware_label": hardware_label,
                "K": K,
                "tau_iter_ms": tau_iter_ms,
                "method": method,
                "overhead_mean_ms": f"{metrics[mean_key]:.5f}",
                "overhead_std_ms": f"{metrics[std_key]:.5f}",
                "overhead_p99_ms": f"{metrics[p99_key]:.5f}",
                "lt_0p5ms": "Yes" if metrics[mean_key] < 0.5 else "No",
                "r2_linear": "",  # filled after loop
            })

        print(
            f"    Whittle rank_adapters : {metrics['rank_adapters_mean_ms']:.4f} ms mean  "
            f"({'<' if whittle_ok else '>='} 0.5ms)"
        )
        print(
            f"    Threshold sort        : {metrics['threshold_mean_ms']:.4f} ms mean  "
            f"({'<' if threshold_ok else '>='} 0.5ms)"
        )

    # Compute R² for Whittle rank_adapters and threshold sort vs. K
    r2_whittle = _fit_linear_r2(k_list, whittle_means)
    r2_threshold = _fit_linear_r2(k_list, threshold_means)

    print(f"\n  Linear fit R² (Whittle rank_adapters vs. K)  : {r2_whittle:.4f}")
    print(f"  Linear fit R² (Threshold sort vs. K)         : {r2_threshold:.4f}")
    print(f"  Pass condition (R² ≥ 0.95): "
          f"{'PASS' if r2_whittle >= 0.95 and r2_threshold >= 0.95 else 'FAIL'}")

    # Back-fill R² into rows
    for row in rows:
        if row["method"] == "whittle_rank_adapters":
            row["r2_linear"] = f"{r2_whittle:.4f}"
        elif row["method"] == "threshold_sort":
            row["r2_linear"] = f"{r2_threshold:.4f}"
        else:
            row["r2_linear"] = "N/A"

    # Write CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Written {len(rows)} rows → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E8 K-scaling overhead: verify Whittle O(K) < 0.5ms at K ≤ 50"
    )
    parser.add_argument("--k-values", nargs="+", type=int,
                        default=[4, 8, 16, 32, 64, 128],
                        help="K values to sweep (default: 4 8 16 32 64 128)")
    parser.add_argument("--tau-iter-ms", type=float, default=30.0,
                        help="Iteration time ms; used to set delta_t (default: 30)")
    parser.add_argument("--hardware-label", type=str, default="a6000_single",
                        help="Hardware label for output (default: a6000_single)")
    parser.add_argument("--n-reps", type=int, default=500,
                        help="Benchmark repetitions per K (default: 500)")
    parser.add_argument("--output", type=str,
                        default="results/whittle_scheduler/a6000_single/overhead_scaling.csv",
                        help="Output CSV path")

    args = parser.parse_args()

    run_overhead_scaling(
        k_values=args.k_values,
        tau_iter_ms=args.tau_iter_ms,
        hardware_label=args.hardware_label,
        n_reps=args.n_reps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
