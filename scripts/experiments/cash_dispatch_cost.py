"""
cash_dispatch_cost.py -- per-tick cost of CASH dispatch against the number of adapters K.

For K in {4, 8, 16, 32, 64, 128} it times, on the CPU:
    cash_compute_indices  the CASH scores I_k = lambda_k s_k P_k(s_k)
    cash_rank             scores and the sort into dispatch order
    threshold_sort        a sort by queue length alone, for reference
and reports the R^2 of a linear fit of cost against K.

Usage:
    python scripts/experiments/cash_dispatch_cost.py \\
        --output results/cash_dispatch/a6000_single/overhead_scaling.csv

Output columns:
    hardware_label, K, method, overhead_mean_ms, overhead_std_ms, overhead_p99_ms, r2_linear
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adapter_slots.dispatch.cash import CASHDispatcher  # noqa: E402


def _time_ms(fn, n_reps: int, warmup: int) -> np.ndarray:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return np.array(out)


def _benchmark(K: int, W: int, n_reps: int, warmup: int) -> Dict[str, np.ndarray]:
    """Timings for K adapters with random queue states."""
    rng = np.random.default_rng(42)
    adapters = [str(k) for k in range(K)]
    queue_lens = {k: int(rng.integers(0, W + 1)) for k in adapters}
    ages = {k: float(rng.uniform(0.0, 0.2)) for k in adapters}
    lambda_est = {k: float(rng.uniform(0.5, 300.0)) for k in adapters}
    deadlines = {k: 0.2 for k in adapters}
    cash = CASHDispatcher(warp_size=W)
    return {
        "cash_compute_indices": _time_ms(
            lambda: cash.compute_indices(queue_lens, ages, lambda_est, deadlines), n_reps, warmup),
        "cash_rank": _time_ms(
            lambda: cash.rank(cash.compute_indices(queue_lens, ages, lambda_est, deadlines)),
            n_reps, warmup),
        "threshold_sort": _time_ms(
            lambda: sorted(adapters, key=lambda k: queue_lens[k], reverse=True), n_reps, warmup),
    }


def _r2_linear(x: List[float], y: List[float]) -> float:
    xa, ya = np.array(x, dtype=float), np.array(y, dtype=float)
    pred = np.polyval(np.polyfit(xa, ya, 1), xa)
    ss_tot = float(np.sum((ya - ya.mean()) ** 2))
    return 1.0 if ss_tot < 1e-12 else 1.0 - float(np.sum((ya - pred) ** 2)) / ss_tot


def main() -> None:
    p = argparse.ArgumentParser(description="CASH dispatch cost per tick against K")
    p.add_argument("--k-values", nargs="+", type=int, default=[4, 8, 16, 32, 64, 128])
    p.add_argument("--hardware-label", default="a6000_single")
    p.add_argument("--n-reps", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--output", default="results/cash_dispatch/a6000_single/overhead_scaling.csv")
    args = p.parse_args()

    rows, means = [], {}
    for K in args.k_values:
        timings = _benchmark(K, 32, args.n_reps, args.warmup)
        for method, t in timings.items():
            means.setdefault(method, []).append(float(t.mean()))
            rows.append({"hardware_label": args.hardware_label, "K": K, "method": method,
                         "overhead_mean_ms": f"{t.mean():.5f}", "overhead_std_ms": f"{t.std():.5f}",
                         "overhead_p99_ms": f"{np.percentile(t, 99):.5f}", "r2_linear": ""})
        print(f"K={K:>4}  CASH rank {timings['cash_rank'].mean() * 1e3:7.1f} us   "
              f"threshold sort {timings['threshold_sort'].mean() * 1e3:6.1f} us")
    r2 = {m: _r2_linear(args.k_values, v) for m, v in means.items()}
    for row in rows:
        row["r2_linear"] = f"{r2[row['method']]:.4f}"
    print(f"linear fit R^2 against K: CASH rank {r2['cash_rank']:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
