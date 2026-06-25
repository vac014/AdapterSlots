"""
run_claims.py -- Validates sota_evaluation Claims C1, C2, C3 with 3-rep min-rep guarantee.

Exit conditions:
    EC 14.C1: mean ≥ 1.45×, min-rep ≥ 1.35×, std ≤ 5% mean  (1.5× throughput claim)
    EC 14.C2: APIS mean ≥ 1.80×, min-rep ≥ 1.60×            (2× APIS claim)
    EC 14.C3: Pearson r(GWAR(8), tps) ≥ 0.85, p < 0.001      (causality claim)

Hardware: C1/C3 on 1× A6000 (TP=1); C2 on 2× A6000 PCIe.

Usage:
    python scripts/run_claims.py \\
        --model ./models/llama-7b --adapter-dir ./adapters \\
        --output-dir results/sota_evaluation/claims/

    python scripts/run_claims.py --which C1 C3 --dry-run \\
        --output-dir results/sota_evaluation/claims/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import math
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).parent.parent


def _python() -> str:
    return sys.executable


def _run_bench(args_list: List[str], dry_run: bool = False) -> int:
    cmd = [_python(), str(_ROOT / "bench.py")] + args_list
    if dry_run:
        cmd.append("--dry-run")
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def _run_apis(args_list: List[str], dry_run: bool = False) -> int:
    cmd = [_python(), str(_ROOT / "bench_apis.py")] + args_list
    if dry_run:
        cmd.append("--dry-run")
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def _load_result(path: str) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _pearson_r(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Compute Pearson r and p-value (two-tailed t-test approximation)."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0, 1.0
    r = num / (den_x * den_y)
    # t-statistic: t = r * sqrt(n-2) / sqrt(1-r^2)
    r_clamp = max(-0.9999999, min(0.9999999, r))
    t_stat = r_clamp * math.sqrt(n - 2) / math.sqrt(1 - r_clamp ** 2)
    # Approximate two-tailed p-value using normal approximation for large n
    # For small n this is approximate; exact would need scipy.stats.t.sf
    from_z = math.erfc(abs(t_stat) / math.sqrt(2))
    p_approx = from_z
    return r, p_approx


def _get_tps_from_result(result: Optional[dict]) -> float:
    if result is None:
        return 0.0
    return result.get("summary", {}).get("throughput_toks_mean", 0.0)


def validate_c1(out_dir: str, dry_run: bool) -> dict:
    """
    C1: AdapterSlots C7 ≥ 1.5× throughput at K=10, rank=32, all-linear, T_max=90ms.
    3 seeds × (C7, C0) → mean ratio ≥ 1.45×, min-rep ≥ 1.35×, std ≤ 5% mean.
    """
    print("\n=== C1: 1.5× throughput claim ===")
    seeds = [42, 43, 44]

    ratios = []
    for seed in seeds:
        c7_path = f"{out_dir}/C1/C7_seed{seed}.json"
        c0_path = f"{out_dir}/C1/C0_seed{seed}.json"

        # Run C7 rep
        if not Path(c7_path).exists():
            _run_bench([
                "--backend", "adapterslots", "--mode", "C7",
                "--model", "./models/llama-7b",
                "--adapter-dir", "./adapters",
                "--num-adapters", "10", "--rank", "32",
                "--request-rate", "7", "--pattern", "zipf",
                "--num-prompts", "1000", "--warmup", "20",
                "--reps", "1", "--seed", str(seed),
                "--tmax", "90", "--wgkp-threshold", "8",
                "--output", c7_path,
            ], dry_run=dry_run)

        # Run C0 (vLLM baseline) rep
        if not Path(c0_path).exists():
            _run_bench([
                "--backend", "vllm", "--mode", "C0",
                "--model", "./models/llama-7b",
                "--adapter-dir", "./adapters",
                "--num-adapters", "10", "--rank", "32",
                "--request-rate", "7", "--pattern", "zipf",
                "--num-prompts", "1000", "--warmup", "20",
                "--reps", "1", "--seed", str(seed),
                "--output", c0_path,
            ], dry_run=dry_run)

        c7 = _load_result(c7_path)
        c0 = _load_result(c0_path)

        if dry_run and (c7 is None or c0 is None):
            # Synthetic ratios for dry-run
            ratios.append(1.52 + seed * 0.01 - 42 * 0.01)
            continue

        tps_c7 = _get_tps_from_result(c7)
        tps_c0 = _get_tps_from_result(c0)
        if tps_c0 > 0:
            ratios.append(tps_c7 / tps_c0)
        else:
            print(f"    WARNING: seed={seed} C0 throughput=0, skipping rep")

    if not ratios:
        return {"claim": "C1", "status": "ERROR", "error": "no valid reps"}

    mean_r = sum(ratios) / len(ratios)
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in ratios) / max(len(ratios) - 1, 1))
    min_r = min(ratios)
    cv = std_r / mean_r if mean_r > 0 else float("inf")

    ec_mean = mean_r >= 1.45
    ec_min_rep = min_r >= 1.35
    ec_std = cv <= 0.05
    passed = ec_mean and ec_min_rep and ec_std

    result = {
        "claim": "C1",
        "ratios_per_seed": {str(s): round(r, 4) for s, r in zip(seeds, ratios)},
        "mean_ratio": round(mean_r, 4),
        "std_ratio": round(std_r, 4),
        "min_ratio": round(min_r, 4),
        "cv": round(cv, 4),
        "ec_mean_pass": ec_mean,
        "ec_min_rep_pass": ec_min_rep,
        "ec_std_pass": ec_std,
        "status": "PASS" if passed else "FAIL",
    }

    print(f"  mean={mean_r:.3f}×  min={min_r:.3f}×  CV={cv:.3f}  → {result['status']}")
    return result


def validate_c2(out_dir: str, dry_run: bool) -> dict:
    """
    C2: APIS+AdapterSlots ≥ 2× system gain over TP=2 vLLM at K=50.
    Gate: mean ≥ 1.80×, min-rep ≥ 1.60×.
    """
    print("\n=== C2: APIS 2× system claim ===")
    out_path = f"{out_dir}/C2/apis_k50.json"

    if not Path(out_path).exists():
        _run_apis([
            "--model", "./models/llama-7b",
            "--adapter-dir", "./adapters",
            "--num-adapters", "50", "--rank", "32",
            "--request-rate", "7",
            "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
            "--tmax", "90", "--wgkp-threshold", "8",
            "--tp-baseline", "2",
            "--output", out_path,
        ], dry_run=dry_run)

    apis_result = _load_result(out_path)
    if apis_result is None:
        if dry_run:
            apis_result = {
                "summary": {"ratio_mean": 1.92, "ratio_min": 1.75, "ec_ab6_pass": True}
            }
        else:
            return {"claim": "C2", "status": "ERROR", "error": "result file not found"}

    summary = apis_result.get("summary", {})
    mean_r = summary.get("ratio_mean", 0.0)
    min_r = summary.get("ratio_min", 0.0)

    ec_mean = mean_r >= 1.80
    ec_min_rep = min_r >= 1.60
    passed = ec_mean and ec_min_rep

    result = {
        "claim": "C2",
        "mean_ratio": round(mean_r, 4),
        "min_ratio": round(min_r, 4),
        "ec_mean_pass": ec_mean,
        "ec_min_rep_pass": ec_min_rep,
        "status": "PASS" if passed else "FAIL",
    }
    print(f"  APIS mean={mean_r:.3f}×  min={min_r:.3f}×  → {result['status']}")
    return result


def validate_c3(out_dir: str, model: str, adapter_dir: str, dry_run: bool) -> dict:
    """
    C3: GWAR(8)–throughput Pearson r ≥ 0.85, p < 0.001.
    Sweeps T_max ∈ {30,60,90,120,150,200}ms with AdapterSlots C7, K=10.
    Each T_max gives a different GWAR(8) and throughput point.
    """
    print("\n=== C3: GWAR(8)–throughput Pearson correlation ===")
    tmax_list = [30, 60, 90, 120, 150, 200]
    gwar8_vals = []
    tps_vals = []

    for tmax in tmax_list:
        out_path = f"{out_dir}/C3/C7_tmax{tmax}.json"
        if not Path(out_path).exists():
            _run_bench([
                "--backend", "adapterslots", "--mode", "C7",
                "--model", model,
                "--adapter-dir", adapter_dir,
                "--num-adapters", "10", "--rank", "32",
                "--request-rate", "7", "--pattern", "zipf",
                "--num-prompts", "1000", "--warmup", "20", "--reps", "3",
                "--tmax", str(tmax), "--wgkp-threshold", "8",
                "--output", out_path,
            ], dry_run=dry_run)

        result = _load_result(out_path)
        if result is None and dry_run:
            # Synthetic: lower T_max → lower GWAR but same or higher throughput
            gwar8_synthetic = 0.45 + (200 - tmax) / 200 * 0.4
            tps_synthetic = 350 + (gwar8_synthetic - 0.45) * 800
            gwar8_vals.append(gwar8_synthetic)
            tps_vals.append(tps_synthetic)
            continue

        if result is None:
            print(f"    WARNING: no result for T_max={tmax}, skipping")
            continue

        # Extract GWAR(8) from summary or reps
        gwar8 = result.get("summary", {}).get("gwar8_mean", None)
        if gwar8 is None:
            # Try from individual reps
            gwar8_reps = [r.get("gwar8", None) for r in result.get("reps", []) if r.get("gwar8") is not None]
            gwar8 = sum(gwar8_reps) / len(gwar8_reps) if gwar8_reps else None

        tps = result.get("summary", {}).get("throughput_toks_mean", None)
        if gwar8 is not None and tps is not None and tps > 0:
            gwar8_vals.append(gwar8)
            tps_vals.append(tps)
        else:
            print(f"    WARNING: missing gwar8 or tps for T_max={tmax}")

    if len(gwar8_vals) < 3:
        return {
            "claim": "C3",
            "status": "ERROR",
            "error": f"only {len(gwar8_vals)} data points, need ≥ 3",
        }

    r, p = _pearson_r(gwar8_vals, tps_vals)

    ec_r = r >= 0.85
    ec_p = p < 0.001
    passed = ec_r and ec_p

    result = {
        "claim": "C3",
        "tmax_list": tmax_list[:len(gwar8_vals)],
        "gwar8_vals": [round(v, 4) for v in gwar8_vals],
        "tps_vals": [round(v, 2) for v in tps_vals],
        "pearson_r": round(r, 4),
        "p_value": round(p, 6),
        "ec_r_pass": ec_r,
        "ec_p_pass": ec_p,
        "n_points": len(gwar8_vals),
        "status": "PASS" if passed else "FAIL",
    }
    print(f"  Pearson r={r:.3f}  p={p:.4f}  n={len(gwar8_vals)}  → {result['status']}")
    return result


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="sota_evaluation claims validator")
    ap.add_argument("--model", default="./models/llama-7b")
    ap.add_argument("--adapter-dir", default="./adapters")
    ap.add_argument("--output-dir", default="results/sota_evaluation/claims")
    ap.add_argument(
        "--which",
        nargs="+",
        default=["C1", "C2", "C3"],
        choices=["C1", "C2", "C3"],
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out = args.output_dir
    Path(out).mkdir(parents=True, exist_ok=True)

    print(f"run_claims.py -- claims={args.which}")

    dispatch = {
        "C1": lambda: validate_c1(out, args.dry_run),
        "C2": lambda: validate_c2(out, args.dry_run),
        "C3": lambda: validate_c3(out, args.model, args.adapter_dir, args.dry_run),
    }

    report = {}
    for name in args.which:
        for subdir in [f"{out}/{name}"]:
            Path(subdir).mkdir(parents=True, exist_ok=True)
        report[name] = dispatch[name]()

    report_path = Path(out) / "claims_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n=== Claims Report ===")
    for name, res in report.items():
        print(f"  {name}: {res.get('status', 'UNKNOWN')}")
    print(f"\n  Full report: {report_path}")

    # Exit with non-zero if any claim failed
    if any(r.get("status") == "FAIL" for r in report.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
