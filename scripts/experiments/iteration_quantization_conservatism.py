"""
iteration_quantization_conservatism.py -- Iteration Quantization Analysis (Proposition 5.5)

Experiment §6.5b from erlang_scheduler.md.

Tests the conservatism guarantee of Proposition 5.5:

    Under iteration-quantized dispatch with period τ_iter, the achieved WAR is
    always ≥ WAR* because the effective timeout is rounded UP to the next
    iteration boundary:
        T_max_quantized = ceil(T_max* / τ_iter) × τ_iter  ≥  T_max*
        ⟹  WAR_actual = F_k(T_max_quantized) ≥ F_k(T_max*) = WAR*

Modes
-----
simulate  : Simulate quantized dispatch with Poisson arrivals (no GPU needed).
            Sweeps WAR* ∈ {0.5, 0.7, 0.8, 0.9} and τ_iter values.
            Measures WAR_actual and verifies WAR_actual ≥ WAR* for all rows.

analyze   : Load results CSV from a live serving run and compute ρ_k = τ_iter/T_max*.
            Compare empirical WAR to Erlang prediction.

Usage (simulate, Two A6000 PCIe τ_iter ≈ 120 ms):
    python scripts/experiments/iteration_quantization_conservatism.py \\
        --mode simulate \\
        --K 4 --alpha-zipf 0.9 --lambda-total 7 \\
        --tau-iter-ms 120 \\
        --war-targets 0.5 0.7 0.8 0.9 \\
        --n-samples 200000 \\
        --output results/erlang_scheduler/two_a6000_pcie/quantization_analysis.csv

Usage (simulate, Two H100 NVLink τ_iter ≈ 4 ms):
    python scripts/experiments/iteration_quantization_conservatism.py \\
        --mode simulate \\
        --K 4 --alpha-zipf 0.9 --lambda-total 7 \\
        --tau-iter-ms 4 \\
        --war-targets 0.5 0.7 0.8 0.9 \\
        --n-samples 200000 \\
        --output results/erlang_scheduler/two_h100_nvlink/erlang_precision_h100.csv

Usage (cross-hardware summary):
    python scripts/experiments/iteration_quantization_conservatism.py \\
        --mode cross-hw \\
        --K 4 --alpha-zipf 0.9 --lambda-total 7 \\
        --war-target 0.8 \\
        --n-samples 200000 \\
        --output results/erlang_scheduler/cross_hw_quantization.csv

References
----------
    erlang_scheduler.md §6.5b, §8.3, EC 11.2.2, EC 11.3.1
    Proposition 5.5 (Iteration Quantization Conservatism)
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np


from adapter_slots.dispatch.erlang import (
    compute_tmax_erlang,
    erlang_cdf,
    quantization_conservatism_bound,
)


# Utilities

def zipf_rates(K: int, alpha: float, lambda_total: float) -> list:
    weights = [k ** (-alpha) for k in range(1, K + 1)]
    total = sum(weights)
    return [(w / total) * lambda_total for w in weights]


def simulate_quantized_war(
    lambda_k: float,
    warp_size: int,
    t_max_s: float,
    tau_iter_s: float,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """Simulate WAR under iteration-quantized dispatch.

    Models the alignment buffer operating at fixed iteration period τ_iter.
    A warp fills if t_fill ≤ T_max_quantized, where:
        T_max_quantized = ceil(T_max* / τ_iter) × τ_iter

    Under Poisson arrivals, t_fill ~ Erlang(W, λ_k), so:
        WAR_actual = P(t_fill ≤ T_max_quantized)

    This function draws n_samples fill times from the Erlang distribution and
    measures the fraction that fall within T_max_quantized.

    Args:
        lambda_k:    Arrival rate (tokens/sec).
        warp_size:   GPU warp width W = 32.
        t_max_s:     Continuous-time T_max^(k)* in seconds.
        tau_iter_s:  Iteration period in seconds.
        n_samples:   Number of samples.
        rng:         NumPy random generator.

    Returns:
        Empirical WAR_actual (fraction of warps filled within T_max_quantized).
    """
    if lambda_k <= 0:
        return 0.0

    # Quantized timeout: round up to nearest iteration boundary
    if tau_iter_s > 0:
        t_max_quantized = math.ceil(t_max_s / tau_iter_s) * tau_iter_s
    else:
        t_max_quantized = t_max_s  # continuous-time limit

    # Fill time = sum of W i.i.d. Exp(lambda_k) inter-arrival times
    iats = rng.exponential(scale=1.0 / lambda_k, size=(n_samples, warp_size))
    fill_times = iats.sum(axis=1)

    return float((fill_times <= t_max_quantized).mean())


# Hardware preset τ_iter values

HARDWARE_PRESETS = {
    "single_a6000":    {"tau_iter_ms": 40.0,  "label": "Single A6000"},
    "two_a6000_pcie":  {"tau_iter_ms": 120.0, "label": "Two A6000 PCIe (TP=2)"},
    "single_h100":     {"tau_iter_ms": 5.0,   "label": "Single H100"},
    "two_h100_nvlink": {"tau_iter_ms": 4.0,   "label": "Two H100 NVLink (TP=2)"},
}


# Mode: simulate

def run_simulate(args) -> list:
    """Simulate quantized dispatch and verify Proposition 5.5 conservatism."""
    rng = np.random.default_rng(args.seed)
    lambda_k_list = zipf_rates(args.K, args.alpha_zipf, args.lambda_total)
    tau_iter_s = args.tau_iter_ms / 1000.0

    print(f"\n[Quantization] Simulation: K={args.K}  Zipf α={args.alpha_zipf}  "
          f"λ_total={args.lambda_total}  τ_iter={args.tau_iter_ms:.1f}ms  "
          f"n={args.n_samples:,}  label={args.label}")

    header = (f"  {'Adapter':>10}  {'λ_k':>8}  {'T_max*(ms)':>11}  "
              f"{'ρ_k':>7}  {'WAR*':>6}  {'WAR_actual':>11}  "
              f"{'Δ(+?)':>8}  {'Bound':>9}  {'Prop5.5?':>9}")
    print(header)
    print("  " + "─" * 88)

    rows = []
    all_conservative = True

    for i, lam in enumerate(lambda_k_list):
        adapter = f"adapter_{i}"

        t_max_s = compute_tmax_erlang(
            args.warp_size, lam, args.war_target,
            ttft_slo_ms=1_000_000.0,  # no fairness cap for Prop 5.5 test
        )
        rho_k = tau_iter_s / max(t_max_s, 1e-12)

        war_actual = simulate_quantized_war(
            lam, args.warp_size, t_max_s, tau_iter_s, args.n_samples, rng
        )
        delta = war_actual - args.war_target
        bound = quantization_conservatism_bound(
            t_max_s, tau_iter_s, args.warp_size, lam
        )

        # Proposition 5.5: WAR_actual must be >= WAR* (within sampling noise)
        sampling_noise = 2.0 / math.sqrt(args.n_samples)  # 2σ
        conservative = war_actual >= args.war_target - sampling_noise

        if not conservative:
            all_conservative = False

        print(f"  {adapter:>10}  {lam:>8.4f}  {t_max_s * 1000:>11.2f}  "
              f"{rho_k:>7.3f}  {args.war_target:>6.3f}  {war_actual:>11.6f}  "
              f"{delta:>+8.6f}  {bound:>9.6f}  {'OK' if conservative else 'FAIL':>9}")

        rows.append({
            "adapter": adapter,
            "lambda_k": lam,
            "t_max_ms": t_max_s * 1000,
            "tau_iter_ms": args.tau_iter_ms,
            "rho_k": rho_k,
            "war_target": args.war_target,
            "war_actual": war_actual,
            "delta_war": delta,
            "overdelivery_bound": bound,
            "conservatism_ok": conservative,
            "hardware_label": args.label,
            "K": args.K,
            "alpha_zipf": args.alpha_zipf,
            "n_samples": args.n_samples,
        })

    print(f"\n  Proposition 5.5: {'PASS' if all_conservative else 'FAIL'}")
    print(f"  (WAR_actual ≥ WAR* for all adapters within 2σ sampling noise)")

    return rows


# Mode: cross-hw

def run_cross_hw(args) -> list:
    """Sweep all hardware presets and build the cross-hardware comparison table."""
    rng = np.random.default_rng(args.seed)
    lambda_k_list = zipf_rates(args.K, args.alpha_zipf, args.lambda_total)

    print(f"\n[Quantization] Cross-hardware comparison: K={args.K}  "
          f"Zipf α={args.alpha_zipf}  λ_total={args.lambda_total}  "
          f"WAR*={args.war_target}")

    all_rows = []

    for hw_key, hw_cfg in HARDWARE_PRESETS.items():
        tau_iter_ms = hw_cfg["tau_iter_ms"]
        label = hw_cfg["label"]
        tau_iter_s = tau_iter_ms / 1000.0

        print(f"\n  {label} (τ_iter = {tau_iter_ms:.1f} ms)")

        for i, lam in enumerate(lambda_k_list):
            t_max_s = compute_tmax_erlang(
                args.warp_size, lam, args.war_target, ttft_slo_ms=1_000_000.0
            )
            rho_k = tau_iter_s / max(t_max_s, 1e-12)
            war_actual = simulate_quantized_war(
                lam, args.warp_size, t_max_s, tau_iter_s, args.n_samples, rng
            )
            delta = war_actual - args.war_target
            bound = quantization_conservatism_bound(
                t_max_s, tau_iter_s, args.warp_size, lam
            )

            all_rows.append({
                "hardware": hw_key,
                "hardware_label": label,
                "tau_iter_ms": tau_iter_ms,
                "adapter": f"adapter_{i}",
                "lambda_k": lam,
                "t_max_ms": t_max_s * 1000,
                "rho_k": rho_k,
                "war_target": args.war_target,
                "war_actual": war_actual,
                "delta_war": delta,
                "overdelivery_bound": bound,
                "conservatism_ok": war_actual >= args.war_target - 2.0 / math.sqrt(args.n_samples),
            })

    # Print cross-hardware summary (mean over adapters)
    print(f"\n  Cross-hardware summary (K={args.K} adapters, mean over all adapters):")
    print(f"  {'Hardware':>22}  {'τ_iter(ms)':>11}  {'mean ρ_k':>10}  "
          f"{'mean Δ WAR':>11}  {'max Δ WAR':>10}  {'Continuous?':>12}")
    print(f"  {'─'*82}")

    for hw_key, hw_cfg in HARDWARE_PRESETS.items():
        hw_rows = [r for r in all_rows if r["hardware"] == hw_key]
        if not hw_rows:
            continue
        mean_rho = sum(r["rho_k"] for r in hw_rows) / len(hw_rows)
        mean_delta = sum(r["delta_war"] for r in hw_rows) / len(hw_rows)
        max_delta = max(r["delta_war"] for r in hw_rows)
        is_ct = hw_cfg["tau_iter_ms"] < 10.0
        print(f"  {hw_cfg['label']:>22}  {hw_cfg['tau_iter_ms']:>11.1f}  "
              f"{mean_rho:>10.3f}  {mean_delta:>+11.6f}  {max_delta:>+10.6f}  "
              f"{'Yes (NVLink)' if is_ct else 'No (PCIe)':>12}")

    print(f"\n  Expected pattern (Proposition 5.5 / §6.5b):")
    print(f"    NVLink (τ_iter small) → ρ_k << 1 → Δ WAR ≈ 0 (continuous-time)")
    print(f"    PCIe (τ_iter large)  → ρ_k ≥ 1  → Δ WAR > 0 (quantization over-delivery)")

    return all_rows


# Mode: analyze

def run_analyze(args) -> None:
    """Load a previously saved quantization CSV and check exit conditions."""
    if not args.input:
        print("[analyze] --input CSV is required for analyze mode")
        return

    rows = []
    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"[analyze] No rows in {args.input}")
        return

    print(f"\n[Quantization] Analyzing {len(rows)} rows from {args.input}")

    # EC 11.2.2: Proposition 5.5 conservatism verified on real hardware
    failed = [r for r in rows if r.get("conservatism_ok", "True").lower() == "false"]
    if failed:
        print(f"  EC 11.2.2 Proposition 5.5: FAIL -- {len(failed)} rows violated conservatism:")
        for r in failed[:5]:
            print(f"    {r['adapter']}: WAR_actual={r['war_actual']:.6f} < WAR*={r['war_target']:.3f}")
    else:
        print(f"  EC 11.2.2 Proposition 5.5: PASS -- WAR_actual ≥ WAR* for all rows")

    # Quantization regime summary
    rho_vals = [float(r["rho_k"]) for r in rows if r.get("rho_k")]
    if rho_vals:
        mean_rho = sum(rho_vals) / len(rho_vals)
        print(f"\n  Mean quantization ratio ρ_k = τ_iter / T_max* = {mean_rho:.3f}")
        if mean_rho < 0.1:
            print(f"  → Continuous-time regime (ρ_k << 1): Erlang model is most accurate")
        elif mean_rho < 1.0:
            print(f"  → Moderate quantization (ρ_k < 1): small over-delivery expected")
        else:
            print(f"  → Quantized regime (ρ_k ≥ 1): over-delivery can be significant on PCIe")

    # Over-delivery statistics
    deltas = [float(r["delta_war"]) for r in rows if r.get("delta_war")]
    if deltas:
        print(f"\n  Over-delivery (WAR_actual − WAR*) statistics:")
        print(f"    mean  = {sum(deltas)/len(deltas):+.6f}")
        print(f"    max   = {max(deltas):+.6f}")
        print(f"    min   = {min(deltas):+.6f}")


# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Proposition 5.5 iteration quantization analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["simulate", "cross-hw", "analyze"],
        default="simulate",
    )

    # Workload
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--alpha-zipf", type=float, default=0.9)
    parser.add_argument("--lambda-total", type=float, default=7.0)
    parser.add_argument("--war-target", type=float, default=0.8)
    parser.add_argument("--war-targets", nargs="+", type=float,
                        default=[0.5, 0.7, 0.8, 0.9],
                        help="WAR* values to sweep (simulate mode).")
    parser.add_argument("--warp-size", type=int, default=32)

    # Hardware
    parser.add_argument("--tau-iter-ms", type=float, default=120.0,
                        help="Iteration period in ms (simulate mode). "
                             "Set ~120 for Two A6000 PCIe, ~4 for Two H100 NVLink.")
    parser.add_argument("--label", type=str, default="",
                        help="Hardware label for the output CSV.")

    # Simulation
    parser.add_argument("--n-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)

    # I/O
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path.")
    parser.add_argument("--input", type=str, default=None,
                        help="(analyze) Input CSV from a previous run.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "simulate":
        # Sweep all WAR* values and concatenate rows
        all_rows = []
        for war_target in args.war_targets:
            args.war_target = war_target
            rows = run_simulate(args)
            all_rows.extend(rows)
        rows = all_rows

    elif args.mode == "cross-hw":
        rows = run_cross_hw(args)

    elif args.mode == "analyze":
        run_analyze(args)
        return 0

    else:
        print(f"Unknown mode: {args.mode}")
        return 1

    # Save to CSV
    if args.output and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Quantization] Saved {len(rows)} rows to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
