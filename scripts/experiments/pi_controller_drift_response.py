"""
pi_controller_drift_response.py -- E7 Drift Experiment: PI Controller Under Workload Drift (pi_controller, §5.1)

Primary experiment for pi_controller (maps to AB3 in the ablation plan).
Simulates PI-adaptive vs. Static T_max under four workload drift scenarios:

    step        -- step change at t=5min: uniform → Zipf α=0.9 (primary E7-drift)
    sinusoidal  -- diurnal drift: λ(t) = base*(1 + A*sin(2πt/period))
    burst       -- Poisson λ switches between 1 req/s and 20 req/s
    adversarial -- rapidly alternating popularity (30s period) for anti-windup test

Cross-hardware usage (same script, same gains -- Proposition 6.7):

  Single RTX A6000 (TP=1, K=4):
    python scripts/experiments/pi_controller_drift_response.py \\
        --workload step --K 4 --lambda-total 7.0 --alpha-zipf 0.9 \\
        --war-target 0.8 --warp-size 32 \\
        --duration 900 --drift-at 300 \\
        --output results/pi_controller/a6000_single/e7_step_drift.csv

  Two RTX A6000 PCIe (TP=2, K=4, step drift):
    python scripts/experiments/pi_controller_drift_response.py \\
        --workload step --K 4 --lambda-total 7.0 --alpha-zipf 0.9 \\
        --war-target 0.8 --warp-size 32 \\
        --duration 900 --drift-at 300 \\
        --tau-iter-ms 100 --hardware-label two_a6000_pcie \\
        --output results/pi_controller/two_a6000_pcie/e7_drift_tp2.csv

  Two RTX A6000 PCIe (TP=2, K=16, step drift -- §5.5c):
    python scripts/experiments/pi_controller_drift_response.py \\
        --workload step --K 16 --lambda-total 14.0 --alpha-zipf 0.9 \\
        --war-target 0.8 --warp-size 32 \\
        --duration 900 --drift-at 300 \\
        --tau-iter-ms 100 --hardware-label two_a6000_pcie \\
        --output results/pi_controller/two_a6000_pcie/e7_k16_drift.csv

  Two H100 NVLink (TP=2, K=4, step drift -- §5.6a):
    python scripts/experiments/pi_controller_drift_response.py \\
        --workload step --K 4 --lambda-total 7.0 --alpha-zipf 0.9 \\
        --war-target 0.8 --warp-size 32 \\
        --duration 900 --drift-at 300 \\
        --tau-iter-ms 5 --hardware-label two_h100_nvlink \\
        --output results/pi_controller/two_h100_nvlink/e7_drift_nvlink.csv

  Two H100 NVLink (TP=2, K=32, step drift -- §5.6c):
    python scripts/experiments/pi_controller_drift_response.py \\
        --workload step --K 32 --lambda-total 20.0 --alpha-zipf 0.9 \\
        --war-target 0.8 --warp-size 32 \\
        --duration 900 --drift-at 180 \\
        --tau-iter-ms 5 --hardware-label two_h100_nvlink \\
        --output results/pi_controller/two_h100_nvlink/e7_k32_drift.csv

  Sinusoidal drift (Single A6000):
    python scripts/experiments/pi_controller_drift_response.py \\
        --workload sinusoidal --K 4 --lambda-total 7.0 \\
        --war-target 0.8 --warp-size 32 \\
        --duration 900 --diurnal-period 300 --diurnal-amplitude 0.5 \\
        --output results/pi_controller/a6000_single/e7_sinusoidal_drift.csv

  Adversarial oscillation (Single A6000, anti-windup test):
    python scripts/experiments/pi_controller_drift_response.py \\
        --workload adversarial --K 4 --lambda-total 7.0 \\
        --war-target 0.8 --warp-size 32 --duration 1800 \\
        --output results/pi_controller/a6000_single/e7_adversarial.csv

Outputs per-iteration CSV:
    iteration, t_sec, phase, policy, war_observed, tmax_current,
    error_e_t, integral_term, proportional_term, tau_iter_ms,
    hardware_label, K, workload
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# project imports
from adapter_slots.control.pi_controller import PIController, IterationBoundaryPIController
from adapter_slots.control.estimator import estimate_lipschitz


# Lambda unit conversion
# The CLI --lambda-total is in tok/ms (tokens per millisecond), consistent with
# the paper spec "λ_k = 2.5 tok/ms" (pi_controller.md §5.1).
# The Erlang CDF is parameterised in tok/s (scale = 1/lambda_tok_s):
#   Erlang mean = W / lambda_tok_s = W * (1ms / lambda_tok_ms) = W/lambda_tok_ms [ms]
# This keeps T_max* in the 5–50ms range for lambda ∈ [1, 10] tok/ms.
# All internal Erlang calls must scale lambda by LAMBDA_TOK_MS_TO_S = 1000.
LAMBDA_TOK_MS_TO_S: float = 1000.0


# Workload / arrival rate generators

def zipf_rates(K: int, alpha: float, lambda_total: float) -> List[float]:
    """Per-adapter rates (tok/ms) under Zipf(alpha) distribution."""
    weights = [k ** (-alpha) for k in range(1, K + 1)]
    total = sum(weights)
    return [(w / total) * lambda_total for w in weights]


def uniform_rates(K: int, lambda_total: float) -> List[float]:
    """Per-adapter rates (tok/ms) under uniform distribution."""
    return [lambda_total / K] * K


def diurnal_lambda(
    t_sec: float,
    base_lambda: float = 5.0,
    amplitude: float = 0.5,
    period_sec: float = 300.0,
) -> float:
    """λ(t) = base * (1 + amplitude * sin(2π t / period)) -- compressed diurnal."""
    return base_lambda * (1.0 + amplitude * math.sin(2.0 * math.pi * t_sec / period_sec))


def burst_lambda(
    t_sec: float,
    quiet_rate: float = 1.0,
    burst_rate: float = 20.0,
    quiet_duration: float = 120.0,
    burst_duration: float = 30.0,
) -> float:
    """Poisson λ switches between quiet and burst phases."""
    cycle = quiet_duration + burst_duration
    pos = t_sec % cycle
    return burst_rate if pos >= quiet_duration else quiet_rate


# WAR simulation (Erlang-CDF model)

def simulate_war_erlang(
    tmax: float,
    lambda_k_list: List[float],
    p_k_list: List[float],
    warp_size: int,
    noise_std: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Simulate observed WAR at a given T_max using the Erlang CDF model plus noise.

    WAR(T_max) = Σ_k p_k * Erlang_CDF(W, λ_k, T_max)

    lambda_k_list is in tok/ms (CLI units); converted to tok/s internally via
    LAMBDA_TOK_MS_TO_S so the Erlang CDF scale matches tmax in seconds.

    Adds Gaussian noise (σ=noise_std) to simulate finite-sample variance.
    Clipped to [0, 1].
    """
    from scipy.stats import erlang

    if rng is None:
        rng = np.random.default_rng()

    war = sum(
        p * erlang.cdf(tmax, a=warp_size,
                       scale=1.0 / max(lam * LAMBDA_TOK_MS_TO_S, 1e-12))
        for lam, p in zip(lambda_k_list, p_k_list)
    )
    war += rng.normal(0.0, noise_std)
    return float(np.clip(war, 0.0, 1.0))


def optimal_tmax_erlang(
    lambda_k_list: List[float],
    p_k_list: List[float],
    warp_size: int,
    war_target: float,
    tmax_range: Tuple[float, float] = (0.0001, 5.0),
) -> float:
    """
    Find T_max* such that WAR(T_max*) = WAR* (no noise, bisection search).
    Used to set the initial T_max for the Static baseline and PI controller.

    lambda_k_list is in tok/ms; converted to tok/s internally.
    """
    from scipy.optimize import brentq
    from scipy.stats import erlang

    def f(tmax):
        war = sum(
            p * erlang.cdf(tmax, a=warp_size,
                           scale=1.0 / max(lam * LAMBDA_TOK_MS_TO_S, 1e-12))
            for lam, p in zip(lambda_k_list, p_k_list)
        )
        return war - war_target

    lo, hi = tmax_range
    if f(hi) < 0:
        return hi  # WAR* unachievable -- return upper bound
    if f(lo) > 0:
        return lo
    return brentq(f, lo, hi, xtol=1e-6)


# Single-run simulation

def auto_calibrate_gains(L: float, rho_target: float = 0.85) -> Tuple[float, float]:
    """
    Compute K_p and K_i from L to achieve a target spectral radius ρ_target.

    K_p is set so (1 - L*K_p) ≈ rho_target (for the proportional settling rate).
    K_i is set to 10% of the stability upper bound.

    Args:
        L:           Lipschitz constant from estimate_lipschitz().
        rho_target:  Target spectral radius for settling time tuning (default 0.85
                     → N_settle ≈ 28 iterations).

    Returns:
        (kp, ki) gains calibrated for the given L.
    """
    kp = (1.0 - rho_target) / max(L, 1e-12)
    # Keep K_p strictly within stability bound (< 2/L)
    kp = min(kp, 0.9 * 2.0 / L)
    ki_upper = kp * (2.0 / L - kp)
    ki = 0.10 * ki_upper
    return kp, ki


def run_drift_simulation(
    workload: str,
    K: int,
    lambda_total: float,
    alpha_zipf: float,
    war_target: float,
    warp_size: int,
    duration_sec: float,
    tau_iter_ms: float,
    drift_at_sec: float,
    diurnal_period: float,
    diurnal_amplitude: float,
    kp: float,
    ki: float,
    hardware_label: str,
    noise_std: float,
    seed: int,
    auto_gains: bool = False,
) -> List[dict]:
    """
    Simulate the E7-drift experiment for both PI-adaptive and Static T_max policies.

    Returns a list of per-iteration rows (both policies interleaved with policy column).
    """
    rng = np.random.default_rng(seed)
    tau_iter_s = tau_iter_ms / 1000.0
    n_iters = int(duration_sec / tau_iter_s)

    # Initial workload (pre-drift)
    if workload in ("step", "adversarial"):
        lam_pre = uniform_rates(K, lambda_total)
        p_pre = [1.0 / K] * K
    elif workload in ("sinusoidal", "burst"):
        lam_pre = uniform_rates(K, lambda_total)
        p_pre = [1.0 / K] * K
    else:
        raise ValueError(f"Unknown workload: {workload}")

    # Post-drift workload (for step / adversarial)
    lam_post = zipf_rates(K, alpha_zipf, lambda_total)
    p_post = [lam / lambda_total for lam in lam_post]

    # Compute Erlang-optimal T_max for pre-drift (shared starting point)
    tmax_init = optimal_tmax_erlang(lam_pre, p_pre, warp_size, war_target)

    # Lipschitz constant (from Erlang model, hardware-independent)
    # Build lambda dict with per-adapter index to avoid duplicate-key collapse
    # for uniform distributions (all lambdas equal → single dict entry otherwise).
    # Scale lambda to tok/s for Erlang CDF (CLI uses tok/ms)
    lam_k_dict_pre = {lam * LAMBDA_TOK_MS_TO_S + i * 1e-12: p for i, (lam, p) in enumerate(zip(lam_pre, p_pre))}
    L = estimate_lipschitz(lam_k_dict_pre, warp_size)

    # Auto-calibrate gains from L if requested (simulation mode).
    # Useful when K_p=0.01 from the real server spec is too small for the simulated L,
    # which would make N_settle unreasonably large (10k+ iters).
    if auto_gains:
        kp, ki = auto_calibrate_gains(L, rho_target=0.85)
        print(f"[E7-drift] Auto-calibrated gains: K_p={kp:.6f}  K_i={ki:.8f}  "
              f"(from L={L:.6f}, ρ_target=0.85)", file=sys.stderr)

    # Validate stability condition before running
    if not (0 < kp < 2.0 / L):
        print(
            f"[WARNING] K_p={kp:.4f} outside stability range (0, {2.0/L:.4f}). "
            "Results may diverge.",
            file=sys.stderr,
        )

    # PI controller (iteration-boundary-driven)
    pi = PIController(
        kp=kp, ki=ki, war_target=war_target,
        lipschitz=L, tmax_init=tmax_init,
    )
    ctrl = IterationBoundaryPIController(pi)

    # Static baseline: T_max fixed at Erlang-optimal for pre-drift
    static_tmax = tmax_init

    rows: List[dict] = []

    for i in range(n_iters):
        t_sec = i * tau_iter_s
        phase = "stationary" if t_sec < drift_at_sec else "post_drift"

        # Determine active arrival rates at this iteration
        if workload == "step":
            if t_sec < drift_at_sec:
                lam_active, p_active = lam_pre, p_pre
            else:
                lam_active, p_active = lam_post, p_post

        elif workload == "adversarial":
            # Switch every 30s between uniform and Zipf
            switch_period = 30.0
            cycle_pos = t_sec % (2 * switch_period)
            if cycle_pos < switch_period:
                lam_active, p_active = lam_pre, p_pre
            else:
                lam_active, p_active = lam_post, p_post
            phase = "adversarial"

        elif workload == "sinusoidal":
            lam_t = diurnal_lambda(t_sec, lambda_total, diurnal_amplitude, diurnal_period)
            lam_active = uniform_rates(K, lam_t)
            p_active = [1.0 / K] * K
            phase = "sinusoidal"

        elif workload == "burst":
            lam_t = burst_lambda(t_sec)
            lam_active = uniform_rates(K, lam_t)
            p_active = [1.0 / K] * K
            phase = "burst"

        # Simulate WAR for each policy
        # PI-adaptive
        war_pi = simulate_war_erlang(
            ctrl.tmax, lam_active, p_active, warp_size, noise_std, rng,
        )
        ctrl.record_batch_war(war_pi)
        new_tmax = ctrl.trigger_iteration_end()
        e_pi = war_target - war_pi

        rows.append({
            "iteration": i,
            "t_sec": round(t_sec, 4),
            "phase": phase,
            "policy": "PI_adaptive",
            "war_observed": round(war_pi, 6),
            "tmax_current": round(new_tmax, 6),
            "error_e_t": round(e_pi, 6),
            "integral_term": round(pi.ki * pi.integral, 6),
            "proportional_term": round(pi.kp * e_pi, 6),
            "tau_iter_ms": tau_iter_ms,
            "hardware_label": hardware_label,
            "K": K,
            "workload": workload,
            "spectral_radius": round(pi.spectral_radius, 6),
        })

        # Static T_max
        war_static = simulate_war_erlang(
            static_tmax, lam_active, p_active, warp_size, noise_std, rng,
        )
        e_static = war_target - war_static

        rows.append({
            "iteration": i,
            "t_sec": round(t_sec, 4),
            "phase": phase,
            "policy": "Static_Tmax",
            "war_observed": round(war_static, 6),
            "tmax_current": round(static_tmax, 6),
            "error_e_t": round(e_static, 6),
            "integral_term": 0.0,
            "proportional_term": 0.0,
            "tau_iter_ms": tau_iter_ms,
            "hardware_label": hardware_label,
            "K": K,
            "workload": workload,
            "spectral_radius": "N/A",
        })

    return rows


# Convergence analysis

def analyze_convergence(
    rows: List[dict],
    war_target: float,
    drift_at_sec: float,
    tolerance: float = 0.05,
) -> dict:
    """
    Compute empirical settling time N_settle and wall-clock t_settle_wc.

    Settling is defined as the first iteration after drift where WAR >= WAR* - tolerance
    and remains there for 10 consecutive iterations.

    Returns dict with analysis results for both policies.
    """
    results = {}
    for policy in ("PI_adaptive", "Static_Tmax"):
        policy_rows = [r for r in rows if r["policy"] == policy and r["phase"] == "post_drift"]
        if not policy_rows:
            results[policy] = {"n_settle_iters": None, "t_settle_wc_sec": None}
            continue

        war_vals = [r["war_observed"] for r in policy_rows]
        tau_iter_ms = policy_rows[0]["tau_iter_ms"]
        tau_iter_s = tau_iter_ms / 1000.0

        # Find first run of 10 consecutive iterations above WAR* - tolerance
        threshold = war_target - tolerance
        n_settle = None
        for j in range(len(war_vals) - 9):
            window = war_vals[j:j + 10]
            if all(w >= threshold for w in window):
                n_settle = j
                break

        results[policy] = {
            "n_settle_iters": n_settle,
            "t_settle_wc_sec": round(n_settle * tau_iter_s, 3) if n_settle is not None else None,
            "post_drift_war_mean": round(float(np.mean(war_vals)), 4),
            "post_drift_war_p10": round(float(np.percentile(war_vals, 10)), 4),
            "steady_state_error": round(float(war_target - np.mean(war_vals[-20:])), 4),
        }

    return results


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="E7 drift experiment (pi_controller §5.1–5.4)")
    p.add_argument("--workload", default="step",
                   choices=["step", "sinusoidal", "burst", "adversarial"],
                   help="Drift workload type")
    p.add_argument("--K", type=int, default=4, help="Number of adapters")
    p.add_argument("--lambda-total", type=float, default=7.0,
                   help="Total arrival rate (req/s)")
    p.add_argument("--alpha-zipf", type=float, default=0.9,
                   help="Zipf skew parameter (post-drift)")
    p.add_argument("--war-target", type=float, default=0.8, help="WAR* target")
    p.add_argument("--warp-size", type=int, default=32, help="Erlang warp size W")
    p.add_argument("--duration", type=float, default=900.0,
                   help="Total experiment duration (seconds)")
    p.add_argument("--drift-at", type=float, default=300.0,
                   help="Drift event time (seconds from start) for step workload")
    p.add_argument("--tau-iter-ms", type=float, default=30.0,
                   help="Simulated per-iteration wall-clock time (ms). "
                        "Single A6000≈30ms, PCIe≈100ms, NVLink≈5ms")
    p.add_argument("--hardware-label", default="single_a6000",
                   choices=["single_a6000", "two_a6000_pcie", "two_h100_nvlink"],
                   help="Hardware label for output annotation")
    p.add_argument("--diurnal-period", type=float, default=300.0,
                   help="Period for sinusoidal workload (seconds)")
    p.add_argument("--diurnal-amplitude", type=float, default=0.5,
                   help="Amplitude for sinusoidal workload (fraction of base rate)")
    p.add_argument("--kp", type=float, default=0.01,
                   help="PI proportional gain. Used when --auto-gains is not set.")
    p.add_argument("--ki", type=float, default=0.001,
                   help="PI integral gain. Used when --auto-gains is not set.")
    p.add_argument("--auto-gains", action="store_true",
                   help="Auto-calibrate K_p and K_i from estimated Lipschitz constant L. "
                        "Sets K_p to achieve ρ≈0.85 (N_settle≈28 iters). "
                        "Use this for simulation runs; use explicit K_p/K_i for real serving.")
    p.add_argument("--noise-std", type=float, default=0.02,
                   help="WAR observation noise std dev (simulates finite-sample variance)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", required=True, help="Output CSV path")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[E7-drift] workload={args.workload}  K={args.K}  "
          f"λ_total={args.lambda_total}  τ_iter={args.tau_iter_ms}ms  "
          f"hardware={args.hardware_label}")
    print(f"[E7-drift] K_p={args.kp}  K_i={args.ki}  WAR*={args.war_target}  "
          f"W={args.warp_size}  duration={args.duration}s")

    t0 = time.monotonic()
    rows = run_drift_simulation(
        workload=args.workload,
        K=args.K,
        lambda_total=args.lambda_total,
        alpha_zipf=args.alpha_zipf,
        war_target=args.war_target,
        warp_size=args.warp_size,
        duration_sec=args.duration,
        tau_iter_ms=args.tau_iter_ms,
        drift_at_sec=args.drift_at,
        diurnal_period=args.diurnal_period,
        diurnal_amplitude=args.diurnal_amplitude,
        kp=args.kp,
        ki=args.ki,
        hardware_label=args.hardware_label,
        noise_std=args.noise_std,
        seed=args.seed,
        auto_gains=args.auto_gains,
    )
    elapsed = time.monotonic() - t0
    print(f"[E7-drift] Simulated {len(rows)} rows in {elapsed:.1f}s")

    # Write CSV
    fieldnames = [
        "iteration", "t_sec", "phase", "policy", "war_observed", "tmax_current",
        "error_e_t", "integral_term", "proportional_term", "tau_iter_ms",
        "hardware_label", "K", "workload", "spectral_radius",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[E7-drift] Wrote {len(rows)} rows → {output_path}")

    # Proposition 6.6 prediction (always shown, informational)
    lam_pre = uniform_rates(args.K, args.lambda_total)
    p_pre = [1.0 / args.K] * args.K
    lam_k_dict_pre = {lam * LAMBDA_TOK_MS_TO_S + i * 1e-12: p for i, (lam, p) in enumerate(zip(lam_pre, p_pre))}
    L = estimate_lipschitz(lam_k_dict_pre, args.warp_size)

    kp_check, ki_check = args.kp, args.ki
    if args.auto_gains:
        kp_check, ki_check = auto_calibrate_gains(L, rho_target=0.85)
    pi_check = PIController(
        kp=kp_check, ki=ki_check, war_target=args.war_target, lipschitz=L,
    )
    n_settle_pred = pi_check.settling_time_prediction()
    t_settle_wc_pred = n_settle_pred * args.tau_iter_ms / 1000.0

    print(f"\n[E7-drift] Proposition 6.6 predictions:")
    print(f"  L = {L:.6f}")
    print(f"  spectral_radius ρ(A) = {pi_check.spectral_radius:.6f}")
    print(f"  N_settle (predicted) = {n_settle_pred} iterations")
    print(f"  t_settle_wc (predicted) = {t_settle_wc_pred:.3f}s  "
          f"(= {n_settle_pred} × {args.tau_iter_ms}ms)")

    # Workload-specific analysis
    if args.workload == "adversarial":
        # Anti-windup stress test: verify T_max stays within controller bounds (EC 10.1.7)
        pi_rows = [r for r in rows if r["policy"] == "PI_adaptive"]
        tmax_vals = [r["tmax_current"] for r in pi_rows]
        war_vals = [r["war_observed"] for r in pi_rows]
        tmax_min_obs, tmax_max_obs = min(tmax_vals), max(tmax_vals)
        tmax_min_bound, tmax_max_bound = 0.001, 5.0
        in_bounds = tmax_min_obs >= tmax_min_bound and tmax_max_obs <= tmax_max_bound
        print(f"\n[E7-drift] Adversarial anti-windup analysis (EC 10.1.7):")
        print(f"  T_max observed: min={tmax_min_obs:.4f}s  max={tmax_max_obs:.4f}s")
        print(f"  T_max bounds:   [{tmax_min_bound}, {tmax_max_bound}]s")
        print(f"  T_max in bounds: {'PASS' if in_bounds else 'FAIL'}")
        print(f"  PI mean WAR: {np.mean(war_vals):.4f}  (target={args.war_target})")
    else:
        # Step / sinusoidal / burst: empirical N_settle vs predicted (EC 10.1.5)
        conv = analyze_convergence(rows, args.war_target, args.drift_at)
        print("\n[E7-drift] Convergence analysis:")
        for policy, res in conv.items():
            print(f"  {policy}:")
            for k, v in res.items():
                print(f"    {k}: {v}")

        pi_emp = conv.get("PI_adaptive", {})
        n_settle_emp = pi_emp.get("n_settle_iters")
        if n_settle_emp is not None:
            print(f"  N_settle (empirical) = {n_settle_emp} iterations  "
                  f"(ratio emp/pred = {n_settle_emp/n_settle_pred:.2f})")
            t_settle_wc_emp = pi_emp.get("t_settle_wc_sec", 0)
            print(f"  t_settle_wc (empirical) = {t_settle_wc_emp}s")
            pass_fail = "PASS" if n_settle_emp <= 2 * n_settle_pred else "FAIL"
            print(f"  EC 10.1.5 (N_settle emp ≤ 2× pred): {pass_fail}")
        else:
            print("  [NOTE] N_settle not reached within simulation duration.")

    print(f"\n[E7-drift] Done. Output: {output_path}")


if __name__ == "__main__":
    main()
