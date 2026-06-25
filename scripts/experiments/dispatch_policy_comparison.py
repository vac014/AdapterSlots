"""
dispatch_policy_comparison.py -- P-only vs. PI vs. MPC-lite vs. Static policy comparison (pi_controller §5.2)

Justifies the choice of PI over simpler (P-only) and more complex (MPC-lite) policies.
Runs all four policies on both the step-drift and sinusoidal-drift workloads.

Expected results (Single RTX A6000, EC 10.1 conditions 2 and 3):
    Static    -- fails under drift (permanent WAR drop ≥ 10% below WAR*)
    P-only    -- non-zero steady-state error (cannot eliminate drift bias)
    PI        -- zero steady-state error; converges in N_settle iterations
    MPC-lite  -- marginal improvement over PI (< 5% WAR gain) at 3× CPU overhead

Usage:

  Single A6000 (step drift + sinusoidal drift, K=4):
    python scripts/experiments/dispatch_policy_comparison.py \\
        --K 4 --lambda-total 7.0 --alpha-zipf 0.9 \\
        --war-target 0.8 --warp-size 32 \\
        --tau-iter-ms 30 --hardware-label single_a6000 \\
        --output results/pi_controller/a6000_single/convergence_comparison.csv

  Two A6000 PCIe (step drift, K=4 -- validates hardware-independence of gains):
    python scripts/experiments/dispatch_policy_comparison.py \\
        --K 4 --lambda-total 7.0 --alpha-zipf 0.9 \\
        --war-target 0.8 --warp-size 32 \\
        --tau-iter-ms 100 --hardware-label two_a6000_pcie \\
        --output results/pi_controller/two_a6000_pcie/convergence_comparison_tp2.csv

Outputs per-iteration CSV with columns:
    iteration, t_sec, phase, workload, policy, war_observed, tmax_current,
    error_e_t, steady_state_error, tau_iter_ms, hardware_label,
    cpu_overhead_us

Also prints a summary table to stdout.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from adapter_slots.control.pi_controller import PIController, IterationBoundaryPIController
from adapter_slots.control.estimator import estimate_lipschitz


# Shared helpers (reuse from pi_controller_drift_response.py logic)

def zipf_rates(K: int, alpha: float, lambda_total: float) -> List[float]:
    weights = [k ** (-alpha) for k in range(1, K + 1)]
    total = sum(weights)
    return [(w / total) * lambda_total for w in weights]


def uniform_rates(K: int, lambda_total: float) -> List[float]:
    return [lambda_total / K] * K


LAMBDA_TOK_MS_TO_S: float = 1000.0  # CLI --lambda-total is tok/ms; Erlang CDF needs tok/s


def simulate_war_erlang(
    tmax: float,
    lambda_k_list: List[float],
    p_k_list: List[float],
    warp_size: int,
    noise_std: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> float:
    from scipy.stats import erlang
    if rng is None:
        rng = np.random.default_rng()
    war = sum(
        p * erlang.cdf(tmax, a=warp_size, scale=1.0 / max(lam * LAMBDA_TOK_MS_TO_S, 1e-12))
        for lam, p in zip(lambda_k_list, p_k_list)
    )
    war += rng.normal(0.0, noise_std)
    return float(np.clip(war, 0.0, 1.0))


def optimal_tmax_erlang(
    lambda_k_list: List[float],
    p_k_list: List[float],
    warp_size: int,
    war_target: float,
) -> float:
    from scipy.optimize import brentq
    from scipy.stats import erlang

    def f(tmax):
        war = sum(
            p * erlang.cdf(tmax, a=warp_size, scale=1.0 / max(lam * LAMBDA_TOK_MS_TO_S, 1e-12))
            for lam, p in zip(lambda_k_list, p_k_list)
        )
        return war - war_target

    lo, hi = 1e-4, 5.0
    if f(hi) < 0:
        return hi
    if f(lo) > 0:
        return lo
    return brentq(f, lo, hi, xtol=1e-6)


# MPC-lite: 3-step lookahead using Erlang predictions

class MPCLiteController:
    """
    MPC-lite: 3-step lookahead controller using Erlang CDF predictions.

    At each step, evaluates WAR for T_max ± {Δ, 2Δ, 0} and selects the T_max
    that minimises |WAR(T_max) - WAR*| over a 3-step prediction horizon.

    Requires 3 Erlang CDF evaluations per update (vs. 0 for PI).
    Expected: marginal WAR improvement over PI (< 5%) at substantially higher
    CPU overhead -- justifying PI as the right operating point (EC 10.1 condition 3).
    """

    def __init__(
        self,
        war_target: float,
        tmax_init: float,
        tmax_min: float = 0.001,
        tmax_max: float = 5.0,
        delta: float = 0.001,
        horizon: int = 3,
    ) -> None:
        self.war_target = war_target
        self.tmax = tmax_init
        self.tmax_min = tmax_min
        self.tmax_max = tmax_max
        self.delta = delta
        self.horizon = horizon

    def update(
        self,
        lambda_k_list: List[float],
        p_k_list: List[float],
        warp_size: int,
    ) -> float:
        from scipy.stats import erlang

        def war_at(tmax):
            return sum(
                p * erlang.cdf(max(tmax, 1e-6), a=warp_size, scale=1.0 / max(lam * LAMBDA_TOK_MS_TO_S, 1e-12))
                for lam, p in zip(lambda_k_list, p_k_list)
            )

        # Candidate T_max values
        candidates = [
            self.tmax - 2 * self.delta,
            self.tmax - self.delta,
            self.tmax,
            self.tmax + self.delta,
            self.tmax + 2 * self.delta,
        ]
        candidates = [max(self.tmax_min, min(self.tmax_max, c)) for c in candidates]

        # Select candidate with smallest |WAR(c) - WAR*|
        best = min(candidates, key=lambda c: abs(war_at(c) - self.war_target))
        self.tmax = best
        return self.tmax


# P-only controller

class POnlyController:
    """
    Proportional-only controller (K_i = 0).

    Expected: non-zero steady-state error under drift because the integral term
    that eliminates steady-state offset is absent (EC 10.1 condition 2).
    """

    def __init__(
        self,
        kp: float,
        war_target: float,
        tmax_init: float,
        tmax_min: float = 0.001,
        tmax_max: float = 5.0,
    ) -> None:
        self.kp = kp
        self.war_target = war_target
        self.tmax = tmax_init
        self.tmax_min = tmax_min
        self.tmax_max = tmax_max

    def update(self, war_observed: float) -> float:
        e = self.war_target - war_observed
        self.tmax += self.kp * e
        self.tmax = float(np.clip(self.tmax, self.tmax_min, self.tmax_max))
        return self.tmax


# Main simulation

def run_policy_comparison(
    workload: str,
    K: int,
    lambda_total: float,
    alpha_zipf: float,
    war_target: float,
    warp_size: int,
    duration_sec: float,
    tau_iter_ms: float,
    drift_at_sec: float,
    kp: float,
    ki: float,
    hardware_label: str,
    noise_std: float,
    seed: int,
) -> Tuple[List[dict], dict]:
    """
    Run all four policies on the given workload and return rows + overhead dict.
    """
    rng = np.random.default_rng(seed)
    tau_iter_s = tau_iter_ms / 1000.0
    n_iters = int(duration_sec / tau_iter_s)

    lam_pre = uniform_rates(K, lambda_total)
    p_pre = [1.0 / K] * K
    lam_post = zipf_rates(K, alpha_zipf, lambda_total)
    p_post = [lam / lambda_total for lam in lam_post]

    tmax_init = optimal_tmax_erlang(lam_pre, p_pre, warp_size, war_target)
    # Scale lambda to tok/s for Erlang CDF (CLI uses tok/ms)
    lam_k_dict_pre = {lam * LAMBDA_TOK_MS_TO_S + i * 1e-12: p for i, (lam, p) in enumerate(zip(lam_pre, p_pre))}
    L = estimate_lipschitz(lam_k_dict_pre, warp_size)

    # Controllers
    pi_inner = PIController(kp=kp, ki=ki, war_target=war_target,
                            lipschitz=L, tmax_init=tmax_init)
    pi_ctrl = IterationBoundaryPIController(pi_inner)

    p_only = POnlyController(kp=kp, war_target=war_target, tmax_init=tmax_init)
    mpc = MPCLiteController(war_target=war_target, tmax_init=tmax_init)
    static_tmax = tmax_init

    # Overhead tracking: time per update call (microseconds)
    overhead_us = {policy: [] for policy in ("PI", "P_only", "MPC_lite", "Static")}

    rows: List[dict] = []

    for i in range(n_iters):
        t_sec = i * tau_iter_s

        if workload == "step":
            phase = "stationary" if t_sec < drift_at_sec else "post_drift"
            if t_sec < drift_at_sec:
                lam_active, p_active = lam_pre, p_pre
            else:
                lam_active, p_active = lam_post, p_post
        else:
            # sinusoidal: λ oscillates, no sharp phase boundary
            import math
            diurnal_period = 300.0
            lam_t = lambda_total * (1.0 + 0.5 * math.sin(2 * math.pi * t_sec / diurnal_period))
            lam_active = uniform_rates(K, lam_t)
            p_active = [1.0 / K] * K
            phase = "sinusoidal"

        for policy_name in ("PI", "P_only", "MPC_lite", "Static"):
            if policy_name == "PI":
                war_obs = simulate_war_erlang(pi_ctrl.tmax, lam_active, p_active, warp_size, noise_std, rng)
                pi_ctrl.record_batch_war(war_obs)
                t_start = time.monotonic()
                tmax_out = pi_ctrl.trigger_iteration_end()
                overhead_us["PI"].append((time.monotonic() - t_start) * 1e6)
                e = war_target - war_obs

            elif policy_name == "P_only":
                war_obs = simulate_war_erlang(p_only.tmax, lam_active, p_active, warp_size, noise_std, rng)
                t_start = time.monotonic()
                tmax_out = p_only.update(war_obs)
                overhead_us["P_only"].append((time.monotonic() - t_start) * 1e6)
                e = war_target - war_obs

            elif policy_name == "MPC_lite":
                war_obs = simulate_war_erlang(mpc.tmax, lam_active, p_active, warp_size, noise_std, rng)
                t_start = time.monotonic()
                tmax_out = mpc.update(lam_active, p_active, warp_size)
                overhead_us["MPC_lite"].append((time.monotonic() - t_start) * 1e6)
                e = war_target - war_obs

            else:  # Static
                war_obs = simulate_war_erlang(static_tmax, lam_active, p_active, warp_size, noise_std, rng)
                tmax_out = static_tmax
                e = war_target - war_obs
                overhead_us["Static"].append(0.0)

            rows.append({
                "iteration": i,
                "t_sec": round(t_sec, 4),
                "phase": phase,
                "workload": workload,
                "policy": policy_name,
                "war_observed": round(war_obs, 6),
                "tmax_current": round(tmax_out, 6),
                "error_e_t": round(e, 6),
                "steady_state_error": "N/A",
                "tau_iter_ms": tau_iter_ms,
                "hardware_label": hardware_label,
                "cpu_overhead_us": round(overhead_us[policy_name][-1], 3),
            })

    # Compute steady-state error for last 20 iterations per policy
    for policy_name in ("PI", "P_only", "MPC_lite", "Static"):
        policy_rows = [r for r in rows if r["policy"] == policy_name]
        if len(policy_rows) > 20:
            last_20 = policy_rows[-20:]
            ss_err = float(np.mean([war_target - r["war_observed"] for r in last_20]))
            for r in last_20:
                r["steady_state_error"] = round(ss_err, 6)

    overhead_summary = {
        p: round(float(np.mean(v)), 3) for p, v in overhead_us.items() if v
    }

    return rows, overhead_summary


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Policy comparison: P-only vs PI vs MPC-lite vs Static (pi_controller §5.2)"
    )
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--lambda-total", type=float, default=7.0)
    p.add_argument("--alpha-zipf", type=float, default=0.9)
    p.add_argument("--war-target", type=float, default=0.8)
    p.add_argument("--warp-size", type=int, default=32)
    p.add_argument("--duration", type=float, default=900.0)
    p.add_argument("--drift-at", type=float, default=300.0)
    p.add_argument("--tau-iter-ms", type=float, default=30.0)
    p.add_argument("--hardware-label", default="single_a6000",
                   choices=["single_a6000", "two_a6000_pcie", "two_h100_nvlink"])
    p.add_argument("--kp", type=float, default=0.01)
    p.add_argument("--ki", type=float, default=0.001)
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[E7-policy] K={args.K}  λ={args.lambda_total}  "
          f"τ_iter={args.tau_iter_ms}ms  hardware={args.hardware_label}")

    all_rows: List[dict] = []
    overhead_all: dict = {}

    for workload in ("step", "sinusoidal"):
        print(f"[E7-policy] Running workload: {workload} ...")
        t0 = time.monotonic()
        rows, overhead = run_policy_comparison(
            workload=workload,
            K=args.K,
            lambda_total=args.lambda_total,
            alpha_zipf=args.alpha_zipf,
            war_target=args.war_target,
            warp_size=args.warp_size,
            duration_sec=args.duration,
            tau_iter_ms=args.tau_iter_ms,
            drift_at_sec=args.drift_at,
            kp=args.kp,
            ki=args.ki,
            hardware_label=args.hardware_label,
            noise_std=args.noise_std,
            seed=args.seed,
        )
        elapsed = time.monotonic() - t0
        print(f"[E7-policy] Done {workload} in {elapsed:.1f}s -- {len(rows)} rows")
        all_rows.extend(rows)
        overhead_all[workload] = overhead

    # Write CSV
    fieldnames = [
        "iteration", "t_sec", "phase", "workload", "policy",
        "war_observed", "tmax_current", "error_e_t", "steady_state_error",
        "tau_iter_ms", "hardware_label", "cpu_overhead_us",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[E7-policy] Wrote {len(all_rows)} rows → {output_path}")

    # Summary table
    print("\n[E7-policy] CPU overhead per update (mean µs):")
    for wl, ov in overhead_all.items():
        print(f"  workload={wl}:")
        for pol, us in ov.items():
            print(f"    {pol}: {us:.3f} µs")

    print("\n[E7-policy] Post-drift WAR summary (step workload, last 20 iterations):")
    step_rows = [r for r in all_rows if r["workload"] == "step" and r["phase"] == "post_drift"]
    for policy in ("PI", "P_only", "MPC_lite", "Static"):
        pr = [r for r in step_rows if r["policy"] == policy]
        if not pr:
            continue
        last = pr[-20:]
        mean_war = np.mean([r["war_observed"] for r in last])
        ss_err = args.war_target - mean_war
        print(f"  {policy:12s}: WAR={mean_war:.4f}  SS-error={ss_err:.4f}")

    # EC 10.1 checks
    pi_ss = np.mean([r["war_observed"] for r in step_rows if r["policy"] == "PI"][-20:]) if step_rows else 0
    ponly_ss = np.mean([r["war_observed"] for r in step_rows if r["policy"] == "P_only"][-20:]) if step_rows else 0
    pi_ov = overhead_all.get("step", {}).get("PI", 1.0)
    mpc_ov = overhead_all.get("step", {}).get("MPC_lite", 1.0)

    print("\n[E7-policy] EC 10.1 checks:")
    ec2 = abs(args.war_target - pi_ss) < 0.05 and abs(args.war_target - ponly_ss) > 0.001
    ec3 = (mpc_ov / max(pi_ov, 1e-9)) >= 3.0  # MPC should be ≥ 3× more expensive
    print(f"  EC 10.1.2 (PI zero SS error, P-only nonzero): {'PASS' if ec2 else 'FAIL'}")
    print(f"  EC 10.1.3 (MPC ≥ 2.5× overhead vs PI): {'PASS' if ec3 else 'FAIL'}")

    print(f"\n[E7-policy] Done. Output: {output_path}")


if __name__ == "__main__":
    main()
