"""
e8_bandit.py -- E8 Bandit Experiment: Whittle vs. Threshold vs. Oracle (impl_7, §5.1)

Primary experiment for impl_7 (AB4 in the ablation plan).

Compares three dispatch policies in a simulated multi-adapter Poisson environment:
    Threshold -- |Q_k| >= W OR age >= T_max^(k) (V21 baseline from impl_5)
    Whittle   -- Whittle index ranking (N3, Theorem 8.7)
    Oracle    -- Offline brute-force DP (K ≤ 4 only; provides 100% reference)
    FIFO      -- Round-robin (lower bound; single-A6000 only)
    Greedy    -- Dispatch fullest adapter (lower bound; single-A6000 only)

Cross-hardware usage -- same script, same logic, hardware determined by tau-iter-ms:

  Single RTX A6000 (TP=1, K ∈ {4,8,16}, primary):
    python scripts/experiments/e8_bandit.py \\
        --K 4 --distribution zipf --zipf-alpha 0.9 --lambda-total 7.0 \\
        --delta-t-ms 30 --tau-iter-ms 30 --hardware-label a6000_single \\
        --policies threshold whittle oracle fifo greedy \\
        --output results/impl_7/a6000_single/e8_bandit_results.csv

  Two RTX A6000 PCIe (TP=2, K=4, delta_t calibration first via §5.5a):
    python scripts/experiments/e8_bandit.py \\
        --K 4 --distribution zipf --zipf-alpha 0.9 --lambda-total 7.0 \\
        --delta-t-ms 100 --tau-iter-ms 100 --hardware-label two_a6000_pcie \\
        --policies threshold whittle oracle \\
        --output results/impl_7/two_a6000_pcie/e8_bandit_results.csv

  Two H100 NVLink (TP=2, K ∈ {4,8,16,32}, delta_t calibration first via §5.6a):
    python scripts/experiments/e8_bandit.py \\
        --K 4 --distribution zipf --zipf-alpha 0.9 --lambda-total 7.0 \\
        --delta-t-ms 5 --tau-iter-ms 5 --hardware-label two_h100_nvlink \\
        --policies threshold whittle oracle \\
        --output results/impl_7/two_h100_nvlink/e8_bandit_k4_32.csv

  delta_t calibration comparison (§5.5a -- wrong vs. correct delta_t):
    python scripts/experiments/e8_bandit.py \\
        --K 4 --distribution zipf --zipf-alpha 0.9 --lambda-total 7.0 \\
        --delta-t-ms 1 --tau-iter-ms 100 --hardware-label two_a6000_pcie \\
        --policies whittle --delta-t-sweep \\
        --output results/impl_7/two_a6000_pcie/delta_t_calibration.csv

  AB4 ablation -- Threshold+Erlang+PI vs. Whittle+Erlang+PI (§5.4):
    python scripts/experiments/e8_bandit.py \\
        --K 4 --distribution zipf --zipf-alpha 0.9 --lambda-total 7.0 \\
        --delta-t-ms 30 --tau-iter-ms 30 --hardware-label a6000_single \\
        --policies threshold whittle \\
        --erlang-tmax --pi-adaptive \\
        --output results/impl_7/a6000_single/ab4_ablation.csv

  Fairness test (§5.3 / §5.5d) -- Whittle + Fair cap vs. Whittle + NoFair:
    python scripts/experiments/e8_bandit.py \\
        --K 8 --distribution zipf --zipf-alpha 1.5 --lambda-total 10.0 \\
        --delta-t-ms 30 --tau-iter-ms 30 --hardware-label a6000_single \\
        --policies whittle --fairness-test \\
        --output results/impl_7/a6000_single/e8_fairness.csv

Outputs CSV with columns:
    hardware_label, K, distribution, zipf_alpha, lambda_total, delta_t_ms,
    policy, war_mean, war_std, pct_oracle, overhead_ms, throughput_tok_s,
    ttft_p50_ms, ttft_p99_ms, indexability_ok, n_ticks
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

# Make project root importable when run as a script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from adapterslots.dispatch.whittle import WhittleDispatcher
from adapterslots.dispatch.oracle import OracleScheduler
from adapterslots.dispatch.baselines import FIFODispatcher, GreedyFillDispatcher
from adapterslots.dispatch.erlang import compute_tmax_erlang


# ── Workload generators ────────────────────────────────────────────────────────

def _zipf_weights(K: int, alpha: float) -> List[float]:
    raw = [1.0 / (k + 1) ** alpha for k in range(K)]
    total = sum(raw)
    return [w / total for w in raw]


def _generate_arrivals(
    K: int,
    n_ticks: int,
    lambda_per_tick: List[float],
    distribution: str,
    rng: np.random.Generator,
) -> List[Dict[int, int]]:
    """Generate per-tick Poisson arrivals for K adapters over n_ticks."""
    arrivals = []
    for _ in range(n_ticks):
        tick: Dict[int, int] = {}
        for k in range(K):
            if distribution == "adversarial":
                # ABAB: alternate between adapter 0 and 1 each tick
                n = int(rng.poisson(lambda_per_tick[k] * 2)) if k % 2 == (_ % 2) else 0
            else:
                n = int(rng.poisson(lambda_per_tick[k]))
            if n > 0:
                tick[k] = n
        arrivals.append(tick)
    return arrivals


# ── Simulation engine ──────────────────────────────────────────────────────────

def _simulate_policy(
    policy_name: str,
    K: int,
    W: int,
    n_ticks: int,
    arrivals: List[Dict[int, int]],
    lambda_per_tick: List[float],
    delta_t_s: float,
    tau_iter_s: float,
    war_target: float,
    ttft_slo_s: float,
    use_erlang_tmax: bool,
    use_fair_cap: bool,
    rng: np.random.Generator,
    oracle_dp: Optional["OracleScheduler"] = None,
) -> Dict:
    """Simulate one dispatch policy for n_ticks. Returns metrics dict."""
    adapters = [str(k) for k in range(K)]

    # Per-adapter queue lengths
    queues: Dict[str, int] = {k: 0 for k in adapters}
    # Simulated enqueue time of oldest token in each queue (ticks, not seconds)
    oldest_tick: Dict[str, Optional[int]] = {k: None for k in adapters}

    # EWMA arrival rate estimates in tokens/sec (initialised uniformly)
    total_rate_s = sum(lambda_per_tick) / tau_iter_s  # convert tok/tick to tok/s
    lam_ewma: Dict[str, float] = {k: total_rate_s / K for k in adapters}

    # T_max per adapter (seconds)
    if use_erlang_tmax:
        tmax_k = {
            k: compute_tmax_erlang(W, lam_ewma[k], war_target, ttft_slo_s * 1000)
            for k in adapters
        }
    else:
        # Fallback: wait up to the TTFT SLO before timing out partial warps
        tmax_k = {k: ttft_slo_s for k in adapters}

    # Instantiate dispatcher
    if policy_name == "whittle":
        dispatcher = WhittleDispatcher(adapters, warp_size=W, delta_t=delta_t_s)
    elif policy_name == "fifo":
        dispatcher = FIFODispatcher(adapters)
    elif policy_name == "greedy":
        dispatcher = GreedyFillDispatcher(adapters)
    elif policy_name in ("threshold",):
        dispatcher = None
    elif policy_name == "oracle":
        dispatcher = None
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    aligned_dispatches = 0
    total_dispatches = 0
    total_tokens = 0
    ttft_samples: List[float] = []
    overhead_samples: List[float] = []

    for tick_idx, tick_arrivals in enumerate(arrivals):
        t_sim_s = tick_idx * tau_iter_s

        # ── 1. Enqueue arriving tokens ─────────────────────────────────────────
        for k_int, n in tick_arrivals.items():
            kid = str(k_int)
            if kid not in queues:
                continue
            queues[kid] += n
            total_tokens += n
            if oldest_tick[kid] is None:
                oldest_tick[kid] = tick_idx

        # EWMA update for ALL adapters every tick (including zero-arrival ticks).
        # Updating only on non-zero ticks biases the estimate to E[n|n>0]/τ = λ/(1-e^{-λ}),
        # which is 2× too high when λ_per_tick < 1, causing Erlang T_max to be too short.
        for k in adapters:
            n_k = tick_arrivals.get(int(k), 0)
            inst_rate = n_k / tau_iter_s
            lam_ewma[k] = 0.9 * lam_ewma[k] + 0.1 * inst_rate

        # ── 2. Update Erlang T_max if needed ──────────────────────────────────
        if use_erlang_tmax:
            for k in adapters:
                tmax_k[k] = compute_tmax_erlang(
                    W, max(lam_ewma[k], 1e-6), war_target, ttft_slo_s * 1000
                )

        # ── 3. Determine dispatch order ────────────────────────────────────────
        t_rank_start = time.monotonic()

        fill_fracs = {k: min(queues[k] / W, 1.0) for k in adapters}
        ages_s = {
            k: (tick_idx - oldest_tick[k]) * tau_iter_s if oldest_tick[k] is not None else 0.0
            for k in adapters
        }

        if policy_name == "threshold":
            # Prioritise: full warp first, then timed-out, then by fill fraction
            ranked = sorted(
                adapters,
                key=lambda k: (
                    int(queues[k] >= W),
                    int(oldest_tick[k] is not None and ages_s[k] >= tmax_k[k]),
                    fill_fracs[k],
                ),
                reverse=True,
            )
        elif policy_name == "oracle":
            # Oracle: use DP best_action for first choice; fall back to fill-fraction order
            if oracle_dp is not None:
                # Pass NEXT H ticks' arrivals -- current tick's arrivals already in queues
                future_h = arrivals[tick_idx + 1: tick_idx + 1 + oracle_dp.H]
                current_q = [queues[str(k)] for k in range(K)]
                best_k = oracle_dp.best_action(current_q, future_h)
            else:
                best_k = -1
            ranked = []
            if best_k >= 0:
                ranked.append(str(best_k))
            for k in sorted(
                adapters,
                key=lambda k: (int(queues[k] >= W), fill_fracs[k]),
                reverse=True,
            ):
                if k not in ranked:
                    ranked.append(k)
        elif policy_name == "whittle":
            ranked = dispatcher.rank_adapters(fill_fracs, lam_ewma)
            counts = {k: tick_arrivals.get(int(k), 0) for k in adapters}
            dispatcher.update_traffic_fractions(counts)
        else:
            ranked = dispatcher.rank_adapters(fill_fracs, lam_ewma)

        overhead_ms = (time.monotonic() - t_rank_start) * 1000.0
        overhead_samples.append(overhead_ms)

        # ── 4. Dispatch top-ranked adapter ─────────────────────────────────────
        dispatched = False
        for kid in ranked:
            if queues[kid] == 0:
                continue
            age_s = ages_s[kid]
            is_full = queues[kid] >= W
            is_timeout = (oldest_tick[kid] is not None) and (age_s >= tmax_k[kid])

            if not (is_full or is_timeout):
                continue

            # Record TTFT (age of oldest token at dispatch time)
            ttft_samples.append(age_s * 1000.0)  # ms

            if is_full:
                # Aligned warp dispatch: dispatch exactly W tokens
                n_warps = queues[kid] // W
                tokens_dispatched = n_warps * W
                # Fair cap: if token has been waiting longer than SLO, don't count aligned
                is_aligned = not (use_fair_cap and age_s > ttft_slo_s)
                if is_aligned:
                    aligned_dispatches += 1
                queues[kid] -= tokens_dispatched
            else:
                # Timeout dispatch: flush all partial tokens (not aligned)
                queues[kid] = 0

            # Reset enqueue time tracker
            oldest_tick[kid] = tick_idx if queues[kid] > 0 else None
            total_dispatches += 1
            dispatched = True
            break

        if not dispatched and any(queues[k] > 0 for k in adapters):
            # No adapter ready to dispatch (none full or timed out) -- idle tick
            pass

    # ── Compute metrics ────────────────────────────────────────────────────────
    war = aligned_dispatches / max(total_dispatches, 1)
    overhead_mean = float(np.mean(overhead_samples)) if overhead_samples else 0.0
    ttft_arr = np.array(ttft_samples) if ttft_samples else np.array([0.0])
    ttft_p50 = float(np.percentile(ttft_arr, 50))
    ttft_p99 = float(np.percentile(ttft_arr, 99))
    throughput = total_tokens / max(n_ticks * tau_iter_s, 1e-9)

    return {
        "war_mean": war,
        "war_std": 0.0,
        "aligned_dispatches": aligned_dispatches,
        "total_dispatches": total_dispatches,
        "overhead_ms": overhead_mean,
        "ttft_p50_ms": ttft_p50,
        "ttft_p99_ms": ttft_p99,
        "throughput_tok_s": throughput,
        "n_ticks": n_ticks,
    }


# ── Main experiment runner ─────────────────────────────────────────────────────

def run_e8_bandit(
    K: int,
    distribution: str,
    zipf_alpha: float,
    lambda_total: float,
    tokens_per_req: int,
    delta_t_ms: float,
    tau_iter_ms: float,
    hardware_label: str,
    policies: List[str],
    n_ticks: int,
    war_target: float,
    ttft_slo_ms: float,
    use_erlang_tmax: bool,
    use_fair_cap: bool,
    delta_t_sweep: bool,
    seed: int,
    output_path: str,
) -> None:
    W = 32
    rng = np.random.default_rng(seed)
    tau_s = tau_iter_ms / 1000.0
    ttft_slo_s = ttft_slo_ms / 1000.0

    # lambda_total is in requests/sec.  Each in-flight request contributes
    # ~1 decode token per iteration to its adapter queue.  At steady state,
    # the average number of in-flight requests is lambda_total * avg_response_time,
    # where avg_response_time ≈ tokens_per_req × tau_iter.
    #
    # Effective token rate ≈ lambda_total * tokens_per_req (tokens/sec, all adapters).
    # This is the rate at which decode tokens arrive at the alignment buffer.
    effective_tok_rate = lambda_total * tokens_per_req  # tokens/sec total

    # Compute per-adapter arrival rates (tokens/tick)
    if distribution == "zipf":
        weights = _zipf_weights(K, zipf_alpha)
    elif distribution == "uniform":
        weights = [1.0 / K] * K
    elif distribution == "adversarial":
        weights = ([0.5] + [0.5 / (K - 1)] * (K - 1)) if K > 1 else [1.0]
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    lambda_per_tick = [w * effective_tok_rate * tau_s for w in weights]

    # Generate arrivals (single sequence reused across policies for fair comparison)
    arrivals = _generate_arrivals(K, n_ticks, lambda_per_tick, distribution, rng)

    # Instantiate oracle DP for K ≤ 4 (used as live policy and reference)
    oracle_dp: Optional[OracleScheduler] = None
    oracle_war: Optional[float] = None  # simulated WAR from oracle policy run
    if K <= 4 and "oracle" in policies:
        oracle_dp = OracleScheduler(W=W, K=K, horizon=min(20, n_ticks))

    rows: List[Dict] = []

    # Optionally sweep delta_t values to validate Proposition 8.10
    delta_t_values = [delta_t_ms]
    if delta_t_sweep:
        delta_t_values = [1.0, tau_iter_ms]  # wrong (1ms) vs. correct (tau_iter)

    for dt_ms in delta_t_values:
        dt_s = dt_ms / 1000.0

        # Run oracle first (if present) so oracle_war is available for pct_oracle
        policies_ordered = (
            ["oracle"] + [p for p in policies if p != "oracle"]
            if "oracle" in policies and K <= 4
            else [p for p in policies if p != "oracle"]
        )

        for policy in policies_ordered:
            if policy == "oracle" and K > 4:
                print(f"  [skip] oracle not tractable for K={K}")
                continue

            print(
                f"  [{hardware_label}] K={K} policy={policy:10s} "
                f"dist={distribution} delta_t={dt_ms:.1f}ms ...",
                flush=True,
            )

            metrics = _simulate_policy(
                policy_name=policy,
                K=K,
                W=W,
                n_ticks=n_ticks,
                arrivals=arrivals,
                lambda_per_tick=lambda_per_tick,
                delta_t_s=dt_s,
                tau_iter_s=tau_s,
                war_target=war_target,
                ttft_slo_s=ttft_slo_s,
                use_erlang_tmax=use_erlang_tmax,
                use_fair_cap=use_fair_cap,
                rng=np.random.default_rng(seed),
                oracle_dp=oracle_dp if policy == "oracle" else None,
            )

            war = metrics["war_mean"]
            # Capture oracle WAR so other policies can report pct_oracle
            if policy == "oracle":
                oracle_war = war

            # pct_oracle: how close this policy is to the oracle policy's WAR
            pct_oracle = (war / oracle_war * 100.0) if oracle_war and oracle_war > 0 else float("nan")

            # Validate indexability for Whittle
            idx_ok = True
            if policy == "whittle":
                wd = WhittleDispatcher([str(k) for k in range(K)], warp_size=W, delta_t=dt_s)
                for k in range(K):
                    if not wd.check_indexability(lambda_k=lambda_per_tick[k] / tau_s, p_k=weights[k]):
                        idx_ok = False
                        break

            rows.append({
                "hardware_label": hardware_label,
                "K": K,
                "distribution": distribution,
                "zipf_alpha": zipf_alpha,
                "lambda_total": lambda_total,
                "delta_t_ms": dt_ms,
                "tau_iter_ms": tau_iter_ms,
                "policy": policy,
                "war_mean": f"{war:.4f}",
                "war_std": f"{metrics['war_std']:.4f}",
                "pct_oracle": f"{pct_oracle:.1f}" if not math.isnan(pct_oracle) else "N/A",
                "oracle_war": f"{oracle_war:.4f}" if oracle_war is not None else "N/A",
                "overhead_ms": f"{metrics['overhead_ms']:.4f}",
                "throughput_tok_s": f"{metrics['throughput_tok_s']:.1f}",
                "ttft_p50_ms": f"{metrics['ttft_p50_ms']:.1f}",
                "ttft_p99_ms": f"{metrics['ttft_p99_ms']:.1f}",
                "indexability_ok": str(idx_ok),
                "n_ticks": metrics["n_ticks"],
            })

    # Write CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Written {len(rows)} rows → {out_path}")

    # Print summary table
    print(f"\n  {'Policy':12s}  {'WAR':>6s}  {'%Oracle':>8s}  {'Overhead(ms)':>13s}  {'Throughput':>12s}")
    print("  " + "-" * 60)
    for r in rows:
        print(
            f"  {r['policy']:12s}  {r['war_mean']:>6s}  {r['pct_oracle']:>8s}  "
            f"{r['overhead_ms']:>13s}  {r['throughput_tok_s']:>12s}"
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E8-bandit: Whittle vs. Threshold vs. Oracle dispatch comparison"
    )
    parser.add_argument("--K", type=int, default=4,
                        help="Number of adapters (default: 4)")
    parser.add_argument("--distribution", choices=["zipf", "uniform", "adversarial"],
                        default="zipf", help="Arrival distribution (default: zipf)")
    parser.add_argument("--zipf-alpha", type=float, default=0.9,
                        help="Zipf skew parameter (default: 0.9)")
    parser.add_argument("--lambda-total", type=float, default=7.0,
                        help="Total REQUEST arrival rate in req/s (default: 7.0)")
    parser.add_argument("--tokens-per-req", type=int, default=17,
                        help="Avg decode tokens per request; scales buffer token rate (default: 17; "
                             "gives WAR~0.7-0.8 for K=4 Zipf, tau=30ms, lambda=7 req/s with Erlang T_max)")
    parser.add_argument("--delta-t-ms", type=float, default=30.0,
                        help="Whittle scheduling tick interval ms; set to tau-iter-ms (default: 30)")
    parser.add_argument("--tau-iter-ms", type=float, default=30.0,
                        help="Decode iteration time ms; calibrate empirically (default: 30)")
    parser.add_argument("--hardware-label", type=str, default="a6000_single",
                        help="Hardware label for output (default: a6000_single)")
    parser.add_argument("--policies", nargs="+",
                        choices=["threshold", "whittle", "oracle", "fifo", "greedy"],
                        default=["threshold", "whittle", "oracle"],
                        help="Policies to evaluate (default: threshold whittle oracle)")
    parser.add_argument("--n-ticks", type=int, default=2000,
                        help="Number of simulation ticks (default: 2000)")
    parser.add_argument("--war-target", type=float, default=0.8,
                        help="WAR* target for Erlang T_max (default: 0.8)")
    parser.add_argument("--ttft-slo-ms", type=float, default=2000.0,
                        help="TTFT SLO in ms for fairness cap (default: 2000)")
    parser.add_argument("--erlang-tmax", action="store_true",
                        help="Use per-adapter Erlang T_max (impl_5); else use fixed 5-iter T_max")
    parser.add_argument("--pi-adaptive", action="store_true",
                        help="Enable PI-adaptive T_max (impl_6); requires --erlang-tmax")
    parser.add_argument("--fairness-test", action="store_true",
                        help="Run fairness comparison: Whittle+Fair vs. Whittle+NoFair")
    parser.add_argument("--delta-t-sweep", action="store_true",
                        help="Sweep delta_t values (1ms wrong vs. tau_iter correct)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default="results/impl_7/e8_bandit_results.csv",
                        help="Output CSV path")

    args = parser.parse_args()

    print(f"\nE8-bandit experiment")
    print(f"  Hardware     : {args.hardware_label}")
    print(f"  K            : {args.K}")
    print(f"  Dist         : {args.distribution} (α={args.zipf_alpha})")
    print(f"  λ_total      : {args.lambda_total} req/s × {args.tokens_per_req} tok/req "
          f"= {args.lambda_total * args.tokens_per_req:.0f} tok/s buffer rate")
    print(f"  delta_t      : {args.delta_t_ms} ms")
    print(f"  τ_iter       : {args.tau_iter_ms} ms")
    print(f"  Policies     : {args.policies}")
    print(f"  Ticks        : {args.n_ticks}")
    print(f"  Output       : {args.output}")
    print()

    if args.fairness_test:
        # Run Whittle+Fair vs. Whittle+NoFair
        for fair in [True, False]:
            suffix = "fair" if fair else "nofair"
            out = args.output.replace(".csv", f"_{suffix}.csv")
            run_e8_bandit(
                K=args.K,
                distribution=args.distribution,
                zipf_alpha=args.zipf_alpha,
                lambda_total=args.lambda_total,
                tokens_per_req=args.tokens_per_req,
                delta_t_ms=args.delta_t_ms,
                tau_iter_ms=args.tau_iter_ms,
                hardware_label=args.hardware_label,
                policies=["whittle"],
                n_ticks=args.n_ticks,
                war_target=args.war_target,
                ttft_slo_ms=args.ttft_slo_ms,
                use_erlang_tmax=args.erlang_tmax,
                use_fair_cap=fair,
                delta_t_sweep=False,
                seed=args.seed,
                output_path=out,
            )
    else:
        run_e8_bandit(
            K=args.K,
            distribution=args.distribution,
            zipf_alpha=args.zipf_alpha,
            lambda_total=args.lambda_total,
            tokens_per_req=args.tokens_per_req,
            delta_t_ms=args.delta_t_ms,
            tau_iter_ms=args.tau_iter_ms,
            hardware_label=args.hardware_label,
            policies=args.policies,
            n_ticks=args.n_ticks,
            war_target=args.war_target,
            ttft_slo_ms=args.ttft_slo_ms,
            use_erlang_tmax=args.erlang_tmax,
            use_fair_cap=True,
            delta_t_sweep=args.delta_t_sweep,
            seed=args.seed,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
