"""
e9_correlation.py -- E9 Autocorrelation + Burst Characterization Experiment (impl_8, §6.1–6.4)

Runs the full E9 workload characterization pipeline for one hardware configuration:
  1. Autocorrelation analysis (§6.1) -- compute rho(k,k,tau) for BurstGPT and synthetic
  2. Burst distribution analysis (§6.2) -- mean/P90 burst length, D_k(hardware)
  3. Non-i.i.d. score comparison (§7.2) -- BurstGPT vs. synthetic
  4. Optional: live replay metrics (§6.3) -- calls replay_harness.py if endpoint given
  5. Optional: WAR comparison BurstGPT vs. synthetic (§6.4) -- computes Δ_WAR_burst

Theorem 8.9 pass condition: BurstGPT rho > 0 AND WAR_BurstGPT > WAR_i.i.d.

Cross-hardware usage (same script, different K and tau_iter_ms):

  Single RTX A6000 (K=4, tau_iter ≈ 30ms):
    python scripts/experiments/e9_correlation.py \\
        --burstgpt-trace data/burstgpt/burstgpt_k4.jsonl \\
        --synthetic-trace workloads/synthetic_k4_iid.jsonl \\
        --K 4 --tau-iter-ms 30 --hardware-label a6000_single \\
        --output-dir results/impl_8/a6000_single

  Two RTX A6000 PCIe (K=16, tau_iter ≈ 100ms):
    python scripts/experiments/e9_correlation.py \\
        --burstgpt-trace data/burstgpt/burstgpt_k16.jsonl \\
        --synthetic-trace workloads/synthetic_k16_iid.jsonl \\
        --K 16 --tau-iter-ms 100 --hardware-label two_a6000_pcie \\
        --output-dir results/impl_8/two_a6000_pcie

  Two H100 NVLink (K=32, tau_iter ≈ 5ms):
    python scripts/experiments/e9_correlation.py \\
        --burstgpt-trace data/burstgpt/burstgpt_k32.jsonl \\
        --synthetic-trace workloads/synthetic_k32_iid.jsonl \\
        --K 32 --tau-iter-ms 5 --hardware-label two_h100_nvlink \\
        --output-dir results/impl_8/two_h100_nvlink

  With live replay (requires running vLLM server):
    python scripts/experiments/e9_correlation.py \\
        --burstgpt-trace data/burstgpt/burstgpt_k4_30min.jsonl \\
        --K 4 --tau-iter-ms 30 --hardware-label a6000_single \\
        --endpoint http://localhost:8000/v1/completions \\
        --speed-multiplier 5.0 \\
        --output-dir results/impl_8/a6000_single

Outputs in --output-dir:
  burstgpt_autocorr.csv         -- per-adapter ACF for BurstGPT
  synthetic_autocorr.csv        -- per-adapter ACF for synthetic (if --synthetic-trace)
  burst_distribution.csv        -- burst length stats + D_k
  noniid_scores.csv             -- non-i.i.d. score comparison table
  burstgpt_replay.csv           -- live replay results (if --endpoint)
  synthetic_replay.csv          -- synthetic replay results (if --endpoint + --synthetic-trace)
  delta_war_burst.csv           -- Δ_WAR_burst = WAR_BurstGPT - WAR_i.i.d.
  e9_summary.csv                -- one-row summary (pass/fail for EC §10.1/10.2/10.3)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# Import analysis functions directly
from analysis.workload_autocorrelation import (
    compute_adapter_autocorrelation,
    noniid_score,
    adapter_persistence,
    load_jsonl as acf_load_jsonl,
)
from analysis.burst_distribution import (
    compute_burst_length_distribution,
    burst_stats,
    compute_dk,
    adapter_persistence as burst_persistence,
    burst_exploitation_rate,
    identify_burst_windows,
)


# ---------------------------------------------------------------------------
# Synthetic i.i.d. trace generation
# ---------------------------------------------------------------------------

def generate_synthetic_iid_trace(
    n_requests: int,
    K: int,
    lambda_total: float,
    zipf_alpha: float = 0.9,
    seed: int = 42,
) -> List[dict]:
    """
    Generate synthetic i.i.d. Poisson + Zipf trace matching BurstGPT statistics.
    Used as the null-hypothesis (uncorrelated) control.
    """
    rng = np.random.default_rng(seed)

    # Zipf adapter probabilities
    ranks = np.arange(1, K + 1, dtype=float)
    probs = ranks ** (-zipf_alpha)
    probs /= probs.sum()

    # Poisson inter-arrival times
    mean_inter_arrival_s = 1.0 / lambda_total
    inter_arrivals = rng.exponential(mean_inter_arrival_s, size=n_requests)
    arrival_times_s = np.cumsum(inter_arrivals)

    # i.i.d. adapter selection
    adapter_ids = rng.choice(K, size=n_requests, p=probs)

    records = []
    for i in range(n_requests):
        records.append({
            "request_id": i,
            "adapter_id": int(adapter_ids[i]),
            "arrival_time_ms": round(float(arrival_times_s[i]) * 1000.0, 3),
            "prompt_len": 64,
            "output_len": 32,
        })
    return records


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_jsonl(records: List[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def save_csv(rows: List[dict], path: str):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def run_autocorrelation(
    trace: List[dict],
    label: str,
    K: int,
    max_lag: int,
    output_path: str,
) -> dict:
    """Run autocorrelation analysis and return summary dict."""
    arrival_sequence = [r["adapter_id"] for r in trace]
    print(f"  [{label}] Computing ACF (max_lag={max_lag}) ...", flush=True)

    autocorrs = compute_adapter_autocorrelation(arrival_sequence, max_lag=max_lag)
    score = noniid_score(arrival_sequence)
    persist = adapter_persistence(arrival_sequence)

    adapters = sorted(set(arrival_sequence))
    rows = []
    for k in adapters:
        acf_arr = autocorrs.get(k, np.zeros(max_lag + 1))
        for lag in range(max_lag + 1):
            rows.append({
                "label": label,
                "adapter_id": k,
                "lag": lag,
                "autocorrelation": round(float(acf_arr[lag]), 6),
                "noniid_score": round(score, 6),
                "persistence_rate": round(persist, 6),
            })

    save_csv(rows, output_path)
    print(f"    noniid_score={score:.4f}, persistence={persist:.4f}", flush=True)

    lag1_per_adapter = {}
    for k in adapters:
        acf_arr = autocorrs.get(k, np.zeros(max_lag + 1))
        lag1_per_adapter[k] = float(acf_arr[1]) if len(acf_arr) > 1 else 0.0

    return {
        "label": label,
        "noniid_score": score,
        "persistence_rate": persist,
        "lag1_primary": lag1_per_adapter.get(adapters[0], 0.0) if adapters else 0.0,
        "n_adapters_rho_gt_01": sum(1 for v in lag1_per_adapter.values() if v > 0.1),
    }


def run_burst_analysis(
    trace: List[dict],
    K: int,
    tau_iter_ms: float,
    hardware_label: str,
    output_path: str,
) -> List[dict]:
    """Run burst distribution analysis and return per-adapter rows."""
    arrival_sequence = [r["adapter_id"] for r in trace]
    span_s = (trace[-1]["arrival_time_ms"] - trace[0]["arrival_time_ms"]) / 1000.0
    if span_s <= 0:
        span_s = len(trace) / 1.0

    adapter_counts: Dict[int, int] = {}
    for a in arrival_sequence:
        adapter_counts[a] = adapter_counts.get(a, 0) + 1
    lambda_k_map = {k: cnt / span_s for k, cnt in adapter_counts.items()}

    burst_dist = compute_burst_length_distribution(arrival_sequence)
    persist = burst_persistence(arrival_sequence)

    rows = []
    for k in sorted(adapter_counts):
        blens = burst_dist.get(k, [])
        stats = burst_stats(blens)
        lam_k = lambda_k_map.get(k, 0.0)
        dk = compute_dk(blens, lam_k, tau_iter_ms)
        rows.append({
            "hardware_label": hardware_label,
            "K": K,
            "tau_iter_ms": tau_iter_ms,
            "adapter_id": k,
            "lambda_k_req_s": round(lam_k, 4),
            "burst_mean_len": round(stats["mean"], 3),
            "burst_p90_len": round(stats["p90"], 3),
            "burst_count": stats["count"],
            "D_k_decisions_per_burst": round(dk, 2),
            "persistence_rate": round(persist, 4),
        })

    save_csv(rows, output_path)
    return rows


def run_replay(
    trace: List[dict],
    label: str,
    endpoint: str,
    adapter_prefix: str,
    model: str,
    speed_multiplier: float,
    timeout_s: float,
    output_path: str,
    summary_path: str,
) -> Optional[dict]:
    """Run live replay and return WAR summary dict."""
    import asyncio
    sys.path.insert(0, str(_ROOT / "scripts"))
    from replay_harness import replay_trace_async, estimate_war_from_replay

    print(f"  [{label}] Replaying {len(trace)} requests at {speed_multiplier}× ...",
          flush=True)

    results = asyncio.run(replay_trace_async(
        trace=trace,
        endpoint=endpoint,
        model_name=model,
        adapter_prefix=adapter_prefix,
        speed_multiplier=speed_multiplier,
        timeout_s=timeout_s,
    ))

    n_ok = sum(1 for r in results if r.get("success"))
    n_err = len(results) - n_ok
    print(f"    {n_ok}/{len(results)} success, {n_err} errors", flush=True)

    war_stats = estimate_war_from_replay(results)
    ttfts = [r["ttft_ms"] for r in results if r.get("success") and r.get("ttft_ms", -1) > 0]

    summary = {
        "label": label,
        "n_requests": len(results),
        "n_success": n_ok,
        "error_rate": round(n_err / max(len(results), 1), 4),
        "war_mean": war_stats["war_mean"],
        "war_std": war_stats["war_std"],
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 1) if ttfts else -1,
        "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 1) if ttfts else -1,
    }
    print(f"    WAR={war_stats['war_mean']:.4f}, "
          f"TTFT P50={summary['ttft_p50_ms']:.0f}ms, "
          f"P99={summary['ttft_p99_ms']:.0f}ms", flush=True)

    from replay_harness import save_csv as rs_save_csv
    rs_save_csv(results, output_path)
    save_csv([summary], summary_path)
    return summary


# ---------------------------------------------------------------------------
# Pass condition checks
# ---------------------------------------------------------------------------

def check_ec_10_1(
    burstgpt_acf: dict,
    synthetic_acf: Optional[dict],
    war_burstgpt: Optional[float],
    war_iid: Optional[float],
    replay_war: Optional[float],
    replay_war_predicted: Optional[float],
) -> dict:
    """
    Check EC §10.1 conditions (Single A6000 gate for Part B).
    Returns dict of {condition: bool}.
    """
    conditions = {}

    # EC1: BurstGPT rho > 0.1 for >= 2 adapters
    n_positive = burstgpt_acf.get("n_adapters_rho_gt_01", 0)
    conditions["ec1_noniid_rho"] = n_positive >= 2

    # EC2: WAR_BurstGPT > WAR_i.i.d. + 0.05 in >= 2/3 Zipf configs
    # (checked externally across multiple alpha runs; here check single run)
    if war_burstgpt is not None and war_iid is not None:
        conditions["ec2_war_burstgpt_gt_iid"] = (war_burstgpt - war_iid) >= 0.05
    else:
        conditions["ec2_war_burstgpt_gt_iid"] = None

    # EC3: Replay harness WAR matches offline prediction within ±0.03
    if replay_war is not None and replay_war_predicted is not None:
        conditions["ec3_harness_validated"] = abs(replay_war - replay_war_predicted) <= 0.03
    else:
        conditions["ec3_harness_validated"] = None

    # EC7: Non-i.i.d. score >= 0.1 for BurstGPT
    conditions["ec7_noniid_score"] = burstgpt_acf.get("noniid_score", 0.0) >= 0.1

    # Synthetic control: score <= 0.05
    if synthetic_acf is not None:
        conditions["ec7_synthetic_iid"] = synthetic_acf.get("noniid_score", 0.0) <= 0.05
    else:
        conditions["ec7_synthetic_iid"] = None

    return conditions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="E9 autocorrelation + burst characterization experiment (impl_8)"
    )
    p.add_argument("--burstgpt-trace", required=True,
                   help="BurstGPT JSONL trace with adapter_id (from assign_adapters.py)")
    p.add_argument("--synthetic-trace", default=None,
                   help="Synthetic i.i.d. JSONL trace (auto-generated if not given)")
    p.add_argument("--K", type=int, default=4,
                   help="Number of adapters (4=A6000, 16=PCIe, 32=NVLink)")
    p.add_argument("--tau-iter-ms", type=float, default=30.0,
                   help="Measured tau_iter in ms for D_k computation")
    p.add_argument("--hardware-label", default="a6000_single",
                   help="Hardware label (a6000_single, two_a6000_pcie, two_h100_nvlink)")
    p.add_argument("--zipf-alpha", type=float, default=0.9,
                   help="Zipf alpha for synthetic trace generation")
    p.add_argument("--lambda-total", type=float, default=None,
                   help="Total arrival rate (req/s) for synthetic trace; "
                        "default: measured from BurstGPT trace")
    p.add_argument("--max-lag", type=int, default=60,
                   help="Maximum ACF lag")
    p.add_argument("--output-dir", required=True,
                   help="Directory for output CSV files")
    # Live replay options
    p.add_argument("--endpoint", default=None,
                   help="vLLM endpoint for live replay (skip if no server)")
    p.add_argument("--model", default="llama-7b",
                   help="Base model name")
    p.add_argument("--adapter-prefix", default="adapter_",
                   help="LoRA adapter name prefix")
    p.add_argument("--speed-multiplier", type=float, default=5.0,
                   help="Replay speed multiplier")
    p.add_argument("--timeout-s", type=float, default=120.0,
                   help="Per-request HTTP timeout")
    # Segment
    p.add_argument("--segment-minutes", type=float, default=30.0,
                   help="Replay segment duration in minutes (0 = use full trace)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load BurstGPT trace
    # -----------------------------------------------------------------------
    print(f"\n=== Loading BurstGPT trace: {args.burstgpt_trace} ===", flush=True)
    burstgpt_full = acf_load_jsonl(args.burstgpt_trace)
    if not burstgpt_full:
        print("[ERROR] Empty BurstGPT trace.")
        sys.exit(1)

    # Extract segment for replay
    if args.segment_minutes > 0:
        seg_ms = args.segment_minutes * 60_000.0
        t0_ms = burstgpt_full[0]["arrival_time_ms"]
        burstgpt_seg = [r for r in burstgpt_full
                        if r["arrival_time_ms"] - t0_ms <= seg_ms]
        if not burstgpt_seg:
            burstgpt_seg = burstgpt_full
    else:
        burstgpt_seg = burstgpt_full

    span_s = (burstgpt_seg[-1]["arrival_time_ms"] - burstgpt_seg[0]["arrival_time_ms"]) / 1000.0
    lambda_measured = len(burstgpt_seg) / max(span_s, 1.0)
    lambda_total = args.lambda_total or lambda_measured
    print(f"  {len(burstgpt_seg):,} requests, span={span_s:.0f}s, "
          f"lambda_mean={lambda_measured:.2f} req/s", flush=True)

    # -----------------------------------------------------------------------
    # Load or generate synthetic trace
    # -----------------------------------------------------------------------
    if args.synthetic_trace:
        print(f"\n=== Loading synthetic trace: {args.synthetic_trace} ===", flush=True)
        synthetic = acf_load_jsonl(args.synthetic_trace)
    else:
        print(f"\n=== Generating synthetic i.i.d. trace (K={args.K}, "
              f"lambda={lambda_total:.2f}, alpha={args.zipf_alpha}) ===", flush=True)
        n_synthetic = min(len(burstgpt_seg), 5000)
        synthetic = generate_synthetic_iid_trace(
            n_requests=n_synthetic,
            K=args.K,
            lambda_total=lambda_total,
            zipf_alpha=args.zipf_alpha,
        )
        syn_path = str(out_dir / f"synthetic_k{args.K}_iid.jsonl")
        save_jsonl(synthetic, syn_path)
        print(f"  Synthetic trace saved → {syn_path}", flush=True)

    # -----------------------------------------------------------------------
    # Step 1: Autocorrelation analysis
    # -----------------------------------------------------------------------
    print(f"\n=== Step 1: Autocorrelation Analysis ===", flush=True)
    burstgpt_acf = run_autocorrelation(
        trace=burstgpt_seg,
        label="BurstGPT",
        K=args.K,
        max_lag=args.max_lag,
        output_path=str(out_dir / "burstgpt_autocorr.csv"),
    )
    synthetic_acf = run_autocorrelation(
        trace=synthetic,
        label="Synthetic_i.i.d.",
        K=args.K,
        max_lag=args.max_lag,
        output_path=str(out_dir / "synthetic_autocorr.csv"),
    )

    # -----------------------------------------------------------------------
    # Step 2: Burst distribution analysis
    # -----------------------------------------------------------------------
    print(f"\n=== Step 2: Burst Distribution Analysis ===", flush=True)
    burst_rows = run_burst_analysis(
        trace=burstgpt_seg,
        K=args.K,
        tau_iter_ms=args.tau_iter_ms,
        hardware_label=args.hardware_label,
        output_path=str(out_dir / "burst_distribution.csv"),
    )
    primary_adapter = (
        max({r["adapter_id"] for r in burstgpt_seg},
            key=lambda k: sum(1 for r in burstgpt_seg if r["adapter_id"] == k))
    )
    dk_primary = next(
        (r["D_k_decisions_per_burst"] for r in burst_rows
         if r["adapter_id"] == primary_adapter), 0.0
    )
    print(f"  Primary adapter: {primary_adapter}, D_k = {dk_primary:.1f} decisions/burst",
          flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Non-i.i.d. score comparison
    # -----------------------------------------------------------------------
    print(f"\n=== Step 3: Non-i.i.d. Score Comparison ===", flush=True)
    noniid_rows = [
        {
            "trace": "BurstGPT",
            "hardware_label": args.hardware_label,
            "K": args.K,
            "noniid_score": round(burstgpt_acf["noniid_score"], 6),
            "lag1_primary_adapter": round(burstgpt_acf["lag1_primary"], 6),
            "n_adapters_rho_gt_01": burstgpt_acf["n_adapters_rho_gt_01"],
            "persistence_rate": round(burstgpt_acf["persistence_rate"], 6),
        },
        {
            "trace": "Synthetic_i.i.d.",
            "hardware_label": args.hardware_label,
            "K": args.K,
            "noniid_score": round(synthetic_acf["noniid_score"], 6),
            "lag1_primary_adapter": round(synthetic_acf["lag1_primary"], 6),
            "n_adapters_rho_gt_01": synthetic_acf["n_adapters_rho_gt_01"],
            "persistence_rate": round(synthetic_acf["persistence_rate"], 6),
        },
    ]
    save_csv(noniid_rows, str(out_dir / "noniid_scores.csv"))
    print(f"  BurstGPT noniid_score = {burstgpt_acf['noniid_score']:.4f} "
          f"({'≥0.1 PASS' if burstgpt_acf['noniid_score'] >= 0.1 else '<0.1 WARN'})",
          flush=True)
    print(f"  Synthetic noniid_score = {synthetic_acf['noniid_score']:.4f} "
          f"({'≤0.05 PASS' if synthetic_acf['noniid_score'] <= 0.05 else '>0.05 WARN'})",
          flush=True)

    # -----------------------------------------------------------------------
    # Step 4: Live replay (optional)
    # -----------------------------------------------------------------------
    war_burstgpt: Optional[float] = None
    war_iid: Optional[float] = None

    if args.endpoint:
        print(f"\n=== Step 4: Live Replay (endpoint={args.endpoint}) ===", flush=True)

        burstgpt_summary = run_replay(
            trace=burstgpt_seg,
            label="BurstGPT",
            endpoint=args.endpoint,
            adapter_prefix=args.adapter_prefix,
            model=args.model,
            speed_multiplier=args.speed_multiplier,
            timeout_s=args.timeout_s,
            output_path=str(out_dir / "burstgpt_replay.csv"),
            summary_path=str(out_dir / "burstgpt_replay_summary.csv"),
        )
        war_burstgpt = burstgpt_summary["war_mean"] if burstgpt_summary else None

        synthetic_summary = run_replay(
            trace=synthetic,
            label="Synthetic_i.i.d.",
            endpoint=args.endpoint,
            adapter_prefix=args.adapter_prefix,
            model=args.model,
            speed_multiplier=args.speed_multiplier,
            timeout_s=args.timeout_s,
            output_path=str(out_dir / "synthetic_replay.csv"),
            summary_path=str(out_dir / "synthetic_replay_summary.csv"),
        )
        war_iid = synthetic_summary["war_mean"] if synthetic_summary else None

        # Δ_WAR_burst
        delta_war = (war_burstgpt - war_iid) if (war_burstgpt and war_iid) else None
        print(f"\n  Δ_WAR_burst = WAR_BurstGPT - WAR_i.i.d. = "
              f"{war_burstgpt:.4f} - {war_iid:.4f} = "
              f"{delta_war:.4f}" if delta_war is not None else "  Δ_WAR_burst = N/A",
              flush=True)

        delta_war_rows = [{
            "hardware_label": args.hardware_label,
            "K": args.K,
            "tau_iter_ms": args.tau_iter_ms,
            "zipf_alpha": args.zipf_alpha,
            "war_burstgpt": round(war_burstgpt, 4) if war_burstgpt is not None else "",
            "war_iid": round(war_iid, 4) if war_iid is not None else "",
            "delta_war_burst": round(delta_war, 4) if delta_war is not None else "",
            "noniid_score_burstgpt": round(burstgpt_acf["noniid_score"], 6),
            "noniid_score_synthetic": round(synthetic_acf["noniid_score"], 6),
            "dk_primary": round(dk_primary, 2),
            "theorem_8_9_confirmed": (
                "YES" if (delta_war is not None and delta_war >= 0.05) else "NO"
            ),
        }]
        save_csv(delta_war_rows, str(out_dir / "delta_war_burst.csv"))
    else:
        print("\n  [skip] No --endpoint given; skipping live replay.", flush=True)
        delta_war = None

    # -----------------------------------------------------------------------
    # Step 5: EC pass-condition check
    # -----------------------------------------------------------------------
    print(f"\n=== EC Pass Conditions ({args.hardware_label}) ===", flush=True)
    conditions = check_ec_10_1(
        burstgpt_acf=burstgpt_acf,
        synthetic_acf=synthetic_acf,
        war_burstgpt=war_burstgpt,
        war_iid=war_iid,
        replay_war=None,
        replay_war_predicted=None,
    )
    for cond, passed in conditions.items():
        if passed is None:
            status = "(not checked -- requires live replay)"
        else:
            status = "PASS" if passed else "FAIL"
        print(f"  {cond}: {status}", flush=True)

    # Write one-row summary
    summary_row = {
        "hardware_label": args.hardware_label,
        "K": args.K,
        "tau_iter_ms": args.tau_iter_ms,
        "zipf_alpha": args.zipf_alpha,
        "burstgpt_noniid_score": round(burstgpt_acf["noniid_score"], 4),
        "synthetic_noniid_score": round(synthetic_acf["noniid_score"], 4),
        "burstgpt_lag1_primary": round(burstgpt_acf["lag1_primary"], 4),
        "synthetic_lag1_primary": round(synthetic_acf["lag1_primary"], 4),
        "dk_primary": round(dk_primary, 2),
        "war_burstgpt": round(war_burstgpt, 4) if war_burstgpt is not None else "",
        "war_iid": round(war_iid, 4) if war_iid is not None else "",
        "delta_war_burst": round(delta_war, 4) if delta_war is not None else "",
        **{f"ec_{k}": ("PASS" if v else ("FAIL" if v is not None else "N/A"))
           for k, v in conditions.items()},
    }
    save_csv([summary_row], str(out_dir / "e9_summary.csv"))
    print(f"\n  Summary written → {out_dir / 'e9_summary.csv'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
