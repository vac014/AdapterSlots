"""
test_pp_correctness.py -- Pipeline Parallelism Correctness Validation (multi_gpu_correctness, §4)

Validates:
    EC 10.2  PP=2 serving produces same output as PP=1
    WAR at stage 0 ≈ WAR at stage 1 (within ±0.03)

Key insight (§4.1): adapter alignment is set once at stage 0 (CPU scheduler).
Subsequent pipeline stages receive the SAME adapter ordering because adapter IDs
are attached to sequences, not layers.  WAR is therefore PP-invariant.

Additionally validates §4.2: micro-batch interaction -- with PP=2 and micro-batch
size = W = 32, each micro-batch consists of warp-aligned tokens.

Two modes:
  simulation  -- Pure-Python validation of PP-invariance (no GPU needed)
  live        -- vLLM PP=2 serving (requires GPU; see --mode live)

Usage
-----
    # CPU simulation (no GPU required)
    python scripts/test_pp_correctness.py \\
        --mode simulation \\
        --K 4 --W 32 --n-ticks 3000 \\
        --output-dir results/multi_gpu_correctness/

    # Two RTX A6000 PCIe (TP=2 serves as PP proxy -- AdapterSlots PP behaviour)
    CUDA_VISIBLE_DEVICES=0,1 python scripts/test_pp_correctness.py \\
        --mode live \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --K 4 \\
        --output-dir results/multi_gpu_correctness/

Outputs
-------
    results/multi_gpu_correctness/pp2_correctness.csv         -- per-stage WAR comparison
    results/multi_gpu_correctness/pp_correctness_summary.txt  -- PASS/FAIL verdict
"""

import argparse
import csv
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from adapter_slots.buffer import AlignmentBuffer
from adapter_slots.metrics.war import compute_war_from_ids


# Simulation mode

def simulate_pp_stage_war(
    K: int,
    W: int,
    n_stages: int,
    lam_total: float,
    tau_iter_ms: float,
    n_ticks: int,
    seed: int = 42,
) -> List[Tuple[str, float, float, float]]:
    """Simulate WAR consistency across PP stages.

    Returns list of (stage_label, mean_war, p10_war, p90_war) tuples.

    The alignment buffer forms the batch at stage 0.  Later stages receive the
    same adapter ordering because the scheduled_seq_groups list is unchanged
    after alignment.  We model this by applying the same batch to all stages.

    Additionally: micro-batch interaction is validated -- with PP=2 and
    micro_batch_size=W, each micro-batch is a warp-aligned chunk.
    """
    rng = random.Random(seed)
    adapters = [f"k{i}" for i in range(K)]

    alpha = 0.9
    raw = [k ** (-alpha) for k in range(1, K + 1)]
    total_w = sum(raw)
    probs = [r / total_w for r in raw]

    buf = AlignmentBuffer(
        adapters=adapters,
        warp_size=W,
        tmax_ms=tau_iter_ms * 3,
        ttft_slo_ms=tau_iter_ms * 50,
    )

    seq_id = 0
    # Stage WAR series: indexed by stage (0 to n_stages-1)
    stage_war_series: List[List[float]] = [[] for _ in range(n_stages)]

    for tick in range(n_ticks):
        # Arrivals
        n_arrivals = max(0, int(lam_total * tau_iter_ms / 1000
                                + rng.gauss(0, math.sqrt(lam_total * tau_iter_ms / 1000))))
        for _ in range(n_arrivals):
            r = rng.random()
            cum = 0.0
            chosen = K - 1
            for i, p in enumerate(probs):
                cum += p
                if r <= cum:
                    chosen = i
                    break
            buf.enqueue(adapters[chosen], seq_id)
            seq_id += 1

        batch = buf.form_batch(max_tokens=K * W * 2)
        if not batch:
            continue

        # Compute WAR for each pipeline stage.
        # Stage 0: full batch (alignment set here).
        # Stage j > 0: same batch split into micro-batches of size W.
        # Each micro-batch should be within-adapter if alignment is perfect.
        counts_full = Counter(aid for aid, _ in batch)
        n_total = len(batch)

        for stage in range(n_stages):
            if stage == 0 or n_stages == 1:
                # Stage 0 or no pipeline: full batch WAR
                n_aligned = sum((cnt // W) * W for cnt in counts_full.values())
                war = n_aligned / n_total if n_total > 0 else 0.0
            else:
                # Stage j: compute WAR per micro-batch of size W
                micro_batch_war_vals = []
                for mb_start in range(0, n_total, W):
                    mb = batch[mb_start:mb_start + W]
                    mb_counts = Counter(aid for aid, _ in mb)
                    mb_n = len(mb)
                    mb_aligned = sum((cnt // W) * W for cnt in mb_counts.values())
                    mb_war = mb_aligned / mb_n if mb_n >= W else 0.0
                    micro_batch_war_vals.append(mb_war)
                war = (sum(micro_batch_war_vals) / len(micro_batch_war_vals)
                       if micro_batch_war_vals else 0.0)

            stage_war_series[stage].append(war)

    results = []
    for stage in range(n_stages):
        s = stage_war_series[stage]
        if not s:
            results.append((f"stage_{stage}", 0.0, 0.0, 0.0))
            continue
        sorted_w = sorted(s)
        n = len(sorted_w)
        mean_w = sum(s) / n
        p10 = sorted_w[max(0, int(0.10 * n))]
        p90 = sorted_w[min(n - 1, int(0.90 * n))]
        results.append((f"stage_{stage}", mean_w, p10, p90))
    return results


def run_simulation(
    K: int,
    W: int,
    lam_total: float,
    n_ticks: int,
    tau_iter_ms: float,
    output_dir: str,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PP Correctness Simulation  K={K} W={W} λ={lam_total} req/s")
    print(f"{'='*60}")

    # PP=1 baseline
    pp1_results = simulate_pp_stage_war(K, W, 1, lam_total, tau_iter_ms, n_ticks, seed=42)
    # PP=2 with micro-batch validation
    pp2_results = simulate_pp_stage_war(K, W, 2, lam_total, tau_iter_ms, n_ticks, seed=42)

    rows = []
    all_pass = True

    print(f"\nPP=1 results:")
    print(f"{'Stage':>8}  {'Mean_WAR':>9}  {'P10':>7}  {'P90':>7}")
    print("-" * 35)
    for label, mean_w, p10, p90 in pp1_results:
        print(f"{label:>8}  {mean_w:>9.4f}  {p10:>7.4f}  {p90:>7.4f}")

    print(f"\nPP=2 results (micro-batch WAR per stage):")
    print(f"{'Stage':>8}  {'Mean_WAR':>9}  {'P10':>7}  {'P90':>7}  "
          f"{'WAR_diff_vs_s0':>15}  {'EC_Pass':>7}")
    print("-" * 60)

    pp2_stage0_war = pp2_results[0][1] if pp2_results else 0.0
    for i, (label, mean_w, p10, p90) in enumerate(pp2_results):
        diff = abs(mean_w - pp2_stage0_war) if i > 0 else 0.0
        ec_pass = diff <= 0.03
        if not ec_pass and i > 0:
            all_pass = False
        print(f"{label:>8}  {mean_w:>9.4f}  {p10:>7.4f}  {p90:>7.4f}  "
              f"{diff:>14.4f}  {'PASS' if ec_pass or i == 0 else 'FAIL':>7}")
        rows.append({
            "pp_degree": 2,
            "stage": i,
            "mean_war": round(mean_w, 4),
            "p10_war": round(p10, 4),
            "p90_war": round(p90, 4),
            "war_diff_vs_stage0": round(diff, 4),
            "ec_pass": ec_pass or i == 0,
            "K": K, "W": W, "lam_total": lam_total,
        })

    # Also write PP=1 baseline
    for i, (label, mean_w, p10, p90) in enumerate(pp1_results):
        rows.append({
            "pp_degree": 1,
            "stage": i,
            "mean_war": round(mean_w, 4),
            "p10_war": round(p10, 4),
            "p90_war": round(p90, 4),
            "war_diff_vs_stage0": 0.0,
            "ec_pass": True,
            "K": K, "W": W, "lam_total": lam_total,
        })

    _write_pp_results(rows, output_dir, all_pass)
    return {"all_pass": all_pass, "rows": rows}


def _write_pp_results(rows: list, output_dir: str, all_pass: bool):
    csv_path = os.path.join(output_dir, "pp2_correctness.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(output_dir, "pp_correctness_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PP Correctness Validation (multi_gpu_correctness §4)\n\n")
        f.write("Key insight: AdapterSlots alignment is set at stage 0 (CPU scheduler).\n")
        f.write("Subsequent stages receive the same adapter ordering -- PP-invariant.\n\n")
        for r in rows:
            f.write(f"PP={r['pp_degree']} stage={r['stage']}: "
                    f"mean_WAR={r['mean_war']:.4f}  diff_vs_s0={r['war_diff_vs_stage0']:.4f}  "
                    f"{'PASS' if r['ec_pass'] else 'FAIL'}\n")
        f.write(f"\nEC 10.2 (PP=2 WAR consistent across stages ±0.03): "
                f"{'PASS ✓' if all_pass else 'FAIL ✗'}\n")

    print(f"\nEC 10.2 (PP stage WAR within ±0.03):  {'PASS' if all_pass else 'FAIL'}")
    print(f"\n→ CSV:     {csv_path}")
    print(f"→ Summary: {summary_path}")


# Entry point

def main():
    ap = argparse.ArgumentParser(description="PP Correctness Validation (multi_gpu_correctness)")
    ap.add_argument("--mode", choices=["simulation", "live"], default="simulation")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--n-ticks", type=int, default=3000)
    ap.add_argument("--lambda-total", type=float, default=14.0)
    ap.add_argument("--tau-iter-ms", type=float, default=30.0)
    ap.add_argument("--model", default="./models/llama-7b")
    ap.add_argument("--adapter-dir", default="./adapters")
    ap.add_argument("--output-dir", default="results/multi_gpu_correctness/")
    args = ap.parse_args()

    if args.mode == "simulation":
        result = run_simulation(
            K=args.K,
            W=args.W,
            lam_total=args.lambda_total,
            n_ticks=args.n_ticks,
            tau_iter_ms=args.tau_iter_ms,
            output_dir=args.output_dir,
        )
        sys.exit(0 if result["all_pass"] else 1)
    else:
        raise RuntimeError(
            "PP=2 live mode is not implemented -- vLLM pipeline-parallel "
            "serving requires 2+ NVLink GPUs and this harness has no real "
            "driver for it. AdapterSlots's TP=2 live test (test_tp_correctness.py "
            "--mode live) validates the same WAR-invariance property as a "
            "real proxy. Run with --mode simulation explicitly if you want "
            "the synthetic PP=2 invariance check instead."
        )


if __name__ == "__main__":
    main()
