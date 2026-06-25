"""
bench_decode_only.py -- E14.B4: Decode-phase isolation benchmark.

Measures throughput and WAR degradation under Distinct adapter pattern
(adversarial: request i → adapter i % K, maximally un-aligned).

Sweeps K and rate, comparing C0 (vLLM baseline) vs C3 (WAR+PredLFU) vs
C7 (full stack). Key metric: how well AdapterSlots C7 mitigates the Distinct penalty.

Usage:
    python bench_decode_only.py \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --num-adapters 4 10 20 50 \\
        --request-rates 3 7 10 \\
        --rank 32 --reps 3 \\
        --output-dir results/sota_evaluation/b4/

    python bench_decode_only.py --dry-run --output-dir /tmp/b4/
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import List, Optional


from bench import run_benchmark, _list_adapters, REPS, N_PROMPTS, WARMUP


def run_decode_sweep(
    model: str,
    adapter_dir: str,
    k_list: List[int],
    rate_list: List[float],
    rank: int,
    modes: List[str],
    output_dir: str,
    reps: int = REPS,
    seeds: List[int] = None,
    num_prompts: int = N_PROMPTS,
    warmup: int = WARMUP,
    tmax_ms: int = 90,
    wgkp_threshold: int = 8,
    port_base: int = 8120,
    dry_run: bool = False,
) -> dict:
    """Run Distinct-pattern sweep across K × rate × mode."""
    if seeds is None:
        seeds = [42, 43, 44]

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    all_results = {}
    port = port_base

    for K in k_list:
        adapter_dirs = _list_adapters(adapter_dir, K, rank)
        for rate in rate_list:
            for mode in modes:
                key = f"K{K}_r{rate}_m{mode}"
                out_path = out_base / f"{key}.json"

                if out_path.exists() and not dry_run:
                    print(f"  Skipping {key} (result exists)")
                    with open(out_path) as f:
                        all_results[key] = json.load(f)
                    continue

                backend = "adapterslots" if mode != "C0" else "vllm"
                actual_mode = mode if backend == "adapterslots" else "C0"

                print(f"\n  B4: K={K} rate={rate} mode={mode} backend={backend}")
                result = run_benchmark(
                    backend_name=backend,
                    mode=actual_mode,
                    model=model,
                    adapter_dirs=adapter_dirs,
                    num_adapters=K,
                    rank=rank,
                    target_modules="all_linear",
                    request_rate=rate,
                    dataset="sharegpt",
                    pattern="distinct",
                    num_prompts=num_prompts,
                    warmup=warmup,
                    reps=reps,
                    seeds=seeds,
                    output=str(out_path),
                    tp=1,
                    tmax_ms=tmax_ms if backend == "adapterslots" else None,
                    wgkp_threshold=wgkp_threshold if backend == "adapterslots" else None,
                    port=port,
                    dry_run=dry_run,
                )
                all_results[key] = result
                port += 10

    # Write combined summary
    summary_path = out_base / "b4_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  B4 combined summary written to {summary_path}")

    _print_b4_table(all_results, k_list, rate_list, modes)
    return all_results


def _print_b4_table(results: dict, k_list: List[int], rate_list: List[float], modes: List[str]) -> None:
    """Print WAR degradation table for Distinct pattern."""
    print("\n--- E14.B4: Distinct-pattern decode degradation ---")
    print(f"{'K':>4}  {'rate':>5}  " + "  ".join(f"{m:>12}" for m in modes))
    for K in k_list:
        for rate in rate_list:
            row = []
            baseline_tps = None
            for mode in modes:
                key = f"K{K}_r{rate}_m{mode}"
                r = results.get(key, {})
                tps = r.get("summary", {}).get("throughput_toks_mean", float("nan"))
                if mode in ("C0", "vllm"):
                    baseline_tps = tps
                row.append(tps)

            def fmt(t, b):
                if b and b > 0:
                    return f"{t:.0f}({t/b:.2f}×)"
                return f"{t:.0f}"

            vals = "  ".join(
                fmt(row[i], baseline_tps) if modes[i] not in ("C0", "vllm") else f"{row[i]:.0f}"
                for i in range(len(modes))
            )
            print(f"{K:>4}  {rate:>5.1f}  {vals}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E14.B4: Decode-phase isolation (Distinct pattern)")
    ap.add_argument("--model", default="./models/llama-7b")
    ap.add_argument("--adapter-dir", default="./adapters")
    ap.add_argument("--num-adapters", nargs="+", type=int, default=[4, 10, 20, 50])
    ap.add_argument("--request-rates", nargs="+", type=float, default=[3.0, 7.0, 10.0])
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--modes", nargs="+", default=["C0", "C3", "C7"])
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--num-prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--tmax", type=int, default=90)
    ap.add_argument("--wgkp-threshold", type=int, default=8)
    ap.add_argument("--port-base", type=int, default=8120)
    ap.add_argument("--output-dir", default="results/sota_evaluation/b4/")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"bench_decode_only.py -- Distinct-pattern sweep K={args.num_adapters} rates={args.request_rates}")
    run_decode_sweep(
        model=args.model,
        adapter_dir=args.adapter_dir,
        k_list=args.num_adapters,
        rate_list=args.request_rates,
        rank=args.rank,
        modes=args.modes,
        output_dir=args.output_dir,
        reps=args.reps,
        seeds=args.seeds,
        num_prompts=args.num_prompts,
        warmup=args.warmup,
        tmax_ms=args.tmax,
        wgkp_threshold=args.wgkp_threshold,
        port_base=args.port_base,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
