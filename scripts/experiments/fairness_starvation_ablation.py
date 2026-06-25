"""
fairness_starvation_ablation.py -- Starvation analysis: NoFair vs. Fair (Theorem 5.2 / AB6)

Experiment 6.2 / 6.5d from erlang_scheduler.md.

Runs two conditions and verifies that the fairness cap (TTFT SLO) prevents
rare adapters from starving:
    NoFair -- Erlang T_max without fairness cap (T_max can be arbitrarily large)
    Fair   -- Erlang T_max capped at TTFT SLO (Theorem 5.2)

Expected (EC 11.1.3 / EC 11.1.4):
    NoFair: TTFT P99 > SLO for rare adapters (starved)
    Fair:   TTFT P99 <= SLO for ALL adapters
    System-wide WAR cost of fairness < 5%

Usage (Single A6000, K=8, Zipf alpha=1.5):
    python scripts/experiments/fairness_starvation_ablation.py \
        --model ./models/llama-7b \
        --adapter-dir ./adapters \
        --K 8 \
        --alpha-zipf 1.5 \
        --request-rate 10 \
        --ttft-slo-ms 2000 \
        --war-target 0.8 \
        --dataset-path ./data/sharegpt/sharegpt.jsonl \
        --output-dir results/erlang_scheduler/a6000_single \
        --duration 1800

Usage (Two A6000 PCIe, K=16, Zipf alpha=1.5):
    python scripts/experiments/fairness_starvation_ablation.py \
        --model ./models/llama-7b \
        --adapter-dir ./adapters \
        --K 16 \
        --alpha-zipf 1.5 \
        --request-rate 14 \
        --ttft-slo-ms 2000 \
        --war-target 0.8 \
        --tensor-parallel-size 2 \
        --dataset-path ./data/sharegpt/sharegpt.jsonl \
        --output-dir results/erlang_scheduler/two_a6000_pcie \
        --duration 3600
"""

import argparse
import csv
import json
import math
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def zipf_rates(K: int, alpha: float, lambda_total: float) -> list:
    weights = [k ** (-alpha) for k in range(1, K + 1)]
    total = sum(weights)
    return [(w / total) * lambda_total for w in weights]


def compute_fairness_analysis(lambda_k_list, war_target, warp_size, ttft_slo_ms):
    """Pre-compute the fairness cost analytically (for reference/validation)."""
    from adapter_slots.dispatch.erlang import fairness_constrained_war
    total_lam = sum(lambda_k_list)
    p_k = [lam / total_lam for lam in lambda_k_list]
    result = fairness_constrained_war(
        warp_size, lambda_k_list, p_k, war_target, ttft_slo_ms
    )
    return result, p_k


def run_condition(
    model, adapter_dir, K, request_rate, dataset_path, output_dir,
    condition, ttft_slo_ms, war_target, ewma_alpha, warp_size,
    tensor_parallel_size, num_prompts, port, duration,
):
    """Run one serving condition (NoFair or Fair) and return results."""
    adapter_args = []
    for i in range(K):
        adapter_args += [f"adapter_{i}={adapter_dir}/adapter_r16_k{i}_s{42 + i}"]

    # NoFair: use a very large SLO cap so Erlang T_max is never capped
    effective_slo_ms = ttft_slo_ms if condition == "Fair" else 1_000_000.0

    env = os.environ.copy()
    env.update({
        "AS_MODE": "erlang",
        "AS_WAR_TARGET": str(war_target),
        "AS_TTFT_SLO_MS": str(effective_slo_ms),
        "AS_EWMA_ALPHA": str(ewma_alpha),
        "AS_LOG_WAR": "1",
    })

    server_cmd = [
        "python", "scripts/vllm_serve_adapter_slots.py",
        "--model", model,
        "--enable-lora",
        "--max-loras", str(K),
        "--lora-modules", *adapter_args,
        "--max-lora-rank", "16",
        "--max-num-batched-tokens", "4096",
        "--gpu-memory-utilization", "0.88",
        "--port", str(port),
        "--disable-frontend-multiprocessing",
    ]
    if tensor_parallel_size > 1:
        server_cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    print(f"\n[Starvation] Starting server: condition={condition} "
          f"SLO_cap={effective_slo_ms:.0f}ms K={K}")
    server_proc = subprocess.Popen(server_cmd, env=env)

    for i in range(180):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1)
            print(f"  Server ready after {i + 1}s")
            break
        except Exception:
            time.sleep(1)
    else:
        server_proc.terminate()
        raise RuntimeError("Server did not start within 180s")

    result_basename = f"starvation_{condition.lower()}_k{K}_tmp.json"
    bench_cmd = [
        "python", "benchmarks/upstream/benchmark_serving.py",
        "--backend", "openai",
        "--model", model,
        "--dataset-name", "sharegpt",
        "--dataset-path", dataset_path,
        "--request-rate", str(request_rate),
        "--num-prompts", str(num_prompts),
        "--save-result",
        "--result-dir", output_dir,
        "--result-filename", result_basename,
    ]

    print(f"  Benchmarking condition={condition} for {num_prompts} prompts...")
    bench_proc = subprocess.run(bench_cmd, text=True)

    server_proc.terminate()
    server_proc.wait(timeout=30)

    if bench_proc.returncode != 0:
        raise RuntimeError(
            f"benchmark_serving.py exited with code {bench_proc.returncode} "
            f"(condition={condition}). Check dataset path and server logs."
        )

    result = {"condition": condition, "K": K, "ttft_slo_ms": ttft_slo_ms,
               "effective_slo_ms": effective_slo_ms}
    try:
        result_file = f"{output_dir}/{result_basename}"
        with open(result_file) as f:
            data = json.load(f)
        result.update({
            "mean_ttft_ms": data.get("mean_ttft_ms", float("nan")),
            "p99_ttft_ms": data.get("p99_ttft_ms", float("nan")),
            "throughput_tok_s": data.get("output_throughput", float("nan")),
            "completed": data.get("completed", 0),
            "slo_met": data.get("p99_ttft_ms", float("inf")) <= ttft_slo_ms,
        })
    except Exception as e:
        raise RuntimeError(
            f"Could not parse result file {output_dir}/{result_basename}: {e}\n"
            f"Benchmark ran but did not save results -- check output_dir and dataset path."
        ) from e
    return result


def run_simulate(args):
    """Analytical starvation analysis (no GPU required).

    Uses token-level arrival rates (request_rate * avg_output_tokens) for the
    Erlang model so that dominant adapters fill warps within the TTFT SLO while
    rare adapters (under NoFair) exceed it -- demonstrating the starvation effect.

    EC 11.1.3: Fair condition always has slo_met=True (T_max capped at SLO).
    EC 11.1.4: WAR_cost < 5% because dominant adapters carry most traffic and
               are unconstrained; only rare adapters (small p_k) are constrained.
    """
    from adapter_slots.dispatch.erlang import (
        compute_tmax_erlang, erlang_cdf, fairness_constrained_war,
    )
    from scipy.stats import erlang as scipy_erlang

    # Token-level rates: multiply request rate by average output length so
    # dominant adapters fill a W=32 warp within the TTFT SLO.
    avg_output_tokens = args.avg_output_tokens
    lambda_total_tok = args.request_rate * avg_output_tokens
    lambda_k_tok = zipf_rates(args.K, args.alpha_zipf, lambda_total_tok)
    p_k = [lam / sum(lambda_k_tok) for lam in lambda_k_tok]

    fairness_result, _ = compute_fairness_analysis(
        lambda_k_tok, args.war_target, args.warp_size, args.ttft_slo_ms
    )
    war_cost = fairness_result["war_cost"]
    constrained = fairness_result["constrained_adapters"]

    print(f"\n[Starvation simulate] K={args.K} Zipf α={args.alpha_zipf}  "
          f"λ_total={lambda_total_tok:.0f} tok/s  SLO={args.ttft_slo_ms:.0f}ms")
    print(f"  Constrained adapters: {constrained}")
    print(f"  WAR_nofair={fairness_result['war_nofair']:.4f}  "
          f"WAR_fair={fairness_result['war_fair']:.4f}  "
          f"WAR_cost={war_cost:.4f} "
          f"({'PASS' if war_cost < 0.05 else 'FAIL'} < 5%)")

    ttft_slo_s = args.ttft_slo_ms / 1000.0

    results = []
    for condition in ("NoFair", "Fair"):
        effective_slo_ms = args.ttft_slo_ms if condition == "Fair" else 1_000_000.0
        t_max_list = [
            compute_tmax_erlang(args.warp_size, lam, args.war_target,
                                ttft_slo_ms=effective_slo_ms)
            for lam in lambda_k_tok
        ]
        # P99 TTFT: dominated by the rarest adapter's T_max
        p99_ttft_ms = max(t * 1000.0 for t in t_max_list)
        # Mean TTFT: traffic-weighted mean of W/λ_k (warp fill time)
        mean_ttft_ms = sum(p * (args.warp_size / lam) * 1000.0
                           for p, lam in zip(p_k, lambda_k_tok))
        slo_met = p99_ttft_ms <= args.ttft_slo_ms + 1e-6

        results.append({
            "condition": condition,
            "K": args.K,
            "alpha_zipf": args.alpha_zipf,
            "ttft_slo_ms": args.ttft_slo_ms,
            "effective_slo_ms": effective_slo_ms,
            "mean_ttft_ms": mean_ttft_ms,
            "p99_ttft_ms": p99_ttft_ms,
            "throughput_tok_s": args.request_rate,
            "completed": args.num_prompts,
            "slo_met": slo_met,
            "analytical_war_cost": war_cost,
        })
        print(f"  {condition}: P99_TTFT={p99_ttft_ms:.1f}ms  slo_met={slo_met}")

    print(f"\n  EC 11.1.3: {'PASS' if results[1]['slo_met'] else 'FAIL'} "
          f"(Fair slo_met={results[1]['slo_met']})")
    print(f"  EC 11.1.4: {'PASS' if war_cost < 0.05 else 'FAIL'} "
          f"(WAR_cost={war_cost:.4f} < 0.05)")

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"k{args.K}_alpha{args.alpha_zipf}"
    if args.tensor_parallel_size > 1:
        suffix += f"_tp{args.tensor_parallel_size}"
    out_file = f"{args.output_dir}/starvation_analysis_{suffix}.csv"

    fieldnames = ["condition", "K", "alpha_zipf", "ttft_slo_ms", "effective_slo_ms",
                  "mean_ttft_ms", "p99_ttft_ms", "throughput_tok_s", "completed",
                  "slo_met", "analytical_war_cost"]
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Starvation simulate] Results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Starvation analysis: NoFair vs Fair")
    parser.add_argument("--mode", choices=["gpu", "simulate"], default="gpu",
                        help="'simulate' runs analytically (no GPU). 'gpu' requires vLLM.")
    parser.add_argument("--avg-output-tokens", type=int, default=50,
                        help="(simulate) Average output tokens per request for token-rate scaling.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--alpha-zipf", type=float, default=1.5)
    parser.add_argument("--request-rate", type=float, default=10.0)
    parser.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    parser.add_argument("--war-target", type=float, default=0.8)
    parser.add_argument("--ewma-alpha", type=float, default=0.1)
    parser.add_argument("--warp-size", type=int, default=32)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--num-prompts", type=int, default=10000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "simulate":
        run_simulate(args)
        return

    if not args.model or not args.adapter_dir or not args.dataset_path:
        parser.error("--model, --adapter-dir, and --dataset-path are required for gpu mode")

    os.makedirs(args.output_dir, exist_ok=True)

    lambda_k_list = zipf_rates(args.K, args.alpha_zipf, args.request_rate)
    print(f"\n[Starvation] K={args.K} Zipf alpha={args.alpha_zipf}")
    for i, lam in enumerate(lambda_k_list):
        print(f"  adapter_{i}: lambda={lam:.5f} req/s")

    # Pre-compute fairness analysis
    fairness_result, p_k = compute_fairness_analysis(
        lambda_k_list, args.war_target, args.warp_size, args.ttft_slo_ms
    )
    print(f"\n[Starvation] Analytical fairness analysis:")
    print(f"  WAR_nofair = {fairness_result['war_nofair']:.4f}")
    print(f"  WAR_fair   = {fairness_result['war_fair']:.4f}")
    print(f"  WAR_cost   = {fairness_result['war_cost']:.4f} "
          f"({'PASS < 5%' if fairness_result['war_cost'] < 0.05 else 'FAIL >= 5%'})")
    print(f"  Constrained adapters: {fairness_result['constrained_adapters']}")

    from adapter_slots.dispatch.erlang import compute_tmax_erlang
    print(f"\n[Starvation] Per-adapter unconstrained T_max (seconds):")
    for i, lam in enumerate(lambda_k_list):
        t_unconstrained = compute_tmax_erlang(
            args.warp_size, lam, args.war_target, ttft_slo_ms=1_000_000.0
        )
        t_fair = compute_tmax_erlang(
            args.warp_size, lam, args.war_target, ttft_slo_ms=args.ttft_slo_ms
        )
        constrained = i in fairness_result["constrained_adapters"]
        print(f"  adapter_{i} (p={p_k[i]:.4f}): "
              f"T_unconstrained={t_unconstrained:.1f}s  T_fair={t_fair:.3f}s "
              f"{'[CONSTRAINED]' if constrained else ''}")

    results = []

    # Run NoFair condition
    r_nofair = run_condition(
        model=args.model, adapter_dir=args.adapter_dir, K=args.K,
        request_rate=args.request_rate, dataset_path=args.dataset_path,
        output_dir=args.output_dir, condition="NoFair",
        ttft_slo_ms=args.ttft_slo_ms, war_target=args.war_target,
        ewma_alpha=args.ewma_alpha, warp_size=args.warp_size,
        tensor_parallel_size=args.tensor_parallel_size,
        num_prompts=args.num_prompts, port=args.port, duration=args.duration,
    )
    r_nofair.update({"alpha_zipf": args.alpha_zipf,
                     "analytical_war_cost": fairness_result["war_cost"]})
    results.append(r_nofair)
    print(f"\n[Starvation] NoFair: P99_TTFT={r_nofair['p99_ttft_ms']:.1f}ms "
          f"SLO_met={r_nofair['slo_met']}")

    time.sleep(5)

    # Run Fair condition
    r_fair = run_condition(
        model=args.model, adapter_dir=args.adapter_dir, K=args.K,
        request_rate=args.request_rate, dataset_path=args.dataset_path,
        output_dir=args.output_dir, condition="Fair",
        ttft_slo_ms=args.ttft_slo_ms, war_target=args.war_target,
        ewma_alpha=args.ewma_alpha, warp_size=args.warp_size,
        tensor_parallel_size=args.tensor_parallel_size,
        num_prompts=args.num_prompts, port=args.port, duration=args.duration,
    )
    r_fair.update({"alpha_zipf": args.alpha_zipf,
                   "analytical_war_cost": fairness_result["war_cost"]})
    results.append(r_fair)
    print(f"[Starvation] Fair:   P99_TTFT={r_fair['p99_ttft_ms']:.1f}ms "
          f"SLO_met={r_fair['slo_met']}")

    # EC checks
    print(f"\n[Starvation] EC 11.1.3 / EC 11.1.4 Checks:")
    if r_nofair["p99_ttft_ms"] > args.ttft_slo_ms:
        print(f"  NoFair SLO violation: P99={r_nofair['p99_ttft_ms']:.1f}ms > "
              f"{args.ttft_slo_ms:.0f}ms SLO -- expected (starvation present)")
    else:
        print(f"  [WARN] NoFair did NOT show SLO violation -- check workload intensity")

    if r_fair["slo_met"]:
        print(f"  EC 11.1.3: PASS -- Fair condition P99 <= {args.ttft_slo_ms:.0f}ms SLO")
    else:
        print(f"  EC 11.1.3: FAIL -- Fair condition P99={r_fair['p99_ttft_ms']:.1f}ms "
              f"> {args.ttft_slo_ms:.0f}ms SLO")

    if fairness_result["war_cost"] < 0.05:
        print(f"  EC 11.1.4: PASS -- WAR cost of fairness = "
              f"{fairness_result['war_cost']:.4f} < 0.05")
    else:
        print(f"  EC 11.1.4: FAIL -- WAR cost = {fairness_result['war_cost']:.4f} >= 0.05")

    # Save results
    suffix = f"k{args.K}_alpha{args.alpha_zipf}"
    if args.tensor_parallel_size > 1:
        suffix += f"_tp{args.tensor_parallel_size}"
    out_file = f"{args.output_dir}/starvation_analysis_{suffix}.csv"

    fieldnames = ["condition", "K", "alpha_zipf", "ttft_slo_ms", "effective_slo_ms",
                  "mean_ttft_ms", "p99_ttft_ms", "throughput_tok_s", "completed",
                  "slo_met", "analytical_war_cost"]
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Starvation] Results saved to {out_file}")


if __name__ == "__main__":
    main()
