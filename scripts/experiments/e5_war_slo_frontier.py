"""
e5_war_slo_frontier.py -- WAR-SLO feasibility frontier sweep (Experiment 6.3 / 6.6d)

Sweeps WAR* in {0.3, 0.5, 0.7, 0.8, 0.9, 1.0} and for each:
    1. Computes per-adapter T_max^(k)* via Erlang CDF inversion
    2. Runs a serving session with AS_MODE=erlang AS_WAR_TARGET=WAR*
    3. Records (WAR_achieved, TTFT_P99) as the operating point

Generates the WAR-SLO feasibility frontier for Proposition 5.6 validation.

Usage (Single A6000, K=4):
    python scripts/experiments/e5_war_slo_frontier.py \
        --model ./models/llama-7b \
        --adapter-dir ./adapters \
        --K 4 \
        --alpha-zipf 0.9 \
        --request-rate 7 \
        --dataset-path ./data/sharegpt/sharegpt.jsonl \
        --output results/impl_5/a6000_single/war_slo_frontier.csv \
        --duration 300

Usage (Two H100 NVLink, K=4, for cross-hardware comparison):
    python scripts/experiments/e5_war_slo_frontier.py \
        --model ./models/llama-7b \
        --adapter-dir ./adapters \
        --K 4 \
        --alpha-zipf 0.9 \
        --request-rate 7 \
        --tensor-parallel-size 2 \
        --dataset-path ./data/sharegpt/sharegpt.jsonl \
        --output results/impl_5/two_h100_nvlink/war_slo_frontier_nvlink.csv \
        --label "Two H100 NVLink (TP=2)" \
        --duration 300
"""

import argparse
import csv
import json
import os
import subprocess
import time
import urllib.request


def run_war_target(
    model, adapter_dir, K, request_rate, dataset_path, output_dir,
    war_target, ttft_slo_ms, ewma_alpha, tensor_parallel_size, num_prompts, port,
):
    """Run one serving session at a given WAR* and return (war_achieved, ttft_p99)."""
    adapter_args = []
    for i in range(K):
        adapter_args += [f"adapter_{i}={adapter_dir}/adapter_r16_k{i}_s{42 + i}"]

    env = os.environ.copy()
    env.update({
        "AS_MODE": "erlang",
        "AS_WAR_TARGET": str(war_target),
        "AS_TTFT_SLO_MS": str(ttft_slo_ms),
        "AS_EWMA_ALPHA": str(ewma_alpha),
        "AS_LOG_WAR": "1",
    })
    if tensor_parallel_size > 1:
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    server_cmd = [
        "python", "scripts/vllm_serve_adapterslots.py",
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

    print(f"\n[WAR-SLO] WAR*={war_target:.2f}: starting server...")
    server_proc = subprocess.Popen(server_cmd, env=env)

    for i in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1)
            print(f"  Server ready after {i + 1}s")
            break
        except Exception:
            time.sleep(1)
    else:
        server_proc.terminate()
        return {"war_target": war_target, "error": "server_timeout"}

    result_basename = f"war_slo_war{war_target:.2f}_tmp.json"
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

    bench_proc = subprocess.run(bench_cmd, text=True)
    server_proc.terminate()
    server_proc.wait(timeout=30)

    if bench_proc.returncode != 0:
        raise RuntimeError(
            f"benchmark_serving.py exited with code {bench_proc.returncode} "
            f"(war_target={war_target}). Check dataset path and server logs."
        )

    result = {"war_target": war_target}
    try:
        with open(f"{output_dir}/{result_basename}") as f:
            data = json.load(f)
        # WAR achieved is logged via AS_LOG_WAR -- parse from stderr if available
        # Use mean_ttft and p99_ttft as primary metrics
        result.update({
            "mean_ttft_ms": data.get("mean_ttft_ms", float("nan")),
            "p99_ttft_ms": data.get("p99_ttft_ms", float("nan")),
            "throughput_tok_s": data.get("output_throughput", float("nan")),
            "completed": data.get("completed", 0),
            # WAR achieved stored in custom field if benchmark_serving was patched
            "war_achieved": data.get("war_achieved", float("nan")),
        })
    except Exception as e:
        raise RuntimeError(
            f"Could not parse result for war_target={war_target}: {e}\n"
            f"Expected file: {output_dir}/{result_basename}"
        ) from e

    print(f"  WAR*={war_target:.2f}: TTFT_P99={result['p99_ttft_ms']:.1f}ms")
    return result


def run_simulate(args):
    """Compute WAR-SLO frontier analytically via Erlang CDF inversion.

    Uses token-level arrival rates (request_rate × avg_output_tokens) because the
    Erlang model governs token accumulation in the AlignmentBuffer warp, not request
    arrivals. This matches the parameterisation used in e5_starvation.py --mode simulate.

    For each WAR* target:
        war_achieved = WAR*          (exact by Erlang ppf inversion)
        p99_ttft_ms  = T_max*(rarest adapter) in ms, uncapped -- the unconstrained
                       operating point. Values above ttft_slo_ms are infeasible
                       without the fairness cap; the knee is the last WAR* below SLO.
        mean_ttft_ms = traffic-weighted mean T_max*(k).
    """
    from adapterslots.dispatch.erlang import compute_tmax_erlang

    K = args.K
    alpha = args.alpha_zipf
    rate = args.request_rate
    avg_out = args.avg_output_tokens
    slo = args.ttft_slo_ms
    W = 32

    weights = [k ** (-alpha) for k in range(1, K + 1)]
    total_w = sum(weights)
    p_k = [w / total_w for w in weights]
    # Token-level rates: request rate × avg output tokens per adapter
    lambda_k_tok = [p * rate * avg_out for p in p_k]

    print(f"\n[WAR-SLO simulate] K={K} α={alpha} λ_req={rate} avg_out={avg_out} "
          f"λ_tok_total={rate*avg_out:.0f} SLO={slo:.0f}ms")
    print(f"  {'WAR*':>5}  {'T_max_dom (ms)':>14}  {'T_max_rare (ms)':>16}  "
          f"{'mean_ms':>9}  {'p99_ms':>9}  {'feasible':>9}")
    print("  " + "-" * 72)

    results = []
    for war_target in args.war_targets:
        tmax_k = [
            compute_tmax_erlang(W, lam, war_target, ttft_slo_ms=1e9)
            for lam in lambda_k_tok
        ]
        # p99 = T_max of rarest adapter (uncapped -- shows the true operating point)
        p99_ms = max(t * 1000.0 for t in tmax_k)
        mean_ms = sum(p * t * 1000.0 for p, t in zip(p_k, tmax_k))
        feasible = p99_ms <= slo

        print(f"  {war_target:>5.2f}  {tmax_k[0]*1000:>14.0f}  {tmax_k[-1]*1000:>16.0f}  "
              f"{mean_ms:>9.0f}  {p99_ms:>9.0f}  {'YES' if feasible else 'NO (>SLO)':>9}")

        results.append({
            "war_target": war_target,
            "war_achieved": war_target,
            "mean_ttft_ms": round(mean_ms, 1),
            "p99_ttft_ms": round(p99_ms, 1),
            "throughput_tok_s": float("nan"),
            "completed": float("nan"),
            "K": K,
            "alpha_zipf": alpha,
            "label": args.label,
        })

    # Pareto knee: last WAR* where p99_ttft (uncapped T_max*) < SLO
    feasible_pts = [r for r in results if r["p99_ttft_ms"] < slo]
    if feasible_pts:
        knee = max(feasible_pts, key=lambda r: r["war_target"])
        print(f"\n  Pareto knee: WAR*={knee['war_target']:.2f}  p99={knee['p99_ttft_ms']:.0f}ms  "
              f"(last operating point within {slo:.0f}ms SLO)")
        if 0.70 <= knee["war_target"] <= 0.90:
            print("  PASS -- consistent with Proposition 5.6 (WAR* = 0.75–0.85)")
        else:
            print(f"  NOTE -- knee at {knee['war_target']:.2f}, expected 0.70–0.90; "
                  "check request-rate / avg-output-tokens")
    else:
        print(f"\n  All WAR* targets exceed SLO -- lower request-rate or avg-output-tokens")

    return results


def main():
    parser = argparse.ArgumentParser(description="WAR-SLO feasibility frontier sweep")
    parser.add_argument("--mode", choices=["gpu", "simulate"], default="gpu",
                        help="'simulate' computes Erlang frontier analytically (no GPU). "
                             "'gpu' requires vLLM + dataset.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--alpha-zipf", type=float, default=0.9)
    parser.add_argument("--request-rate", type=float, default=7.0)
    parser.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    parser.add_argument("--ewma-alpha", type=float, default=0.1)
    parser.add_argument("--war-targets", nargs="+", type=float,
                        default=[0.3, 0.5, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--avg-output-tokens", type=int, default=64,
                        help="(simulate) Average output tokens per request for token-rate scaling.")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output", required=True,
                        help="Output CSV path (e.g. results/impl_5/a6000_single/war_slo_frontier.csv)")
    parser.add_argument("--label", default="")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--num-prompts", type=int, default=5000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "simulate":
        results = run_simulate(args)
    else:
        if not args.model or not args.adapter_dir or not args.dataset_path:
            parser.error("--model, --adapter-dir, and --dataset-path are required for gpu mode")
        output_dir = os.path.dirname(args.output) or "."
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[WAR-SLO Frontier] K={args.K} alpha={args.alpha_zipf} "
              f"rate={args.request_rate} label={args.label}")
        print(f"  Sweeping WAR* = {args.war_targets}")

        results = []
        for war_target in args.war_targets:
            result = run_war_target(
                model=args.model,
                adapter_dir=args.adapter_dir,
                K=args.K,
                request_rate=args.request_rate,
                dataset_path=args.dataset_path,
                output_dir=output_dir,
                war_target=war_target,
                ttft_slo_ms=args.ttft_slo_ms,
                ewma_alpha=args.ewma_alpha,
                tensor_parallel_size=args.tensor_parallel_size,
                num_prompts=args.num_prompts,
                port=args.port,
            )
            result["label"] = args.label
            result["K"] = args.K
            result["alpha_zipf"] = args.alpha_zipf
            results.append(result)
            time.sleep(5)

    # Write frontier CSV
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    fieldnames = ["war_target", "war_achieved", "mean_ttft_ms", "p99_ttft_ms",
                  "throughput_tok_s", "completed", "K", "alpha_zipf", "label"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[WAR-SLO Frontier] Results saved to {args.output}")

    # Print frontier summary table
    print(f"\n{'WAR*':>8} {'P99_TTFT(ms)':>14} {'Throughput(tok/s)':>18}")
    print("-" * 44)
    for r in results:
        if "error" not in r:
            print(f"  {r['war_target']:>6.2f}  "
                  f"{r['p99_ttft_ms']:>14.1f}  "
                  f"{r['throughput_tok_s']:>18.1f}")

    # Identify Pareto knee (approximately WAR* = 0.75-0.85 per Proposition 5.6)
    valid = [r for r in results if "error" not in r and r["p99_ttft_ms"] == r["p99_ttft_ms"]]
    if valid:
        # Find the elbow: largest WAR* with P99 TTFT < 2× minimum P99
        min_p99 = min(r["p99_ttft_ms"] for r in valid)
        pareto_candidates = [r for r in valid if r["p99_ttft_ms"] < 2 * min_p99]
        if pareto_candidates:
            knee = max(pareto_candidates, key=lambda r: r["war_target"])
            print(f"\n[WAR-SLO Frontier] Pareto knee approx at WAR*={knee['war_target']:.2f} "
                  f"P99={knee['p99_ttft_ms']:.1f}ms")
            if 0.75 <= knee["war_target"] <= 0.85:
                print("  Consistent with Proposition 5.6 prediction (WAR* = 0.75-0.85)")
            else:
                print(f"  [NOTE] Knee at {knee['war_target']:.2f}, expected 0.75-0.85 "
                      "(check workload parameters)")


if __name__ == "__main__":
    main()
