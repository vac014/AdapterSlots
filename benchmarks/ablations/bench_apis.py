"""
bench_apis.py -- APIS two-GPU benchmark harness.

Starts two independent TP=1 vLLM+AdapterSlots servers (one per A6000, GPU 0 and GPU 1).
With route="affinity" (the default) adapters are assigned to the servers by
rate-sorted round-robin (adapter_slots/integrations/apis_router.py) and each
request goes to the server holding its adapter; route="round_robin" alternates
requests. Measures:
  - Per-server throughput (tok/s)
  - Combined system_tps = GPU0_tps + GPU1_tps
  - APIS / TP=2-baseline ratio

Hardware: two RTX A6000 48 GB on the same PCIe bus; the τ_iter constants
below are for that setup.

Usage:
    # Full APIS run (C7 vs TP=2 baseline at K=50, λ=7)
    CUDA_VISIBLE_DEVICES=0,1 python bench_apis.py \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --num-adapters 50 --rank 32 \\
        --request-rate 7 --num-prompts 1000 --warmup 20 --reps 3 \\
        --output results/apis_k50.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


from adapter_slots.integrations.apis_router import APISRouter
from backends.backend_adapterslots import AdapterSlotsBackend
from backends.backend_vllm import VLLMBackend
from benchmarks.metrics_collector import MetricsCollector, parse_as_metrics_jsonl
from benchmarks.streaming import stream_completion
from workloads.pattern_generator import ArrivalPatternGenerator, Request
from workloads.sharegpt_loader import get_prompts

# Methodology constants
REPS = 3
N_PROMPTS = 1000
WARMUP = 20
MAX_TOKENS = 128
ZIPF_ALPHA = 0.9

# τ_iter on RTX A6000 (PCIe)
TAU_ITER_TP1_MS = 30.0   # single A6000, TP=1
TAU_ITER_TP2_MS = 100.0  # two A6000 PCIe, TP=2


def _list_adapters(adapter_dir: str, num_adapters: int, rank: int) -> List[str]:
    base = Path(adapter_dir)
    dirs = []
    for i in range(num_adapters):
        candidates = [
            base / f"adapter_r{rank}_k{i}_s42",
            base / f"adapter_r{rank}_k{i}",
            base / f"adapter_{i}",
        ]
        found = next((str(c) for c in candidates if c.exists()), None)
        if found:
            dirs.append(found)
        else:
            available = sorted(base.glob("adapter_*"))
            if available:
                dirs.append(str(available[i % len(available)]))
            else:
                dirs.append(str(base))
    return dirs


class APISBenchmarkRunner:
    """
    Two independent TP=1 servers (GPU 0 and GPU 1) and the routing of the
    trace between them (adapter affinity or round-robin).
    """

    def __init__(
        self,
        model: str,
        adapter_dirs: List[str],
        port_a: int = 8110,
        port_b: int = 8111,
        rank: int = 32,
        tmax_ms: Optional[int] = 90,
        wgkp_threshold: Optional[int] = 8,
        mode: str = "C7",
        route: str = "affinity",
    ) -> None:
        self.model = model
        # "affinity": each adapter lives on one shard (APISRouter);
        # "round_robin": requests alternate.
        self.route = route
        self.adapter_dirs = adapter_dirs
        self.port_a = port_a
        self.port_b = port_b
        self.rank = rank
        self.tmax_ms = tmax_ms
        self.wgkp_threshold = wgkp_threshold
        self.mode = mode

        num_adapters = len(adapter_dirs)
        # Under adapter affinity a shard only ever serves its half of the pool.
        shard_loras = -(-num_adapters // 2) if route == "affinity" else num_adapters
        self.max_loras = max(16, shard_loras)

        self.server_a = AdapterSlotsBackend(
            model=model,
            adapter_dirs=adapter_dirs,
            port=port_a,
            tp=1,
            mode=mode,
            tmax_ms=tmax_ms,
            wgkp_threshold=wgkp_threshold,
            max_lora_rank=rank,
            max_loras=self.max_loras,
            gpu_id=0,
        )
        self.server_b = AdapterSlotsBackend(
            model=model,
            adapter_dirs=adapter_dirs,
            port=port_b,
            tp=1,
            mode=mode,
            tmax_ms=tmax_ms,
            wgkp_threshold=wgkp_threshold,
            max_lora_rank=rank,
            max_loras=self.max_loras,
            gpu_id=1,
        )

    def start(self) -> bool:
        ok_a = self.server_a.start()
        ok_b = self.server_b.start()
        return ok_a and ok_b

    def stop(self) -> None:
        self.server_a.stop()
        self.server_b.stop()

    def run(
        self,
        requests: List[Request],
        warmup_count: int,
    ) -> Tuple[dict, dict, dict]:
        """Send the trace to both servers; return (metrics_a, metrics_b, combined).

        With route="affinity" each request goes to the shard that holds its
        adapter: APISRouter assigns adapters by rate-sorted round-robin, starting
        from the trace's request rates (converged estimates) and refreshing the
        assignment from its EWMA rates every 30 s. With route="round_robin"
        requests alternate between the servers.
        """
        send_at, t = [], 0.0
        for r in requests:
            t += r.inter_arrival_s
            send_at.append(t)
        if self.route == "affinity":
            counts: Dict[int, int] = {}
            for r in requests:
                counts[r.adapter_id] = counts.get(r.adapter_id, 0) + 1
            span = max(send_at[-1], 1e-9)
            router = APISRouter(n_shards=2, rebalance_s=30.0,
                                initial_rates={a: c / span for a, c in counts.items()})
            shards = [router.route(r.adapter_id, ts) for r, ts in zip(requests, send_at)]
        else:
            shards = [i % 2 for i in range(len(requests))]

        warm = {r.req_id for r in requests[:warmup_count]}
        collectors = (MetricsCollector(), MetricsCollector())
        for c in collectors:
            c.mark_run_start()
        asyncio.run(self._send_async(requests, send_at, shards, collectors, warm))
        for c in collectors:
            c.mark_run_end()
        s_a, s_b = (c.compute() for c in collectors)

        def side(summary, server) -> dict:
            return {
                "throughput_toks": summary.throughput_toks,
                "ttft_p50_ms": summary.ttft_p50_ms,
                "ttft_p99_ms": summary.ttft_p99_ms,
                "slo_attainment": summary.slo_attainment,
                "n_completed": summary.n_completed,
                **(parse_as_metrics_jsonl(server.metrics_path)
                   if hasattr(server, "metrics_path") else {}),
            }

        metrics_a, metrics_b = side(s_a, self.server_a), side(s_b, self.server_b)
        measured = shards[warmup_count:]
        # Throughput adds up across the shards; latency is the worse of the two.
        combined = {
            "throughput_toks": metrics_a["throughput_toks"] + metrics_b["throughput_toks"],
            "ttft_p50_ms": max(s_a.ttft_p50_ms, s_b.ttft_p50_ms),
            "ttft_p99_ms": max(s_a.ttft_p99_ms, s_b.ttft_p99_ms),
            "slo_attainment": min(s_a.slo_attainment, s_b.slo_attainment),
            "n_completed": s_a.n_completed + s_b.n_completed,
            "shard0_request_share": measured.count(0) / max(len(measured), 1),
        }
        return metrics_a, metrics_b, combined

    async def _send_async(self, requests, send_at, shards, collectors, warm) -> None:
        """Send every request at its trace time to its shard, streaming responses."""
        import aiohttp

        servers = (self.server_a, self.server_b)

        async def _one(session, req: Request, shard: int) -> None:
            collector, backend = collectors[shard], servers[shard]
            if req.req_id in warm:
                collector.mark_warmup(req.req_id)
            collector.record_request_start(req.req_id, req.adapter_id, time.perf_counter())
            url, payload = backend.build_request_payload(req.prompt, req.adapter_id, req.max_tokens)
            try:
                n_toks = await stream_completion(session, url, payload, 120.0,
                                                 collector, req.req_id)
                collector.record_completion(req.req_id, time.perf_counter(), n_toks)
            except Exception:
                collector.record_completion(req.req_id, time.perf_counter(), 0)

        conn = aiohttp.TCPConnector(limit=int(os.environ.get("BENCH_CONN_LIMIT", "1024")))
        async with aiohttp.ClientSession(connector=conn) as session:
            tasks, last_t = [], 0.0
            for req, sched_t, shard in zip(requests, send_at, shards):
                if sched_t > last_t:
                    await asyncio.sleep(sched_t - last_t)
                    last_t = sched_t
                tasks.append(asyncio.create_task(_one(session, req, shard)))
            await asyncio.gather(*tasks)


def run_apis_benchmark(
    model: str,
    adapter_dir: str,
    num_adapters: int,
    rank: int,
    request_rate: float,
    num_prompts: int,
    warmup: int,
    reps: int,
    seeds: List[int],
    output: str,
    port_a: int = 8110,
    port_b: int = 8111,
    tmax_ms: int = 90,
    wgkp_threshold: int = 8,
    tp_baseline: int = 2,
    dataset: str = "sharegpt",
) -> dict:
    """Run APIS two-GPU benchmark and write JSON result."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    adapter_dirs = _list_adapters(adapter_dir, num_adapters, rank)
    prompts = get_prompts(dataset=dataset, n=num_prompts + warmup + 50, seed=seeds[0])
    gen = ArrivalPatternGenerator(prompts, max_tokens=MAX_TOKENS)

    rep_results = []

    for rep_idx, seed in enumerate(seeds[:reps]):
        print(f"\n  APIS Rep {rep_idx + 1}/{reps} (seed={seed})")
        requests = gen.zipf(request_rate, num_prompts + warmup, K=num_adapters,
                            alpha=ZIPF_ALPHA, seed=seed)

        runner = APISBenchmarkRunner(
            model=model,
            adapter_dirs=adapter_dirs,
            port_a=port_a + rep_idx * 10,
            port_b=port_b + rep_idx * 10,
            rank=rank,
            tmax_ms=tmax_ms,
            wgkp_threshold=wgkp_threshold,
        )

        # Also start TP=2 baseline (C0) for in-session comparison
        baseline = VLLMBackend(
            model=model,
            adapter_dirs=adapter_dirs,
            port=port_a + 5 + rep_idx * 10,
            tp=tp_baseline,
            max_lora_rank=rank,
            max_loras=max(16, num_adapters),
        )

        print("    Starting APIS servers (GPU 0 + GPU 1)...")
        apis_ok = runner.start()
        print("    Starting TP=2 baseline server...")
        baseline_ok = baseline.start()

        try:
            if not apis_ok:
                print("    ERROR: APIS servers failed to start")
                continue

            m_a, m_b, combined = runner.run(requests, warmup)

            baseline_tps = 0.0
            if baseline_ok:
                # Run baseline measurement with same requests
                from bench import _run_one_rep
                b_metrics, _ = _run_one_rep(baseline, requests, warmup)
                baseline_tps = b_metrics["throughput_toks"]

            ratio = combined["throughput_toks"] / max(baseline_tps, 1.0)
            rep_results.append({
                "seed": seed,
                "gpu0_throughput_toks": round(m_a["throughput_toks"], 2),
                "gpu1_throughput_toks": round(m_b["throughput_toks"], 2),
                "apis_throughput_toks": round(combined["throughput_toks"], 2),
                "apis_ttft_p99_ms": round(combined["ttft_p99_ms"], 1),
                "apis_slo_attainment": round(combined["slo_attainment"], 4),
                "baseline_tp2_toks": round(baseline_tps, 2),
                "ratio": round(ratio, 4),
            })
            print(
                f"    APIS={combined['throughput_toks']:.1f} tok/s  "
                f"baseline={baseline_tps:.1f} tok/s  ratio={ratio:.3f}x"
            )
        finally:
            runner.stop()
            baseline.stop()

    if not rep_results:
        result = {"config": {}, "reps": [], "summary": {"error": "no reps completed"}}
        out_path.write_text(json.dumps(result, indent=2))
        return result

    apis_vals = [r["apis_throughput_toks"] for r in rep_results]
    ratio_vals = [r["ratio"] for r in rep_results]
    mean_ratio = statistics.mean(ratio_vals)

    print(f"\n  APIS/TP2 ratio mean={mean_ratio:.3f}")

    result = {
        "config": {
            "K": num_adapters,
            "rank": rank,
            "request_rate": request_rate,
            "mode": "C7",
            "tmax_ms": tmax_ms,
            "wgkp_threshold": wgkp_threshold,
            "tp_baseline": tp_baseline,
            "hardware": "two_a6000_pcie",
            "tau_iter_tp1_ms": TAU_ITER_TP1_MS,
            "tau_iter_tp2_ms": TAU_ITER_TP2_MS,
        },
        "reps": rep_results,
        "summary": {
            "apis_throughput_mean": round(statistics.mean(apis_vals), 2),
            "apis_throughput_std": round(
                statistics.stdev(apis_vals) if len(apis_vals) > 1 else 0.0, 2
            ),
            "ratio_mean": round(mean_ratio, 4),
            "ratio_min": round(min(ratio_vals), 4),
        },
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Result written to {output}")
    return result


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="APIS two-GPU benchmark")
    ap.add_argument("--model", default="./models/llama-7b")
    ap.add_argument("--adapter-dir", default="./adapters")
    ap.add_argument("--num-adapters", type=int, default=50)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--request-rate", type=float, default=7.0)
    ap.add_argument("--num-prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--port-a", type=int, default=8110)
    ap.add_argument("--port-b", type=int, default=8111)
    ap.add_argument("--tmax", type=int, default=90)
    ap.add_argument("--wgkp-threshold", type=int, default=8)
    ap.add_argument("--tp-baseline", type=int, default=2,
                    help="TP degree for the vLLM baseline (default 2 for TP=2 A6000 PCIe)")
    ap.add_argument("--dataset", default="sharegpt")
    ap.add_argument("--output", default="results/apis.json")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"bench_apis.py: APIS K={args.num_adapters} rank={args.rank} rate={args.request_rate}")
    run_apis_benchmark(
        model=args.model,
        adapter_dir=args.adapter_dir,
        num_adapters=args.num_adapters,
        rank=args.rank,
        request_rate=args.request_rate,
        num_prompts=args.num_prompts,
        warmup=args.warmup,
        reps=args.reps,
        seeds=args.seeds,
        output=args.output,
        port_a=args.port_a,
        port_b=args.port_b,
        tmax_ms=args.tmax,
        wgkp_threshold=args.wgkp_threshold,
        tp_baseline=args.tp_baseline,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
