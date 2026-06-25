"""
bench_apis.py -- APIS two-GPU benchmark harness (sota_evaluation §3.3).

Starts two independent TP=1 vLLM+AdapterSlots servers (one per A6000, GPU 0 and GPU 1).
Dispatches requests round-robin across both servers. Measures:
  - Per-server throughput (tok/s)
  - Combined system_tps = GPU0_tps + GPU1_tps
  - APIS / TP=2-baseline ratio  →  gate EC 14.AB6 (≥ 1.8×)

Hardware assumption: two RTX A6000 48 GB on the same PCIe bus.
DO NOT run on NVLink or H100 hardware -- τ_iter constants are A6000-specific.

Usage:
    # Full APIS run (C7 vs TP=2 baseline at K=50, λ=7)
    CUDA_VISIBLE_DEVICES=0,1 python bench_apis.py \\
        --model ./models/llama-7b \\
        --adapter-dir ./adapters \\
        --num-adapters 50 --rank 32 \\
        --request-rate 7 --num-prompts 1000 --warmup 20 --reps 3 \\
        --output results/sota_evaluation/ab6/apis_k50.json

    # Dry-run (no GPU)
    python bench_apis.py --dry-run --output /tmp/apis_test.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


from backends.backend_adapterslots import AdapterSlotsBackend
from backends.backend_vllm import VLLMBackend
from benchmarks.metrics_collector import MetricsCollector, parse_as_metrics_jsonl
from workloads.pattern_generator import ArrivalPatternGenerator, Request
from workloads.sharegpt_loader import get_prompts

# Frozen methodology constants
REPS = 3
N_PROMPTS = 1000
WARMUP = 20
MAX_TOKENS = 128
ZIPF_ALPHA = 0.9

# A6000 PCIe hardware constants (DO NOT change for H100/NVLink)
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
    Manages two independent TP=1 servers (GPU 0 and GPU 1) and dispatches
    requests round-robin between them to measure APIS combined throughput.
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
    ) -> None:
        self.model = model
        self.adapter_dirs = adapter_dirs
        self.port_a = port_a
        self.port_b = port_b
        self.rank = rank
        self.tmax_ms = tmax_ms
        self.wgkp_threshold = wgkp_threshold
        self.mode = mode

        num_adapters = len(adapter_dirs)

        self.server_a = AdapterSlotsBackend(
            model=model,
            adapter_dirs=adapter_dirs,
            port=port_a,
            tp=1,
            mode=mode,
            tmax_ms=tmax_ms,
            wgkp_threshold=wgkp_threshold,
            max_lora_rank=rank,
            max_loras=max(16, num_adapters),
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
            max_loras=max(16, num_adapters),
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
        use_async: bool = True,
    ) -> Tuple[dict, dict, dict]:
        """
        Dispatch requests round-robin across server A and B.
        Returns (metrics_a, metrics_b, combined_metrics).
        """
        reqs_a = [r for i, r in enumerate(requests) if i % 2 == 0]
        reqs_b = [r for i, r in enumerate(requests) if i % 2 == 1]

        wids_a = {r.req_id for r in reqs_a[: max(1, warmup_count // 2)]}
        wids_b = {r.req_id for r in reqs_b[: max(1, warmup_count // 2)]}

        collector_a = MetricsCollector()
        collector_b = MetricsCollector()

        collector_a.mark_run_start()
        collector_b.mark_run_start()

        if use_async:
            try:
                asyncio.run(self._send_both_async(
                    reqs_a, reqs_b, collector_a, collector_b, wids_a, wids_b
                ))
            except ImportError:
                self._send_both_sync(reqs_a, reqs_b, collector_a, collector_b, wids_a, wids_b)
        else:
            self._send_both_sync(reqs_a, reqs_b, collector_a, collector_b, wids_a, wids_b)

        collector_a.mark_run_end()
        collector_b.mark_run_end()

        s_a = collector_a.compute()
        s_b = collector_b.compute()

        align_a = parse_as_metrics_jsonl(self.server_a.metrics_path) if hasattr(self.server_a, "metrics_path") else {}
        align_b = parse_as_metrics_jsonl(self.server_b.metrics_path) if hasattr(self.server_b, "metrics_path") else {}

        metrics_a = {
            "throughput_toks": s_a.throughput_toks,
            "ttft_p50_ms": s_a.ttft_p50_ms,
            "ttft_p99_ms": s_a.ttft_p99_ms,
            "slo_attainment": s_a.slo_attainment,
            "n_completed": s_a.n_completed,
            **align_a,
        }
        metrics_b = {
            "throughput_toks": s_b.throughput_toks,
            "ttft_p50_ms": s_b.ttft_p50_ms,
            "ttft_p99_ms": s_b.ttft_p99_ms,
            "slo_attainment": s_b.slo_attainment,
            "n_completed": s_b.n_completed,
            **align_b,
        }

        # Combined: throughput is additive; latency is the worse (max) of the two
        combined = {
            "throughput_toks": metrics_a["throughput_toks"] + metrics_b["throughput_toks"],
            "ttft_p50_ms": max(s_a.ttft_p50_ms, s_b.ttft_p50_ms),
            "ttft_p99_ms": max(s_a.ttft_p99_ms, s_b.ttft_p99_ms),
            "slo_attainment": min(s_a.slo_attainment, s_b.slo_attainment),
            "n_completed": s_a.n_completed + s_b.n_completed,
        }

        return metrics_a, metrics_b, combined

    async def _send_both_async(
        self,
        reqs_a: List[Request],
        reqs_b: List[Request],
        col_a: MetricsCollector,
        col_b: MetricsCollector,
        wids_a: set,
        wids_b: set,
    ) -> None:
        import aiohttp

        async def _one(session, backend, req: Request, collector: MetricsCollector, wids: set) -> None:
            if req.req_id in wids:
                collector.mark_warmup(req.req_id)
            collector.record_request_start(req.req_id, req.adapter_id, time.perf_counter())
            url, payload = backend.build_request_payload(req.prompt, req.adapter_id, req.max_tokens)
            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=120.0)
                ) as resp:
                    body = await resp.json()
                    collector.record_first_token(req.req_id, time.perf_counter())
                    text = body.get("choices", [{}])[0].get("text", "")
                    n_toks = body.get("usage", {}).get("completion_tokens", len(text.split()))
                    collector.record_completion(req.req_id, time.perf_counter(), n_toks)
            except Exception:
                collector.record_completion(req.req_id, time.perf_counter(), 0)

        async with aiohttp.ClientSession() as session:
            tasks = []
            # Interleave requests from both streams preserving arrival order
            queue: List[Tuple[float, Request, str]] = []
            t = 0.0
            for req in reqs_a:
                queue.append((t, req, "a"))
                t += req.inter_arrival_s
            t = 0.0
            for req in reqs_b:
                queue.append((t, req, "b"))
                t += req.inter_arrival_s
            queue.sort(key=lambda x: x[0])

            last_t = 0.0
            for sched_t, req, side in queue:
                if sched_t > last_t:
                    await asyncio.sleep(sched_t - last_t)
                    last_t = sched_t
                backend = self.server_a if side == "a" else self.server_b
                collector = col_a if side == "a" else col_b
                wids = wids_a if side == "a" else wids_b
                tasks.append(asyncio.create_task(_one(session, backend, req, collector, wids)))
            await asyncio.gather(*tasks)

    def _send_both_sync(
        self,
        reqs_a: List[Request],
        reqs_b: List[Request],
        col_a: MetricsCollector,
        col_b: MetricsCollector,
        wids_a: set,
        wids_b: set,
    ) -> None:
        for req in reqs_a:
            if req.inter_arrival_s > 0:
                time.sleep(req.inter_arrival_s)
            if req.req_id in wids_a:
                col_a.mark_warmup(req.req_id)
            col_a.record_request_start(req.req_id, req.adapter_id, time.perf_counter())
            try:
                _, n_toks, _ = self.server_a.send_request(req.prompt, req.adapter_id, req.max_tokens)
                col_a.record_first_token(req.req_id, time.perf_counter())
                col_a.record_completion(req.req_id, time.perf_counter(), n_toks)
            except Exception:
                col_a.record_completion(req.req_id, time.perf_counter(), 0)

        for req in reqs_b:
            if req.inter_arrival_s > 0:
                time.sleep(req.inter_arrival_s)
            if req.req_id in wids_b:
                col_b.mark_warmup(req.req_id)
            col_b.record_request_start(req.req_id, req.adapter_id, time.perf_counter())
            try:
                _, n_toks, _ = self.server_b.send_request(req.prompt, req.adapter_id, req.max_tokens)
                col_b.record_first_token(req.req_id, time.perf_counter())
                col_b.record_completion(req.req_id, time.perf_counter(), n_toks)
            except Exception:
                col_b.record_completion(req.req_id, time.perf_counter(), 0)


def _dry_run_result(num_adapters: int, request_rate: float, reps: int, seeds: List[int]) -> dict:
    """Synthetic result for --dry-run mode."""
    rep_results = []
    for i, seed in enumerate(seeds[:reps]):
        apis_tps = 820.0 + i * 12
        rep_results.append({
            "seed": seed,
            "gpu0_throughput_toks": apis_tps / 2,
            "gpu1_throughput_toks": apis_tps / 2,
            "apis_throughput_toks": apis_tps,
            "baseline_tp2_toks": 440.0,
            "ratio": apis_tps / 440.0,
        })
    tps_vals = [r["apis_throughput_toks"] for r in rep_results]
    return {
        "config": {"K": num_adapters, "rate": request_rate, "mode": "C7", "dry_run": True},
        "reps": rep_results,
        "summary": {
            "apis_throughput_mean": statistics.mean(tps_vals),
            "apis_throughput_std": statistics.stdev(tps_vals) if len(tps_vals) > 1 else 0.0,
            "ratio_mean": statistics.mean(r["ratio"] for r in rep_results),
            "ec_ab6_pass": statistics.mean(r["ratio"] for r in rep_results) >= 1.8,
        },
    }


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
    dry_run: bool = False,
) -> dict:
    """Run APIS two-GPU benchmark and write JSON result."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        result = _dry_run_result(num_adapters, request_rate, reps, seeds)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  [dry-run] APIS result written to {output}")
        return result

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
                f"baseline={baseline_tps:.1f} tok/s  ratio={ratio:.3f}×"
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

    print(f"\n  EC 14.AB6: APIS/TP2 ratio mean={mean_ratio:.3f}  "
          f"[threshold ≥ 1.80] → {'PASS' if mean_ratio >= 1.80 else 'FAIL'}")

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
            "ec_ab6_pass": mean_ratio >= 1.80,
        },
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Result written to {output}")
    return result


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="sota_evaluation APIS two-GPU benchmark")
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
    ap.add_argument("--output", default="results/sota_evaluation/ab6/apis.json")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"bench_apis.py -- APIS K={args.num_adapters} rank={args.rank} rate={args.request_rate}")
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
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
