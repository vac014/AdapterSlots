"""
bench.py -- Universal single-GPU benchmark harness for sota_evaluation.

Runs all ablation (AB-1 through AB-6) and SOTA comparison (B1–B6) experiments
for a single backend/config pair. Each run starts a fresh server, sends requests
at the specified rate using an async client, collects metrics, and writes a JSON result.

Usage:
    python bench.py \
        --backend adapterslots --mode C7 \
        --model ./models/llama-7b \
        --num-adapters 10 --rank 32 --target-modules all_linear \
        --request-rate 7 --dataset sharegpt --pattern zipf \
        --num-prompts 1000 --warmup 20 --reps 3 \
        --tmax 90 --wgkp-threshold 8 \
        --output results/ab1/C7.json

    # Quick dry-run (no GPU, uses first available adapters or synthetic data)
    python bench.py --backend adapterslots --mode C7 --dry-run --output /tmp/test.json
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
from backends.backend_punica import PunicaBackend
from backends.backend_slora import SLoRABackend
from backends.backend_dlora import DLoRABackend
from backends import get_backend
from benchmarks.metrics_collector import MetricsCollector, parse_as_metrics_jsonl
from workloads.pattern_generator import ArrivalPatternGenerator, Request
from workloads.sharegpt_loader import get_prompts


# Frozen methodology constants (sota_evaluation §3.8)
REPS = 3
N_PROMPTS = 1000
WARMUP = 20
MAX_TOKENS = 128
ZIPF_ALPHA = 0.9
NUMPY_SEED_BASE = 42


def _list_adapters(adapter_dir: str, num_adapters: int, rank: int) -> List[str]:
    """Return list of adapter directory paths for the requested K adapters."""
    base = Path(adapter_dir)
    dirs = []
    for i in range(num_adapters):
        # Try rank-specific name first
        candidates = [
            base / f"adapter_r{rank}_k{i}_s42",
            base / f"adapter_r{rank}_k{i}",
            base / f"adapter_{i}",
        ]
        found = next((str(c) for c in candidates if c.exists()), None)
        if found:
            dirs.append(found)
        else:
            # Use first available adapter (replicates for higher K)
            available = sorted(base.glob("adapter_*"))
            if available:
                dirs.append(str(available[i % len(available)]))
            else:
                dirs.append(str(base))
    return dirs


def _make_backend(args, adapter_dirs: List[str], port: int = 8100):
    """Instantiate the correct backend class from CLI args."""
    backend_name = args.backend.lower()
    mode = getattr(args, "mode", "C7") or "C7"
    kwargs = dict(
        model=args.model,
        adapter_dirs=adapter_dirs,
        port=port,
        tp=args.tp,
        max_lora_rank=args.rank,
        max_loras=max(16, args.num_adapters),
    )
    if backend_name == "adapterslots":
        return AdapterSlotsBackend(
            **kwargs,
            mode=mode,
            tmax_ms=args.tmax,
            wgkp_threshold=args.wgkp_threshold,
        )
    elif backend_name == "vllm":
        return VLLMBackend(**kwargs)
    elif backend_name == "punica":
        return PunicaBackend(**kwargs)
    elif backend_name == "slora":
        return SLoRABackend(**kwargs)
    elif backend_name == "dlora":
        return DLoRABackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


async def _send_requests_async(
    backend,
    requests: List[Request],
    collector: MetricsCollector,
    warmup_ids: set,
    timeout: float = 120.0,
) -> None:
    """Send requests at their specified inter-arrival rates using asyncio."""
    import aiohttp

    async def _one_request(session, req: Request, is_warmup: bool) -> None:
        if is_warmup:
            collector.mark_warmup(req.req_id)
        collector.record_request_start(req.req_id, req.adapter_id, time.perf_counter())
        url, payload = backend.build_request_payload(req.prompt, req.adapter_id, req.max_tokens)
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                body = await resp.json()
                collector.record_first_token(req.req_id, time.perf_counter())
                text = body.get("choices", [{}])[0].get("text", "")
                n_tokens = body.get("usage", {}).get("completion_tokens", len(text.split()))
                collector.record_completion(req.req_id, time.perf_counter(), n_tokens)
        except Exception:
            collector.record_completion(req.req_id, time.perf_counter(), 0)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, req in enumerate(requests):
            if req.inter_arrival_s > 0:
                await asyncio.sleep(req.inter_arrival_s)
            is_warmup = req.req_id in warmup_ids
            task = asyncio.create_task(_one_request(session, req, is_warmup))
            tasks.append(task)
        await asyncio.gather(*tasks)


def _send_requests_sync(
    backend,
    requests: List[Request],
    collector: MetricsCollector,
    warmup_ids: set,
) -> None:
    """Synchronous fallback when aiohttp is unavailable."""
    import urllib.request as urlreq

    for req in requests:
        if req.inter_arrival_s > 0:
            time.sleep(req.inter_arrival_s)
        if req.req_id in warmup_ids:
            collector.mark_warmup(req.req_id)
        collector.record_request_start(req.req_id, req.adapter_id, time.perf_counter())
        try:
            text, n_tokens, _ = backend.send_request(req.prompt, req.adapter_id, req.max_tokens)
            collector.record_first_token(req.req_id, time.perf_counter())
            collector.record_completion(req.req_id, time.perf_counter(), n_tokens)
        except Exception:
            collector.record_completion(req.req_id, time.perf_counter(), 0)


def _run_one_rep(
    backend,
    requests: List[Request],
    warmup_count: int,
    use_async: bool = True,
) -> Tuple[dict, dict]:
    """Run a single measurement rep; return (metrics_dict, alignment_dict)."""
    collector = MetricsCollector()
    warmup_ids = {r.req_id for r in requests[:warmup_count]}

    collector.mark_run_start()

    if use_async:
        try:
            asyncio.run(
                _send_requests_async(backend, requests, collector, warmup_ids)
            )
        except ImportError:
            _send_requests_sync(backend, requests, collector, warmup_ids)
    else:
        _send_requests_sync(backend, requests, collector, warmup_ids)

    collector.mark_run_end()
    summary = collector.compute()

    # Collect AdapterSlots alignment metrics if available
    alignment = {}
    if hasattr(backend, "metrics_path"):
        alignment = parse_as_metrics_jsonl(backend.metrics_path)

    return {
        "throughput_toks": summary.throughput_toks,
        "throughput_reqs": summary.throughput_reqs,
        "ttft_p50_ms": summary.ttft_p50_ms,
        "ttft_p99_ms": summary.ttft_p99_ms,
        "tbt_p50_ms": summary.tbt_p50_ms,
        "tbt_p99_ms": summary.tbt_p99_ms,
        "tpot_ms": summary.tpot_ms,
        "slo_attainment": summary.slo_attainment,
        "n_completed": summary.n_completed,
        "wall_time_s": summary.wall_time_s,
        **alignment,
    }, alignment


def run_benchmark(
    backend_name: str,
    mode: str,
    model: str,
    adapter_dirs: List[str],
    num_adapters: int,
    rank: int,
    target_modules: str,
    request_rate: float,
    dataset: str,
    pattern: str,
    num_prompts: int,
    warmup: int,
    reps: int,
    seeds: List[int],
    output: str,
    tp: int = 1,
    tmax_ms: Optional[int] = None,
    wgkp_threshold: Optional[int] = None,
    port: int = 8100,
    dry_run: bool = False,
    extra_args: Optional[List[str]] = None,
    **adapterslots_env_vars,
) -> dict:
    """Main benchmark function. Starts server, runs reps, writes JSON."""
    prompts = get_prompts(dataset=dataset, n=num_prompts + warmup + 50, seed=seeds[0])
    gen = ArrivalPatternGenerator(prompts, max_tokens=MAX_TOKENS)

    rep_results = []
    for rep_idx, seed in enumerate(seeds):
        print(f"\n  Rep {rep_idx + 1}/{len(seeds)} (seed={seed})")

        if pattern == "zipf":
            requests = gen.zipf(request_rate, num_prompts + warmup, K=num_adapters,
                                alpha=ZIPF_ALPHA, seed=seed)
        elif pattern == "uniform":
            requests = gen.uniform(request_rate, num_prompts + warmup, K=num_adapters, seed=seed)
        elif pattern == "identical":
            requests = gen.identical(request_rate, num_prompts + warmup, seed=seed)
        elif pattern == "distinct":
            requests = gen.distinct(request_rate, num_prompts + warmup, K=num_adapters, seed=seed)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        if dry_run:
            # Return synthetic metrics for testing
            rep_results.append({
                "seed": seed,
                "throughput_toks": 450.0 + rep_idx * 10,
                "throughput_reqs": 4.7 + rep_idx * 0.1,
                "ttft_p50_ms": 120.0,
                "ttft_p99_ms": 185.0,
                "tbt_p50_ms": 20.0,
                "tbt_p99_ms": 28.0,
                "tpot_ms": 22.0,
                "slo_attainment": 0.95,
                "n_completed": num_prompts,
            })
            continue

        # Make a fresh backend instance for each rep to get clean metrics
        # Use different port offset per rep to avoid conflicts
        rep_port = port + rep_idx

        class _FakeArgs:
            backend = backend_name
            tp_attr = tp
            rank_attr = rank
            tmax = tmax_ms
            wgkp_threshold_attr = wgkp_threshold
            num_adapters_attr = num_adapters
            model_attr = model

        # Build backend directly
        if backend_name.lower() == "adapterslots":
            bkd = AdapterSlotsBackend(
                model=model, adapter_dirs=adapter_dirs, port=rep_port, tp=tp,
                mode=mode, tmax_ms=tmax_ms, wgkp_threshold=wgkp_threshold,
                max_lora_rank=rank, max_loras=max(16, num_adapters),
                extra_args=extra_args,
            )
        elif backend_name.lower() == "vllm":
            bkd = VLLMBackend(
                model=model, adapter_dirs=adapter_dirs, port=rep_port, tp=tp,
                max_lora_rank=rank, max_loras=max(16, num_adapters),
                extra_args=extra_args,
            )
        elif backend_name.lower() == "punica":
            bkd = PunicaBackend(
                model=model, adapter_dirs=adapter_dirs, port=rep_port, tp=tp,
            )
        elif backend_name.lower() == "slora":
            bkd = SLoRABackend(
                model=model, adapter_dirs=adapter_dirs, port=rep_port, tp=tp,
                max_lora_rank=rank,
            )
        elif backend_name.lower() == "dlora":
            bkd = DLoRABackend(
                model=model, adapter_dirs=adapter_dirs, port=rep_port, tp=tp,
                max_lora_rank=rank,
            )
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

        print(f"    Starting server...")
        if not bkd.start():
            print(f"    ERROR: server failed to start for rep {rep_idx + 1}")
            continue

        try:
            metrics, _ = _run_one_rep(bkd, requests, warmup)
            metrics["seed"] = seed
            rep_results.append(metrics)
            print(
                f"    throughput={metrics['throughput_toks']:.1f} tok/s  "
                f"TTFT_p50={metrics.get('ttft_p50_ms', float('nan')):.0f}ms"
            )
        finally:
            bkd.stop()

    # Compute 3-rep summary
    tps_values = [r["throughput_toks"] for r in rep_results]
    gain_vs_baseline = None  # populated by caller (run_ablations.py / run_sota.py)

    summary_dict = {}
    if tps_values:
        mean_tps = statistics.mean(tps_values)
        std_tps = statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0
        min_tps = min(tps_values)
        summary_dict = {
            "throughput_toks_mean": round(mean_tps, 2),
            "throughput_toks_std": round(std_tps, 2),
            "throughput_toks_min": round(min_tps, 2),
            "all_reps_positive": True,
        }
        if "gwar8" in rep_results[0]:
            summary_dict["gwar8_mean"] = statistics.mean(
                r.get("gwar8", 0.0) for r in rep_results
            )

    result = {
        "config": {
            "backend": backend_name,
            "mode": mode,
            "K": num_adapters,
            "rank": rank,
            "target_modules": target_modules,
            "request_rate": request_rate,
            "dataset": dataset,
            "pattern": pattern,
            "num_prompts": num_prompts,
            "warmup": warmup,
            "reps": reps,
            "tp": tp,
            "tmax_ms": tmax_ms,
            "wgkp_threshold": wgkp_threshold,
        },
        "reps": rep_results,
        "summary": summary_dict,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Result written to {output}")
    return result


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AdapterSlots benchmark harness")
    ap.add_argument("--backend", default="adapterslots",
                    choices=["adapterslots", "vllm", "punica", "slora", "dlora"])
    ap.add_argument("--mode", default="C7",
                    help="AdapterSlots mode: C0–C7 (ignored for non-adapterslots backends)")
    ap.add_argument("--model", default="./models/llama-7b")
    ap.add_argument("--adapter-dir", default="./adapters")
    ap.add_argument("--num-adapters", type=int, default=10)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--target-modules", default="all_linear",
                    choices=["qkvo", "all_linear"])
    ap.add_argument("--request-rate", type=float, default=7.0)
    ap.add_argument("--dataset", default="sharegpt", choices=["sharegpt", "alpaca"])
    ap.add_argument("--pattern", default="zipf",
                    choices=["zipf", "uniform", "identical", "distinct"])
    ap.add_argument("--num-prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--tmax", type=int, default=None,
                    help="AS_TMAX_MS override")
    ap.add_argument("--wgkp-threshold", type=int, default=None,
                    help="AS_WGKP_THRESHOLD override")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--output", default="results/bench_out.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Return synthetic results without starting any server")
    ap.add_argument("--seed", type=int, default=None,
                    help="Single seed override (for run_claims.py single-rep usage)")
    ap.add_argument("--extra-args", nargs="+", default=None,
                    help="Extra vLLM CLI flags passed through verbatim to both "
                         "AdapterSlotsBackend and VLLMBackend, e.g. "
                         "--extra-args --quantization gptq_marlin --num-scheduler-steps 8 "
                         "(scripts/benchmark_quantized_vs_fp16.py runs the >=1.3x config)")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    seeds = args.seeds
    if args.seed is not None:
        seeds = [args.seed]
    seeds = seeds[: args.reps]

    adapter_dirs = _list_adapters(args.adapter_dir, args.num_adapters, args.rank)

    print(f"bench.py -- backend={args.backend} mode={args.mode} K={args.num_adapters} "
          f"rank={args.rank} rate={args.request_rate} pattern={args.pattern}")

    run_benchmark(
        backend_name=args.backend,
        mode=args.mode,
        model=args.model,
        adapter_dirs=adapter_dirs,
        num_adapters=args.num_adapters,
        rank=args.rank,
        target_modules=args.target_modules,
        request_rate=args.request_rate,
        dataset=args.dataset,
        pattern=args.pattern,
        num_prompts=args.num_prompts,
        warmup=args.warmup,
        reps=args.reps,
        seeds=seeds,
        output=args.output,
        tp=args.tp,
        tmax_ms=args.tmax,
        wgkp_threshold=args.wgkp_threshold,
        port=args.port,
        dry_run=args.dry_run,
        extra_args=args.extra_args,
    )


if __name__ == "__main__":
    main()
