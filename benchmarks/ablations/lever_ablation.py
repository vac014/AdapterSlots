"""
lever_ablation.py -- isolates the marginal throughput contribution of each
AS serving lever within the full deployment stack: the mp-frontend
engine, the AS_TMAX_MS admission-control fix, the packed-nslice fused
kernel, and vLLM's own --num-scheduler-steps. The deployment's base model
is held fixed across every AS arm (only the listed lever changes between
arms) so each delta isolates exactly one component.

Condition: llama-13b, K=8 adapters, zipf overload_rate=40, alpha=0.9,
seed=42, n=120 prompts, max_tokens=128, TP=1 -- the same condition as the
headline deployment-comparison result (scripts/benchmark_quantized_vs_fp16.py).

Arms:
    C  adapterslots-C6-tmax-default   AS_TMAX_MS unset (5ms default), mp-engine on, no multi-step
    D  adapterslots-C6-tmax90         C + tmax fix only                  -> isolates tmax fix     (D vs C)
    E  adapterslots-C6-tmax90-inproc  D + --disable-frontend-multiprocessing -> isolates mp-engine (D vs E)
    F  adapterslots-C7-tmax90         D + fused kernel (C7)               -> isolates fused kernel  (F vs D)
    G  adapterslots-C6-tmax90-ms8     D + --num-scheduler-steps 8         -> isolates multi-step    (G vs D)
    H  adapterslots-C7-tmax90-ms8     D + fused kernel + multi-step (full stack)
Reference arms (vLLM, no quantization, run once and reused -- see results
file for the measured values):
    A  vllm-stock             vanilla vLLM
    B  vllm-ms8                A + --num-scheduler-steps 8

Results from the last real run on real hardware (2x RTX A6000) are in
results/ablations/lever_ablation_K8.json. Re-running this script requires
./models/llama-13b-gptq (scripts/download_models.py --models llama-13b-gptq)
and adapters_13b/ already present.

Usage:
    python benchmarks/ablations/lever_ablation.py
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import time
from pathlib import Path

import aiohttp

from backends.backend_adapterslots import AdapterSlotsBackend
from backends.backend_vllm import VLLMBackend
from benchmarks.metrics_collector import MetricsCollector
from workloads.pattern_generator import ArrivalPatternGenerator
from workloads.sharegpt_loader import get_prompts

_ROOT = Path(__file__).parent.parent.parent
K, RATE, N, ALPHA, SEED = 8, 40, 120, 0.9, 42
MODEL_GPTQ = str(_ROOT / "models" / "llama-13b-gptq")
OUT_PATH = _ROOT / "results" / "ablations" / "lever_ablation_K8.json"


async def _one_request(session, backend, req, collector, timeout):
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


async def _send_burst(backend, requests, collector, timeout=180.0):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for req in requests:
            if req.inter_arrival_s > 0:
                await asyncio.sleep(req.inter_arrival_s)
            tasks.append(asyncio.create_task(_one_request(session, backend, req, collector, timeout)))
        await asyncio.gather(*tasks)


def _adapter_dirs_for(k):
    same_rank = sorted(glob.glob(str(_ROOT / "adapters_13b" / "adapter_r32_k*")))
    by_k = {}
    for d in same_rank:
        parts = Path(d).name.split("_")
        k_part = next((p for p in parts if p.startswith("k") and p[1:].isdigit()), None)
        if k_part is not None:
            by_k.setdefault(int(k_part[1:]), d)
    return [by_k.get(i, same_rank[i % len(same_rank)]) for i in range(k)]


def _run_arm(backend, tag):
    prompts = get_prompts(dataset="sharegpt", n=N + 50, seed=SEED)
    gen = ArrivalPatternGenerator(prompts, max_tokens=128)
    requests = gen.zipf(RATE, N, K=K, alpha=ALPHA, seed=SEED)
    print(f"=== [{tag}] starting on port {backend.port} ===", flush=True)
    backend.start()
    try:
        collector = MetricsCollector()
        collector.mark_run_start()
        asyncio.run(_send_burst(backend, requests, collector))
        collector.mark_run_end()
        raw = sorted(
            (r.first_token_time - r.submit_time) * 1000.0
            for r in collector._records.values() if r.first_token_time is not None
        )
        n = len(raw)
        summary = collector.compute()
        result = {
            "tag": tag,
            "throughput": summary.throughput_toks,
            "wall_time_s": summary.wall_time_s,
            "n_completed": summary.n_completed,
            "p50_ttft_ms": raw[n // 2] if n else None,
            "p99_ttft_ms": raw[int(0.99 * (n - 1))] if n else None,
        }
        print(f"[{tag}] {result}", flush=True)
        return result
    finally:
        backend.stop()


def _make_arms():
    dirs = _adapter_dirs_for(K)
    common = dict(model=MODEL_GPTQ, adapter_dirs=dirs, tp=1, max_lora_rank=32, max_loras=max(16, K))
    return [
        ("C_adapterslots_tmax_default", AdapterSlotsBackend(port=8512, mode="C6", tmax_ms=None,
                                              extra_args=["--quantization", "gptq_marlin"], **common)),
        ("D_adapterslots_tmax90", AdapterSlotsBackend(port=8513, mode="C6", tmax_ms=90,
                                       extra_args=["--quantization", "gptq_marlin"], **common)),
        ("E_adapterslots_tmax90_inprocess", AdapterSlotsBackend(port=8514, mode="C6", tmax_ms=90,
                                                 extra_args=["--quantization", "gptq_marlin",
                                                             "--disable-frontend-multiprocessing"], **common)),
        ("F_adapterslots_C7_tmax90", AdapterSlotsBackend(port=8515, mode="C7", tmax_ms=90,
                                          extra_args=["--quantization", "gptq_marlin"], **common)),
        ("G_adapterslots_tmax90_ms8", AdapterSlotsBackend(port=8516, mode="C6", tmax_ms=90,
                                           extra_args=["--quantization", "gptq_marlin",
                                                       "--num-scheduler-steps", "8"], **common)),
        ("H_adapterslots_C7_tmax90_ms8", AdapterSlotsBackend(port=8517, mode="C7", tmax_ms=90,
                                              extra_args=["--quantization", "gptq_marlin",
                                                          "--num-scheduler-steps", "8"], **common)),
    ]


def main():
    os.chdir(_ROOT)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    results["A_vllm_stock"] = _run_arm(
        VLLMBackend(model="./models/llama-13b", adapter_dirs=_adapter_dirs_for(K), port=8500,
                    tp=1, max_lora_rank=32, max_loras=max(16, K)),
        "A_vllm_stock")
    results["B_vllm_ms8"] = _run_arm(
        VLLMBackend(model="./models/llama-13b", adapter_dirs=_adapter_dirs_for(K), port=8501,
                    tp=1, max_lora_rank=32, max_loras=max(16, K),
                    extra_args=["--num-scheduler-steps", "8"]),
        "B_vllm_ms8")

    for tag, backend in _make_arms():
        results[tag] = _run_arm(backend, tag)
        OUT_PATH.write_text(json.dumps(results, indent=2))

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
