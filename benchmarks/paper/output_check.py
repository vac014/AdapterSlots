"""Greedy outputs of AdapterSlots against vLLM on the same prompts.

160 ShareGPT prompts, 48 tokens each, sent concurrently so decode batches are
large enough for the grouped kernels; LLaMA-7B, K=10, rank 32, temperature 0.
Reports how many completions match exactly and the mean fraction of the longer
completion that the two share as a common prefix. fp16 reductions in a
different order can make long greedy completions diverge late; a wrong kernel
shows up as near-zero agreement.

Settings (environment): SEGMODE (default 3), OUT, GPU, PORT0.
"""
import asyncio
import json
import os
import statistics

from common import as_env, free_port, out_dir, record, setup

setup()
import aiohttp  # noqa: E402
from benchmarks.ablations.bench import _list_adapters  # noqa: E402
from backends.backend_adapterslots import AdapterSlotsBackend  # noqa: E402
from backends.backend_vllm import VLLMBackend  # noqa: E402
from workloads.sharegpt_loader import get_prompts  # noqa: E402

K, N = 10, 160
OUT = out_dir("kernel")
ad = _list_adapters("./adapters", K, 32)
prompts = get_prompts(dataset="sharegpt", n=N, seed=7)
extra = ["--max-num-seqs", "256", "--num-scheduler-steps", "8"]
port = free_port(int(os.environ.get("PORT0", "47100")))


async def collect(b):
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1024)) as s:
        async def one(i):
            url, payload = b.build_request_payload(prompts[i], i % K, 48)
            async with s.post(url, json=dict(payload, temperature=0.0)) as r:
                return (await r.json())["choices"][0]["text"]
        return await asyncio.gather(*[one(i) for i in range(N)])


def run(b):
    b.start()
    try:
        return asyncio.run(collect(b))
    finally:
        b.stop()


def prefix_frac(x, y):
    n = 0
    for a, b in zip(x, y):
        if a != b:
            break
        n += 1
    return n / max(1, max(len(x), len(y)))


ref = run(VLLMBackend(model="./models/llama-7b", adapter_dirs=ad, port=port, tp=1,
                      max_lora_rank=32, max_loras=16, extra_args=extra))
os.environ.update(as_env())
got = run(AdapterSlotsBackend(model="./models/llama-7b", adapter_dirs=ad,
                              port=free_port(port + 1), tp=1, max_lora_rank=32, mode="C7",
                              extra_args=extra))
exact = sum(a == b for a, b in zip(ref, got))
row = {"mode": int(os.environ["AS_SEGGEMM"]), "n": N, "exact": exact,
       "mean_prefix_frac": statistics.mean(prefix_frac(a, b) for a, b in zip(ref, got))}
record(OUT, f"output_check_mode{row['mode']}", row)
json.dump({"vllm": ref, "adapterslots": got}, open(f"{OUT}/output_check_texts.json", "w"))
print(f"exact={exact}/{N} mean_prefix_frac={row['mean_prefix_frac']:.3f}", flush=True)
