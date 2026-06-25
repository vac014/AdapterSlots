"""WGKP L3 served end to end: LLaMA-7B with one rank-32 adapter, so every decode
step belongs to one adapter, eager mode (no CUDA graphs) and 75% GPU memory for
vLLM, which leaves room for merged weights. The same 200 ShareGPT prompts (64
greedy tokens each, all sent at once) run with L3 on (AS_MWC_K_HOT=5) and off
(AS_MWC_K_HOT=0); the run records throughput, how many completions match, and a
lower bound on the promoted steps (the server logs the count at 1, 10, 100 and
every 1000 promoted steps).

The adapter is adapters/adapter_r32_k0_s100 with its B matrices replaced by
seeded random values (the generated adapters have B = 0, for which the merged
weight would equal the base weight).

Settings (environment): OUT, GPU, PORT0.
"""
import asyncio
import os
import re
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import free_port, out_dir, record, setup  # noqa: E402

setup()
import aiohttp  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from backends.backend_adapterslots import AdapterSlotsBackend  # noqa: E402
from workloads.sharegpt_loader import get_prompts  # noqa: E402

OUT = out_dir("kernel")
port = int(os.environ.get("PORT0", "29900"))
prompts = get_prompts(dataset="sharegpt", n=200, seed=42)[:200]

tmp = tempfile.mkdtemp(prefix="l3_adapter_")
adapter = os.path.join(tmp, "adapter_r32_k0_s100")
shutil.copytree("./adapters/adapter_r32_k0_s100", adapter)
weights = load_file(f"{adapter}/adapter_model.safetensors")
gen = torch.Generator().manual_seed(0)
for k in sorted(weights):
    if k.endswith("lora_B.weight"):
        weights[k] = (torch.randn(weights[k].shape, generator=gen) * 0.02).to(weights[k].dtype)
save_file(weights, f"{adapter}/adapter_model.safetensors")


async def send_all(p: int):
    url = f"http://localhost:{p}/v1/completions"
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1024)) as s:
        async def one(prompt):
            body = {"model": "adapter_0", "prompt": prompt, "max_tokens": 64,
                    "temperature": 0.0, "ignore_eos": True}
            async with s.post(url, json=body) as r:
                j = await r.json()
                return j["choices"][0]["text"], j["usage"]["completion_tokens"]
        t0 = time.perf_counter()
        outs = await asyncio.gather(*[one(q) for q in prompts])
        return outs, time.perf_counter() - t0


def promoted_steps(log_text: str) -> int:
    counts = [int(m) for m in re.findall(r"L3: (\d+) promoted steps", log_text)]
    return max(counts, default=0)


runs = {}
for k_hot in ("5", "0"):
    os.environ.update(AS_MWC_K_HOT=k_hot, AS_SEGGEMM="3", AS_SEGGEMM_MIN_TOKENS="64")
    port = free_port(port + 1)
    log_path = os.path.join(tmp, f"server_{k_hot}.log")
    backend = AdapterSlotsBackend(
        model="./models/llama-7b", adapter_dirs=[adapter], port=port, mode="C7",
        max_lora_rank=32, max_loras=1,
        extra_args=["--enforce-eager", "--gpu-memory-utilization", "0.75", "--max-num-seqs", "256"])
    with open(log_path, "w") as log:
        backend.log_file = log
        backend.start()
        try:
            asyncio.run(send_all(port))  # first pass: the merge is requested and made
            outs, wall = asyncio.run(send_all(port))
        finally:
            backend.stop()
    tokens = sum(n for _, n in outs)
    runs[k_hot] = {"texts": [t for t, _ in outs], "tput": tokens / wall,
                   "promoted_steps": promoted_steps(open(log_path).read())}
    print(f"AS_MWC_K_HOT={k_hot}: {runs[k_hot]['tput']:.1f} tok/s, "
          f"{runs[k_hot]['promoted_steps']} promoted steps", flush=True)


def common_prefix(a: str, b: str) -> float:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n / max(len(a), len(b), 1)


on, off = runs["5"]["texts"], runs["0"]["texts"]
row = {"tput_l3": runs["5"]["tput"], "tput_no_l3": runs["0"]["tput"],
       "speedup": runs["5"]["tput"] / runs["0"]["tput"],
       "promoted_steps_min": runs["5"]["promoted_steps"],
       "identical": sum(a == b for a, b in zip(on, off)), "n": len(prompts),
       "mean_shared_prefix": sum(common_prefix(a, b) for a, b in zip(on, off)) / len(prompts)}
record(OUT, "l3_check", row)
print(f"L3 {row['speedup']:.3f}x, identical {row['identical']}/{row['n']}, "
      f"shared prefix {row['mean_shared_prefix']:.3f}", flush=True)
shutil.rmtree(tmp, ignore_errors=True)
