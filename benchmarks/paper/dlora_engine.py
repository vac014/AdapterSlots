"""dLoRA on the K=10 workload of C1 and Table II, through dLoRA's own API
server (deps/dlora: exec type LORA, dummy weights as in its artifact scripts).

The client sends the request set the other K=10 runs measure: 30 warmup
requests, then the same 400 ShareGPT prompts and Zipf-0.9 adapter ids at rate
300, 128 output tokens each (ignore_eos). The server runs in the dLoRA
environment ($DLORA_PYTHON); this client runs in the AdapterSlots environment.

Settings (environment): SEED (default 42), DLORA_PYTHON, DLORA_SRC (default
deps/dlora), PORT (default 8300), GPU, OUT. envs/setup_dlora.sh builds the
dLoRA environment.
"""
import asyncio
import os
import signal
import subprocess
import sys
import time

from common import REPO, out_dir, record, setup

setup()
sys.path.insert(0, "benchmarks/ablations")
import aiohttp  # noqa: E402
import bench_apis as BA  # noqa: E402
from workloads.sharegpt_loader import get_prompts  # noqa: E402

K, NP, WU, MT = 10, 400, 30, 128
SEED = int(os.environ.get("SEED", "42"))
PORT = int(os.environ.get("PORT", "8300"))
PY = os.environ.get("DLORA_PYTHON", str(REPO / "venvs" / "dlora" / "bin" / "python"))
SRC = os.environ.get("DLORA_SRC", str(REPO / "deps" / "dlora"))
OUT = out_dir("c1_k10/dlora")

prompts = get_prompts(dataset="sharegpt", n=NP + WU + 50, seed=SEED)
gen = BA.ArrivalPatternGenerator(prompts, max_tokens=MT)
reqs = gen.zipf(300.0, NP + WU, K=K, alpha=BA.ZIPF_ALPHA, seed=SEED)

model = str(REPO / "models" / "llama-7b")
cmd = [PY, "-m", "vllm.entrypoints.api_server", "--host", "127.0.0.1", "--port", str(PORT),
       "--model", model, "--tokenizer", model, "--use-dummy-weights", "--exec-type", "3",
       "--num-models", str(K), "--num-groups", "1", "--max-r", "32", "--engine-use-ray",
       "--worker-use-ray", "--disable-log-requests", "--max-num-seqs", "256",
       "--swap-space", "4"]
log = open(f"{OUT}/dlora_server_s{SEED}.txt", "w")
srv = subprocess.Popen(cmd, cwd=SRC, env=dict(os.environ, RAY_DEDUP_LOGS="0"), stdout=log,
                       stderr=subprocess.STDOUT, start_new_session=True)


async def one(s, r, delay, t0, done):
    await asyncio.sleep(max(0.0, t0 + delay - time.perf_counter()))
    payload = {"prompt": r.prompt, "model_id": int(r.adapter_id), "max_tokens": MT,
               "temperature": 0.0, "ignore_eos": True}
    async with s.post(f"http://127.0.0.1:{PORT}/generate", json=payload) as resp:
        await resp.json()
    done.append(time.perf_counter())


async def main():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1024),
                                     timeout=aiohttp.ClientTimeout(total=3600)) as s:
        for _ in range(600):
            try:
                async with s.post(f"http://127.0.0.1:{PORT}/generate",
                                  json={"prompt": "hi", "model_id": 0, "max_tokens": 2}) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            raise RuntimeError("dLoRA server did not come up")
        send, t = [], 0.0
        for r in reqs:
            t += r.inter_arrival_s
            send.append(t)
        warm = []
        await asyncio.gather(*[one(s, r, 0.0, time.perf_counter(), warm) for r in reqs[:WU]])
        done, t0, base = [], time.perf_counter(), send[WU]
        await asyncio.gather(*[one(s, r, send[i + WU] - base, t0, done)
                               for i, r in enumerate(reqs[WU:])])
        return len(done), max(done) - t0


try:
    n, wall = asyncio.run(main())
    record(OUT, f"dlora_s{SEED}", {"tput": n * MT / wall, "n": n, "wall": wall})
    print(f"s{SEED} tput={n * MT / wall:.1f} tok/s wall={wall:.1f}s", flush=True)
finally:
    os.killpg(srv.pid, signal.SIGTERM)
    time.sleep(5)
    ray = os.path.join(os.path.dirname(PY), "ray")
    if os.path.exists(ray):
        subprocess.run([ray, "stop", "--force"], capture_output=True)
