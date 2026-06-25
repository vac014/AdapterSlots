"""APIS on two RTX A6000 over PCIe (claim C2): LLaMA-7B, K=50 adapters,
Zipf-0.9 routing, saturated arrivals (rate 300), --max-num-seqs 256
--num-scheduler-steps 8 for every server.

Arms, each alone on both GPUs, on the identical request trace per seed:
  apis       two TP=1 AdapterSlots servers; adapters assigned to them by rate-sorted
             round-robin (adapter_slots/integrations/apis_router.py)
  tp2        one vLLM server with TP=2 holding all 50 adapters
  tp2_v050   the same on vLLM 0.5.0
  topo       two TP=1 vLLM servers with the APIS adapter assignment
  topo_rr    two TP=1 vLLM servers, round-robin, every adapter on both
  topo_rr25  as topo_rr with the APIS shard's LoRA budget (25 GPU slots per
             server, all 50 adapters cached in host memory)

Settings (environment): RANK (default 32), ARMS, SEEDS, NP (default 1000), WU
(default 40), SEGMODE, OUT, PORT0. AS_* variables reach the APIS servers.
"""
import os
import statistics
import time

from common import as_env, free_port, load, out_dir, setup, strip_scheduler_steps, vllm050_backend

setup("0,1")
import json  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, "benchmarks/ablations")
import bench_apis as BA  # noqa: E402
from bench import _list_adapters, _run_one_rep  # noqa: E402
from backends.backend_vllm import VLLMBackend  # noqa: E402
from workloads.sharegpt_loader import get_prompts  # noqa: E402

K = 50
RANK = int(os.environ.get("RANK", "32"))
ADIR = {16: "./adapters_r16d", 32: "./adapters", 64: "./adapters_r64", 128: "./adapters_r128"}[RANK]
NP, WU = int(os.environ.get("NP", "1000")), int(os.environ.get("WU", "40"))
ARMS = os.environ.get("ARMS", "apis,tp2,topo,topo_rr25").split(",")
EXTRA = ["--max-num-seqs", "256", "--num-scheduler-steps", "8"]
OUT = out_dir(f"apis_r{RANK}")
port = int(os.environ.get("PORT0", "23100"))
res = load(OUT)
adapter_dirs = _list_adapters(ADIR, K, RANK)
prompts = get_prompts(dataset="sharegpt", n=NP + WU + 50, seed=42)
gen = BA.ArrivalPatternGenerator(prompts, max_tokens=BA.MAX_TOKENS)
V050 = vllm050_backend()


def two_server(reqs, extra, vanilla, route="affinity", slots=None):
    global port
    port = free_port(port + 2)
    while free_port(port + 1) != port + 1:
        port = free_port(port + 2)
    runner = BA.APISBenchmarkRunner(model="./models/llama-7b", adapter_dirs=adapter_dirs,
                                    port_a=port, port_b=port + 1, rank=RANK,
                                    tmax_ms=90, wgkp_threshold=8, route=route)
    if slots:
        runner.max_loras = slots
        extra = list(extra) + ["--max-cpu-loras", str(K)]
    if vanilla:
        runner.server_a, runner.server_b = (
            VLLMBackend(model="./models/llama-7b", adapter_dirs=adapter_dirs, port=p, tp=1,
                        max_lora_rank=RANK, max_loras=runner.max_loras, gpu_id=g)
            for p, g in ((port, 0), (port + 1, 1)))
    for srv in (runner.server_a, runner.server_b):
        srv.extra_args = list(extra)
    try:
        if not runner.start():
            raise RuntimeError("two-server start failed")
        _, _, comb = runner.run(reqs, WU)
        return comb["throughput_toks"]
    finally:
        runner.stop()
        time.sleep(10)


def tp2(reqs, extra, cls=VLLMBackend):
    global port
    port = free_port(port + 1)
    b = cls(model="./models/llama-7b", adapter_dirs=adapter_dirs, port=port,
            tp=2, max_lora_rank=RANK, max_loras=max(16, K), extra_args=extra)
    try:
        b.start()
        m, _ = _run_one_rep(b, reqs, WU)
        return m["throughput_toks"]
    finally:
        b.stop()
        time.sleep(10)


for seed in [int(x) for x in os.environ.get("SEEDS", "42,43,44").split(",")]:
    key = f"sat_s{seed}"
    row = res.setdefault(key, {})
    reqs = gen.zipf(300.0, NP + WU, K=K, alpha=BA.ZIPF_ALPHA, seed=seed)
    for arm in ARMS:
        if arm in row:
            continue
        envs = as_env() if arm == "apis" else {}
        os.environ.update(envs)
        try:
            if arm == "tp2":
                row[arm] = tp2(reqs, EXTRA)
            elif arm == "tp2_v050":
                row[arm] = tp2(reqs, strip_scheduler_steps(EXTRA), cls=V050)
            elif arm == "topo_rr":
                row[arm] = two_server(reqs, EXTRA, vanilla=True, route="round_robin")
            elif arm == "topo_rr25":
                row[arm] = two_server(reqs, EXTRA, vanilla=True, route="round_robin",
                                      slots=-(-K // 2))
            else:
                row[arm] = two_server(reqs, EXTRA, vanilla=(arm == "topo"))
            print(f"{key} {arm}={row[arm]:.1f}", flush=True)
        except Exception as e:
            print(f"{key} {arm} FAILED: {type(e).__name__}: {e}", flush=True)
        finally:
            for k in envs:
                os.environ.pop(k, None)
        json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)

for a, b in (("apis", "tp2"), ("apis", "topo_rr25"), ("topo_rr25", "tp2"), ("apis", "topo"),
             ("apis", "tp2_v050")):
    rs = [r[a] / r[b] for r in res.values() if r.get(a) and r.get(b)]
    if rs:
        print(f"{a}/{b} = {statistics.mean(rs):.4f} over {len(rs)} seeds", flush=True)
