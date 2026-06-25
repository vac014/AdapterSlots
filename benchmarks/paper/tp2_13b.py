"""LLaMA-13B on two RTX A6000 with TP=2, K = 4, 8, 16 adapters: vLLM against
AdapterSlots C7 on the identical trace. Saturated arrivals (rate 300), 400
measured prompts, warmup 30, --max-num-seqs 256 --num-scheduler-steps 8.

Settings (environment): RANKS (default 32), KS (default 4,8,16), ARMS
(default vllm,as), SEED (default 42), SEGMODE, OUT, PORT0, GPU (default 0,1).
AS_* variables reach the AdapterSlots server.
"""
import os

from common import as_env, free_port, load, out_dir, record, rep_row, setup

setup("0,1")
from benchmarks.ablations.bench import run_benchmark, _list_adapters  # noqa: E402

TP = int(os.environ.get("TP", "2"))
RANKS = [int(r) for r in os.environ.get("RANKS", "32").split(",")]
ADIR = {16: "./adapters_13b_r16", 32: "./adapters_13b", 64: "./adapters_13b_r64",
        128: "./adapters_13b_r128"}
EXTRA = ["--max-num-seqs", "256", "--num-scheduler-steps", "8"]
SEED = int(os.environ.get("SEED", "42"))
OUT = out_dir("table5_13b_tp2")
port = int(os.environ.get("PORT0", "27100"))

for rank in RANKS:
    for K in [int(k) for k in os.environ.get("KS", "4,8,16").split(",")]:
        for arm in os.environ.get("ARMS", "vllm,as").split(","):
            tag = f"r{rank}_sat_K{K}_{arm}" + ("" if SEED == 42 else f"_s{SEED}")
            if tag in load(OUT):
                continue
            port = free_port(port + 1)
            envs = as_env() if arm == "as" else {}
            os.environ.update(envs)
            try:
                kw = dict(backend_name="vllm" if arm == "vllm" else "adapterslots",
                          mode="C0" if arm == "vllm" else "C7", model="./models/llama-13b",
                          adapter_dirs=_list_adapters(ADIR[rank], K, rank), num_adapters=K,
                          rank=rank, target_modules="all_linear", dataset="sharegpt",
                          pattern="zipf", reps=1, seeds=[SEED], tp=TP, output=f"{OUT}/{tag}.json",
                          port=port, extra_args=EXTRA, request_rate=300, num_prompts=400,
                          warmup=30)
                if arm == "as":
                    kw.update(tmax_ms=90, wgkp_threshold=8)
                run_benchmark(**kw)
                row = dict(rep_row(kw["output"]), arm=arm, K=K, rank=rank, tp=TP)
                record(OUT, tag, row)
                print(f"{tag} tput={row['tput']:.1f}", flush=True)
            except Exception as e:
                print(f"{tag} FAILED: {type(e).__name__}: {e}", flush=True)
            finally:
                for k in envs:
                    os.environ.pop(k, None)
