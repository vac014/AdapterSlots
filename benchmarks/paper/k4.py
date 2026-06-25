"""K=4 cross-validation (Table III): LLaMA-7B, K=4, rank 16, lambda = 7 req/s,
500 measured prompts, warmup 20, streamed requests.

Variants (VARIANTS, comma list):
  engine  default engine flags; AdapterSlots also writes its per-request engine
          timestamps (AS_REQ_METRICS_PATH), so TTFT is reported both at the
          client and inside the engine
  sync    as engine with --disable-async-output-proc, so the first token is
          returned right after the prefill step

TTFT is client-side (first streamed token) in ttft_p50/ttft_p99 and
engine-side (first token time minus arrival in the engine) in eng_ttft_p50/p99.

Settings (environment): VARIANTS (default sync), ARMS (default vllm,as),
SEEDS, OUT, PORT0, GPU.
"""
import json
import os

import numpy as np

from common import as_env, free_port, load, out_dir, record, rep_row, setup

setup()
from benchmarks.ablations.bench import run_benchmark, _list_adapters  # noqa: E402

VARIANTS = {"engine": (None, {}),
            "sync": (["--disable-async-output-proc"], {})}
OUT = out_dir("table3_k4")
port = int(os.environ.get("PORT0", "25100"))

for var in os.environ.get("VARIANTS", "sync").split(","):
    extra, var_env = VARIANTS[var]
    for seed in [int(x) for x in os.environ.get("SEEDS", "42,43,44").split(",")]:
        for arm in os.environ.get("ARMS", "vllm,as").split(","):
            tag = f"k4_{var}_{arm}_s{seed}"
            if tag in load(OUT):
                continue
            port = free_port(port + 1)
            reqm = os.path.abspath(f"{OUT}/{tag}_req.jsonl")
            envs = {}
            if arm == "as":
                envs = dict(as_env(), **var_env, AS_REQ_METRICS_PATH=reqm)
                if os.path.exists(reqm):
                    os.remove(reqm)
            os.environ.update(envs)
            try:
                kw = dict(backend_name="vllm" if arm == "vllm" else "adapterslots",
                          mode="C0" if arm == "vllm" else "C7", model="./models/llama-7b",
                          adapter_dirs=_list_adapters("./adapters_r16d", 4, 16), num_adapters=4,
                          rank=16, target_modules="all_linear", request_rate=7,
                          dataset="sharegpt", pattern="zipf", num_prompts=500, warmup=20,
                          reps=1, seeds=[seed], tp=1, output=f"{OUT}/{tag}.json", port=port,
                          extra_args=extra)
                if arm == "as":
                    kw.update(tmax_ms=90, wgkp_threshold=8)
                run_benchmark(**kw)
                row = rep_row(kw["output"])
                if arm == "as" and os.path.exists(reqm):
                    rows = sorted((json.loads(line) for line in open(reqm)),
                                  key=lambda x: x["arrival"])[20:]   # the client's warmup cut
                    eng = np.array([(x["first_token"] - x["arrival"]) * 1e3 for x in rows
                                    if x["first_token"]])
                    row.update(eng_ttft_p50=float(np.percentile(eng, 50)),
                               eng_ttft_p99=float(np.percentile(eng, 99)), eng_n=len(eng))
                record(OUT, tag, row)
                print(f"{tag} tput={row['tput']:.1f} ttft p50={row['ttft_p50']:.1f} "
                      f"engine p50={row.get('eng_ttft_p50', float('nan')):.1f}", flush=True)
            except Exception as e:
                print(f"{tag} FAILED: {type(e).__name__}: {e}", flush=True)
            finally:
                for k in envs:
                    os.environ.pop(k, None)
