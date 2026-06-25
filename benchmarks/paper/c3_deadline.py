"""C3: GWAR(8) against throughput over a sweep of the admission alignment deadline.

LLaMA-13B on one RTX A6000, K=16 adapters, 64 output tokens, saturated
arrivals (rate 300), --max-num-seqs 256 --num-scheduler-steps 8. AdapterSlots
runs its full stack: grouped kernels, KV-aware admission and adapter-aligned
admission. The swept deadline, AS_GROUP_MAX_WAIT_S, is how long a waiting
request may be held so that each admitted wave stays aligned to as few
adapters as possible; a request that has waited longer is admitted in arrival
order. Deadline 0 turns alignment at admission off.

Each point logs every decode batch (AS_LOG_WAR=1); the row keeps the mean WAR
and GWAR(8) of those batches next to throughput.

Settings (environment): RANK (default 64), ADIR, SEED (default 42), C3_NP
(measured prompts, default 2000), DEADLINES (seconds, comma list), C3_EXTRA
(extra engine flags), OUT, PORT0, GPU.
"""
import os

from common import as_env, batch_metrics, free_port, load, out_dir, record, rep_row, setup

setup()
import benchmarks.ablations.bench as BM  # noqa: E402

RANK = int(os.environ.get("RANK", "64"))
ADIR = os.environ.get("ADIR", "./adapters_13b_r64")
SEED = int(os.environ.get("SEED", "42"))
NP = int(os.environ.get("C3_NP", "2000"))
DEADLINES = [float(x) for x in os.environ.get("DEADLINES", "0,15,30,60,100,10000").split(",")]
OUT = out_dir(f"c3_deadline_r{RANK}_s{SEED}")
BM.MAX_TOKENS = 64
port = int(os.environ.get("PORT0", "51100"))

for d in DEADLINES:
    tag = f"dl{d:g}"
    if tag in load(OUT):
        continue
    port = free_port(port + 1)
    mpath = os.path.abspath(f"{OUT}/{tag}.metrics.jsonl")
    if os.path.exists(mpath):
        os.remove(mpath)
    envs = dict(as_env(), AS_KV_ADMISSION="1", AS_ADMIT_GROUPED="1",
                AS_GROUP_MAX_WAIT_S=str(d), AS_LOG_WAR="1", AS_METRICS_PATH=mpath)
    os.environ.update(envs)
    try:
        kw = dict(backend_name="adapterslots", mode="C7", model="./models/llama-13b",
                  adapter_dirs=BM._list_adapters(ADIR, 16, RANK), num_adapters=16, rank=RANK,
                  target_modules="all_linear", request_rate=300, dataset="sharegpt",
                  pattern="zipf", num_prompts=NP, warmup=30, reps=1, seeds=[SEED], tp=1,
                  output=f"{OUT}/{tag}.json", port=port,
                  extra_args=["--max-num-seqs", "256", "--num-scheduler-steps", "8"]
                  + os.environ.get("C3_EXTRA", "").split(),
                  tmax_ms=90, wgkp_threshold=16)
        BM.run_benchmark(**kw)
        row = dict(rep_row(kw["output"]), deadline_s=d, **(batch_metrics(mpath) or {}))
        record(OUT, tag, row)
        print(f"{tag} tput={row['tput']:.1f} gwar8={row.get('gwar8', 0):.4f}", flush=True)
    except Exception as e:
        print(f"{tag} FAILED: {type(e).__name__}: {e}", flush=True)
    finally:
        for k in envs:
            os.environ.pop(k, None)

res = load(OUT)
pts = sorted((v["deadline_s"], v["tput"], v["gwar8"]) for v in res.values() if "gwar8" in v)
if len(pts) >= 3:
    import numpy as np
    r = np.corrcoef([p[2] for p in pts], [p[1] for p in pts])[0, 1]
    print(f"Pearson r(GWAR(8), throughput) = {r:.4f} over {len(pts)} points", flush=True)
