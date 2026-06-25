"""K=10 comparison (C1, Table II, Table I lattice): LLaMA-7B, rank 32 on all
linear projections, ShareGPT prompts, Zipf-0.9 adapter routing.

Arms on the identical request trace per seed:
  as        AdapterSlots C7 (grouped kernels, $SEGMODE), --num-scheduler-steps 8
  asC1..C7  AdapterSlots lattice modes (backends/backend_adapterslots.py)
  v063      vLLM 0.6.3 with the AdapterSlots engine flags
  v050      vLLM 0.5.0, its defaults ($VLLM050_PYTHON)
  slora     S-LoRA from deps/slora (backends/backend_slora.py)

Settings (environment): RATE (default 7), NP (default 1000), WU (warmup,
default 20), MT (output tokens, default 128), SEEDS, ARMS, OUT, PORT0, GPU,
SEGMODE. AS_* variables reach the AdapterSlots server. With AS_LOG_WAR=1 the
AdapterSlots rows carry the mean WAR and GWAR(8) of the logged decode batches.
"""
import os
import statistics

from common import REPO, as_env, batch_metrics, free_port, load, out_dir, record, rep_row, \
    setup, vllm050_backend

setup()
os.environ.setdefault("SLORA_ROOT", str(REPO / "deps" / "slora"))
import benchmarks.ablations.bench as BM  # noqa: E402

if os.environ.get("MT"):
    BM.MAX_TOKENS = int(os.environ["MT"])
NP = int(os.environ.get("NP", "1000"))
RATE = float(os.environ.get("RATE", "7"))
WU = int(os.environ.get("WU", "20"))
OUT = out_dir(f"k10_rate{RATE:g}_n{NP}")
EXTRA = ["--max-num-seqs", "256", "--num-scheduler-steps", "8"]
ARMS = os.environ.get("ARMS", "v050,as,v063").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,43,44").split(",")]
port = int(os.environ.get("PORT0", "30100"))

for seed in SEEDS:
    for arm in ARMS:
        tag = f"{arm}_s{seed}"
        if tag in load(OUT):
            continue
        port = free_port(port + 1)
        is_as = arm == "as" or arm.startswith("asC")
        # C6 runs WGKP scheduling on vLLM's LoRA kernels, as in cell.py.
        envs = (dict(as_env(), AS_SEGGEMM="0") if arm == "asC6"
                else as_env() if is_as else {})
        mpath = os.path.abspath(f"{OUT}/{tag}.metrics.jsonl")
        if is_as and os.environ.get("AS_LOG_WAR") == "1":
            envs["AS_METRICS_PATH"] = mpath
            if os.path.exists(mpath):
                os.remove(mpath)
        os.environ.update(envs)
        saved = BM.VLLMBackend
        if arm == "v050":
            BM.VLLMBackend = vllm050_backend()
        try:
            backend = "adapterslots" if is_as else ("slora" if arm == "slora" else "vllm")
            kw = dict(backend_name=backend,
                      mode=arm[2:] if arm.startswith("asC") else ("C7" if arm == "as" else "C0"),
                      model="./models/llama-7b", adapter_dirs=BM._list_adapters("./adapters", 10, 32),
                      num_adapters=10, rank=32, target_modules="all_linear", request_rate=RATE,
                      dataset="sharegpt", pattern="zipf", num_prompts=NP, warmup=WU, reps=1,
                      seeds=[seed], tp=1, output=f"{OUT}/{tag}.json", port=port,
                      extra_args=None if arm in ("v050", "slora") else EXTRA)
            if is_as:
                kw.update(tmax_ms=90, wgkp_threshold=8)
            BM.run_benchmark(**kw)
            row = rep_row(kw["output"])
            m = batch_metrics(mpath) if is_as else None
            if m:
                row.update(m)
            record(OUT, tag, row)
            print(f"{tag} tput={row['tput']:.1f} p50={row['ttft_p50']:.0f} "
                  f"tok/req={row['tok_per_req']:.1f}", flush=True)
        except Exception as e:
            print(f"{tag} FAILED: {type(e).__name__}: {e}", flush=True)
        finally:
            BM.VLLMBackend = saved
            for k in envs:
                os.environ.pop(k, None)
res = load(OUT)
for b in ("v050", "v063", "slora"):
    rs = [res[f"as_s{s}"]["tput"] / res[f"{b}_s{s}"]["tput"] for s in SEEDS
          if f"as_s{s}" in res and f"{b}_s{s}" in res]
    if rs:
        print(f"as/{b} = {statistics.mean(rs):.4f} over {len(rs)} seeds", flush=True)
