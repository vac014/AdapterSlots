"""One serving cell per arm, saturated arrivals (rate 300), warmup 30.

Used for Table V (LLaMA-13B), the C1 cells (LLaMA-7B, K=10) and the cumulative
ablations behind Table I. Both arms run the same engine flags:
--max-num-seqs <seqs> --num-scheduler-steps 8, plus $EXTRA_ARGS.

Settings (environment):
  MODEL, ADIR, RANK, K       model, adapter directory, LoRA rank, adapter count
  SEED, NP                   trace seed and number of measured prompts
  CELLS                      comma list of max_tokens:max_num_seqs:arm
  EXTRA_ARGS                 appended to both arms (e.g. "--max-loras 4")
  SEGMODE                    AdapterSlots kernel mode (default 3)
  OUT, PORT0, GPU            output directory, first port, visible GPUs

Arms:
  C0      vLLM
  C6      AdapterSlots WGKP scheduling on vLLM's LoRA kernels
  C7      AdapterSlots WGKP scheduling with the grouped kernels ($SEGMODE)
  nolora  vLLM serving the base model with LoRA disabled

Any AS_* variable in the environment reaches the AdapterSlots server
(AS_KV_ADMISSION, AS_ADMIT_GROUPED, AS_MAX_LORAS, ...). With AS_LOG_WAR=1 the
row also carries the mean WAR and GWAR(8) of the logged decode batches.
"""
import os

from common import as_env, batch_metrics, free_port, load, out_dir, record, rep_row, setup, \
    vllm_nolora_backend

setup()
import benchmarks.ablations.bench as BM  # noqa: E402

OUT = out_dir("cell")
MODEL = os.environ.get("MODEL", "./models/llama-7b")
RANK = int(os.environ.get("RANK", "32"))
K = int(os.environ.get("K", "10"))
ADIR = os.environ.get("ADIR", "./adapters")
NP = int(os.environ.get("NP", "400"))
SEED = int(os.environ.get("SEED", "42"))
port = int(os.environ.get("PORT0", "28100"))

for cell in os.environ["CELLS"].split(","):
    mt, seqs, arm = cell.split(":")
    tag = f"r{RANK}_mt{mt}_seqs{seqs}_{arm}_s{SEED}_n{NP}"
    if tag in load(OUT):
        continue
    port = free_port(port + 1)
    BM.MAX_TOKENS = int(mt)
    is_as = arm in ("C6", "C7")
    envs = dict(as_env(), AS_SEGGEMM="0") if arm == "C6" else as_env() if arm == "C7" else {}
    mpath = os.path.abspath(f"{OUT}/{tag}.metrics.jsonl")
    if is_as and os.environ.get("AS_LOG_WAR") == "1":
        envs["AS_METRICS_PATH"] = mpath
        if os.path.exists(mpath):
            os.remove(mpath)
    os.environ.update(envs)
    saved = BM.VLLMBackend
    if arm == "nolora":
        BM.VLLMBackend = vllm_nolora_backend()
    try:
        kw = dict(backend_name="adapterslots" if is_as else "vllm",
                  mode=arm if is_as else "C0", model=MODEL,
                  adapter_dirs=BM._list_adapters(ADIR, K, RANK), num_adapters=K, rank=RANK,
                  target_modules="all_linear", request_rate=300, dataset="sharegpt",
                  pattern="zipf", num_prompts=NP, warmup=30, reps=1, seeds=[SEED], tp=1,
                  output=f"{OUT}/{tag}.json", port=port,
                  extra_args=["--max-num-seqs", seqs, "--num-scheduler-steps", "8"]
                  + os.environ.get("EXTRA_ARGS", "").split())
        if is_as:
            kw.update(tmax_ms=90, wgkp_threshold=16)
        BM.run_benchmark(**kw)
        row = dict(rep_row(kw["output"]), mt=int(mt), seqs=int(seqs), arm=arm)
        m = batch_metrics(mpath) if is_as else None
        if m:
            row.update(m)
        record(OUT, tag, row)
        print(f"{tag} tput={row['tput']:.1f} tok/req={row['tok_per_req']:.1f}", flush=True)
    except Exception as e:
        print(f"{tag} FAILED: {type(e).__name__}: {e}", flush=True)
    finally:
        BM.VLLMBackend = saved
        for k in envs:
            os.environ.pop(k, None)
