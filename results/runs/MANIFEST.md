# Measured runs

Each directory holds the rows of one experiment as the drivers in
`benchmarks/paper/` write them: `results.json` with one row per run, the
harness's per-run JSON (`<tag>.json`), per-request engine timestamps where the
driver records them (`<tag>_req.jsonl`), and for AdapterSlots runs that log their
decode batches the scheduler's batch log (`<tag>.metrics.jsonl.gz`). Rows with a
batch log carry `war`, `gwar8`, `gwar8_tok` (token-weighted) and `n_batches`,
computed from the logged token order by `benchmarks/paper/common.py:gwar`.

`bash benchmarks/paper/run_all.sh <experiment>` runs the commands for a
directory, and `benchmarks/paper/summarize.py` turns the rows into
`results/claims/`. All runs use RTX A6000 48 GB GPUs (two for C2 and TP=2),
vLLM 0.6.3 under AdapterSlots, ShareGPT prompts and Zipf-0.9 adapter routing. A
seed fixes one request trace (prompts, adapter ids, send times), and every arm of
an experiment replays the same trace for a given seed.

AdapterSlots settings used below:

| Setting | Meaning |
|---|---|
| `SEGMODE=3` | grouped LoRA kernels (`AS_SEGGEMM=3`): tile-scheduled, packed QKV and gate_up in one launch, prefill grouped |
| `AS_KV_ADMISSION=1` | admit a request only if KV memory covers its whole lifetime |
| `AS_ADMIT_GROUPED=1` | adapter-aligned admission with alignment deadline `AS_GROUP_MAX_WAIT_S` (default 30 s) |
| `AS_LOG_WAR=1` | write the per-batch log |
| `AS_MWC_K_HOT=5` | every AdapterSlots run: L3 merged-weight cache bound; the cache holds the adapters that fit in free GPU memory, which the server log records |

| Directory | What | `run_all.sh` |
|---|---|---|
| `c1_k10/cells` | C1: AdapterSlots (`C7`, `SEGMODE=3 AS_ADMIT_GROUPED=1`) and vLLM 0.6.3 (`C0`), LLaMA-7B, K=10, rank 32, 128-token outputs, rate 300, 400 requests, seeds 42 to 44 | `c1` |
| `c1_k10/baselines` | Table II: vLLM 0.5.0 (`v050`) and S-LoRA (`slora`) on the C1 traces | `table2` |
| `c1_k10/punica`, `c1_k10/dlora` | Table II: Punica and dLoRA on the C1 traces, each through its own engine | `table2` |
| `c1_lam7_mt384` | C1 at lambda = 7 req/s with 384-token outputs, 1000 requests; vLLM 0.5.0 on seeds 42 and 43 | `lam7` |
| `table1_lattice` | Table I, lattice modes `asC1` to `asC7` (`backends/backend_adapterslots.py`) on the C1 configuration, with batch logs; `asC7` runs the grouped kernels, `asC1` to `asC6` vLLM's LoRA kernels | `lattice` |
| `table1_ablation_13b/L0..L6` | Table I mechanism by mechanism: LLaMA-13B, rank 64, K=16, 64-token outputs, seed 42 | `ablation13` |
| `table1_ablation_7b/L1..L6` | the same on the C1 configuration; row 0 is the seed-42 `C0` cell of `c1_k10/cells` | `ablation7` |
| `c2_apis` | C2: APIS, vLLM TP=2, and two TP=1 vLLM servers (`topo`: the APIS adapter assignment; `topo_rr25`: requests round-robin, 25 LoRA slots each), K=50, rank 128, 1000 requests, seeds 42 to 44 | `c2` |
| `c3_deadline/s{42,43}_run{0,1}` | C3: alignment-deadline sweep 0 to 10000 s, LLaMA-13B, rank 64, K=16, 64-token outputs, 2000 requests, two runs per point | `c3` |
| `table3_k4` | Table III: K=4, rank 16, lambda = 7, 500 requests; AdapterSlots with `--disable-async-output-proc` (`k4_sync_as_*`), vLLM with its default flags (`k4_engine_vllm_s42`) | `table3` |
| `table5_13b/k{K}_{a,v}_ml{S}` | Table V: LLaMA-13B on one A6000, rank 64, 64-token outputs, 2000 requests, seeds 42 to 44; `a` AdapterSlots, `v` vLLM, `S` LoRA slots for both (K, or 16) | `table5` |
| `table5_13b_tp2` | Table V setting with TP=2 on two A6000s, rank 32, 128-token outputs, 400 requests | `table5_tp2` |
| `kernel` | per-layer LoRA timings (`kernel_layer_*`), tile sweep (`kernel_tiles_r32.json`), greedy output check (`output_check_*`), L3 served end to end (`l3_check`), fused-kernel crossover (`fused_crossover/`) | `kernel` |

The lattice, the Table I ablations and C3 run with `AS_LOG_WAR=1` and keep their
batch logs; the other experiments measure throughput and latency only.
