# Design

AdapterSlots runs inside vLLM 0.6.3. A scheduler subclass decides which
requests run together and in what order, and a replacement for vLLM's
`PunicaWrapper` runs the LoRA projections on the adapter-aligned batches the
scheduler produces. This note maps each mechanism of the paper to the code.

## Scheduling stack

`adapter_slots/integrations/vllm_scheduler.py` (`AlignmentAwareScheduler`)
replaces vLLM's scheduler; vLLM 0.6.3 has no flag for that, so the server entry
point `scripts/vllm_serve_adapter_slots.py` builds the engine with it
(`integrations/aligned_engine.py`, `aligned_mp_engine.py`). Each tick it takes
vLLM's selection, passes each sequence's first decode token through the
AlignmentBuffer and hands vLLM the batch in adapter order, so a decode batch is
a set of contiguous same-adapter segments.

| Mechanism | Paper | Code | Selected by |
|---|---|---|---|
| WAR and GWAR(n*) | Definition 1 | `adapter_slots/metrics/war.py`, `metrics/gwar.py` | |
| AlignmentBuffer: queues Q_k, dispatch on a full warp or an expired deadline | dispatch rule, Sec. II | `adapter_slots/buffer.py` | `AS_MODE=threshold` (lattice C1) |
| Erlang deadlines T*(k) = min{F^-1(WAR*), T_SLO} from EWMA rates | Theorem 1 | `adapter_slots/dispatch/erlang.py`, `control/estimator.py` | `AS_MODE=erlang` (C2) |
| PI control of T_max, gains kept in 2K_p + K_i < 2/L | Theorem 2 | `adapter_slots/control/pi_controller.py` | `AS_MODE=pi_adaptive` (C3) |
| CASH dispatch order | Proposition 1 | `adapter_slots/dispatch/cash.py` | `AS_MODE=cash` (C4) |
| PredictiveLFU | Sec. IV | `adapter_slots/prefetch/` | `AS_PREFETCH_POLICY=predictive` (C5) |
| CASH holdback with WGKP promotion flags | Sec. III | `buffer.py:form_batch_wgkp`, `kernel/wgkp_dispatcher.py` | `AS_MODE=wgkp` (C6, C7) |
| Alignment-aware prefetch of merged weights | Sec. III | `vllm_scheduler.py:_request_merges`, `kernel/l3.py` | `AS_MWC_K_HOT` (default 5) |
| APIS adapter assignment | Sec. IV | `adapter_slots/integrations/apis_router.py` | `benchmarks/ablations/bench_apis.py` |
| Shadow queues Q_k' for preempted tokens | Sec. IV | `buffer.py:preempt_and_hold`, `resume_from_shadow` | buffer API (unit tests) |
| KV-aware admission | | `vllm_scheduler.py:_schedule_prefills_admit` | `AS_KV_ADMISSION=1` |
| Adapter-aligned admission | | `vllm_scheduler.py:_group_waiting_by_adapter` | `AS_ADMIT_GROUPED=1`, deadline `AS_GROUP_MAX_WAIT_S` |
| Adapter rotation when LoRA slots are fewer than adapters | | `vllm_scheduler.py:_rotation_holdout` | `AS_ROTATE_AFTER_S` (default 3 s) |

The defaults follow the paper: W = 32, WAR* = 0.8, T_SLO = 200 ms, n* = 8 and
K_hot = 5 for the merged-weight cache.

The PI controller applies T_max(t+1) = T_max(t) + K_p (e(t) - e(t-1)) + K_i e(t)
with e = WAR* - WAR every tick, estimates the slope L of WAR against T_max from
finite differences of what it observes, and scales the gains into the stable
region of Theorem 2 before each update.

CASH scores queue k as I_k = lambda_k s_k P_k(s_k): lambda_k is the EWMA rate at
which tokens enter Q_k, s_k = |Q_k| / W the fill fraction, and P_k the Erlang
probability that the W - |Q_k| missing tokens arrive before the queue's deadline
min(T*(k), T_SLO) expires. Queues are visited in decreasing score. In WGKP mode a queue leaves the buffer once it holds a
full warp or n* tokens, once it is half-full past half its deadline, or at its
deadline; n* is capped at one adapter's share of a warp, W / K. Segments of at
least n* tokens whose adapter is resident in the merged-weight cache are flagged
L3-eligible.

Adapter-aligned admission orders the waiting queue adapter by adapter, longest
queue first, so each admitted wave spans as few adapters as possible and every
decode step then reads fewer adapters' weights. A request that has waited longer
than `AS_GROUP_MAX_WAIT_S` is admitted in arrival order. Claim C3 sweeps this
alignment deadline: it decides how aligned the running batch can get, while the
buffer's T_max acts only on each sequence's first decode token. `AS_DEADLINE=fixed`
replaces the Erlang deadlines by one T_max (`AS_TMAX_MS`) for every adapter.

`backends/backend_adapterslots.py` maps the lattice modes `C0` to `C7` to these
settings and starts the server through `scripts/vllm_serve_adapter_slots.py`.

## Kernel tiers

`adapter_slots/kernel/model_runner.py` installs the kernel wrapper when the
server starts.

| Tier | Code | Selected by |
|---|---|---|
| L1: vLLM's own LoRA kernels (BGMV for decode) | vLLM | lattice C1 to C6; batches below `AS_SEGGEMM_MIN_TOKENS` |
| L2: grouped Triton LoRA kernels over the adapter segments | `kernel/segmented_punica_wrapper.py`, `kernel/tiled_segment_gemm.py`, `kernel/tiled_multislice.py` | `AS_SEGGEMM=3` (C7 and every AdapterSlots result) |
| L2: fused base + LoRA Triton kernel (psi_fuse) | `kernel/fused_lora_kernel.py` | measured by `scripts/experiments/fused_kernel_crossover_benchmark.py` |
| L3: GEMM on the merged weight W_k = W + alpha B_k A_k (psi_promote) | `kernel/l3.py`, `kernel/model_runner.py` | `AS_MWC_K_HOT` (default 5); psi_promote measured by the same script |

The L2 gate is open: every segment of a batch of at least
`AS_SEGGEMM_MIN_TOKENS` (64 in our runs) tokens runs on the grouped kernels.
`AS_SEGGEMM=3` runs every LoRA projection of a step as a grouped GEMM over the
adapter segments: a tile map built once per step (no host sync) lets one launch
cover all segments, the packed QKV and gate_up projections each take one shrink
and one expand launch, and prefill uses the same path when its adapter runs fit
the LoRA slots. Block sizes per rank and shape are in
`segmented_punica_wrapper.py` (`_tiled_shrink_cfg`, `_tiled_expand_cfg`;
`benchmarks/paper/kernel_tiles.py` measures them). `AS_SEGGEMM=2` issues one
launch per projection slice, and `AS_SEGGEMM_PREFILL=0` leaves prefill on vLLM's
SGMV path; the Table I ablation uses both.

L3 merges the four attention projections (packed QKV and O) of up to K_hot
adapters into W_k. The scheduler requests a merge once the queue of one of the
K_hot hottest adapters holds half of n* tokens (alignment-aware prefetch); the
model runner performs it between steps. A step is promoted when the paper's gate
holds for the whole step: a decode step whose tokens all belong to one adapter,
at least n* of them, with that adapter's merged weights resident. The merged
weight then replaces the base weight in one GEMM and the attention projections
skip their LoRA kernels. In a mixed batch a merged weight would be an extra
full weight read for its segment, more than the grouped LoRA kernels cost, so
mixed steps stay on L2. Promoted steps cannot replay a CUDA graph,
so promotion also requires a step that runs without one. The cache holds as many
adapters as fit in the GPU memory left after vLLM allocates its KV cache, up to
K_hot, and logs the number; merged weights need the unsharded matrices, so L3
runs with tensor parallelism 1. `benchmarks/paper/l3_check.py` serves one adapter
with memory left for merged weights and compares L3 on and off.

## APIS

`benchmarks/ablations/bench_apis.py` runs two TP=1 AdapterSlots servers, one per
GPU. `APISRouter` assigns adapters by rate-sorted round-robin: adapters ordered by
their EWMA request rate, highest first, the adapter at position i to shard
i mod 2, refreshed every 30 s. The router starts from the trace's request rates
(converged estimates). Each request goes to the shard that holds its adapter, so
each shard serves its part of the adapters with its own LoRA slots and scheduler.

## Measurement

`benchmarks/ablations/bench.py` is the serving harness every driver uses: it
starts a server, replays a request trace (ShareGPT prompts, Zipf adapter ids,
Poisson send times at the given rate), streams the responses and records
throughput and TTFT. With `AS_LOG_WAR=1` the scheduler writes one JSON line per
decode batch (WAR, GWAR, token order by adapter) to `AS_METRICS_PATH`, and with
`AS_REQ_METRICS_PATH` it writes per-request engine timestamps.
