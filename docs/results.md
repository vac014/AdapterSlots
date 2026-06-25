# Results

We ran every experiment on NVIDIA RTX A6000 48 GB GPUs (two over PCIe for C2
and TP=2) with FP16 LLaMA-7B and LLaMA-13B, vLLM 0.6.3 under AdapterSlots,
ShareGPT prompts and Zipf-0.9 adapter routing. Adapters are LoRA on all seven linear
projections. Throughput is output tokens per second over the measured requests.
A seed fixes one request trace, and every system in an experiment replays the
same trace; where we give a speedup over several seeds it is the mean of the
per-seed ratios.

AdapterSlots runs with WGKP scheduling (`AS_MODE=wgkp`), the grouped LoRA
kernels (`AS_SEGGEMM=3`, from 64 tokens per batch) and adapter-aligned admission
(`AS_ADMIT_GROUPED=1`), plus KV-aware admission (`AS_KV_ADMISSION=1`) on
LLaMA-13B. Unless a section says otherwise, vLLM 0.6.3 and AdapterSlots both run
with `--max-num-seqs 256 --num-scheduler-steps 8`.

You can find every number below in `results/claims/claims.json` and
`results/claims/tables.md`, which `benchmarks/paper/summarize.py` builds from
the per-run rows in `results/runs/`; [`REPRODUCE.md`](../REPRODUCE.md) gives the
command for each experiment.

| Claim | Result |
|---|---|
| C1: single GPU, K=10, rank 32 | 2.180x over vLLM 0.5.0 and 1.408x over vLLM 0.6.3 (3 seeds) |
| C2: APIS, K=50, two A6000 over PCIe | 2.680x over vLLM TP=2 (3 seeds, CV 1.1%) |
| C3: GWAR(8) against throughput | Pearson r = 0.974 on each of two traces (n = 6) |
| Table V: LLaMA-13B, one A6000 | 1.509x, 1.533x, 1.570x at K = 4, 8, 16 |

## Adapter mixing and SGMV

We time SGMV on the same tokens in different adapter layouts. With K=16
adapters and N=2048 tokens, SGMV throughput on a fully interleaved batch is 88.5% lower than on the same tokens in adapter-aligned order (80.6% to
89.2% across N = 512 to 4096 and K = 2 to 16; `results/e1/scale/`).

We profile the SGMV shrink kernel with Nsight Compute (K=4, N=4096,
`results/e1/ncu/a6000/`):

| Layout | thread blocks | DRAM throughput (% of peak) | warps active | L2 hit rate |
|---|---|---|---|---|
| A: one adapter | 16 | 9.6 | 8.33% | 86.1% |
| B: two adapters, contiguous | 32 | 18.1 | 8.34% | 86.0% |
| S: four adapters, sorted | 64 | 30.1 | 8.32% | 85.9% |
| C: two adapters, 16-token blocks | 256 | 73.4 | 15.21% | 75.0% |
| D: four adapters, token-by-token interleaved | 4096 | 6.9 | 16.62% | 97.1% |

Across the adapter-aligned layouts occupancy stays at 8.33% while DRAM
throughput grows with the number of distinct weight sets, 3.14x from one to four.
Finer interleaving multiplies the segments and the thread blocks: with 16-token
blocks DRAM throughput reaches 73.4%, and token by token the kernel runs 4096
single-token segments over four weight sets that stay resident in L2.

## CASH dispatch cost

We time the CASH ranking on the CPU: it costs 21 us per scheduler tick at K=4 and
121 us at K=128, linear in K (R^2 = 0.9999; `results/cash_dispatch/`).

## Kernels

We time the LoRA projections through vLLM's `PunicaWrapper` classes under a CUDA
graph (`results/runs/kernel/`). For one LLaMA-13B decode step at K=16, batch 100,
rank 32 (all seven projections, 40 layers), the grouped kernels spend 8.73 ms on
LoRA; vLLM's BGMV path spends 21.22 ms on the same adapter-sorted batch and
24.85 ms on the batch in arrival order. On random LoRA weights the grouped path's
output matches vLLM's to within 1e-3 (fp16) at every measured K, batch size and
rank.

psi_fuse, the fused Triton kernel over SGMV, is 1.910 at n = 32 tokens, rank 16.
psi_promote, the merged-weight GEMM over SGMV, is 1.519 at n = 8, rank 32
(`results/runs/kernel/fused_crossover/`).

Served end to end, AdapterSlots produces the same greedy output as vLLM for
156 of 160 prompts (48 tokens each), with a mean shared prefix of 0.983. The
generated adapters keep PEFT's initialization (B = 0), so this check covers the
serving path end to end, and the per-layer check above covers the LoRA
arithmetic.

L3 promotes a step when all its tokens belong to one adapter whose merged
weights are resident; with the K = 10 mixes of our serving runs that never
happens, and the merged-weight cache holds no adapter at vLLM's default memory
share. We serve it on its own (`benchmarks/paper/l3_check.py`): LLaMA-7B, one
rank-32 adapter with random B, eager mode, 75% of GPU memory for vLLM, which
leaves room for two merged adapters. With L3 the server reaches 1939.6 tok/s
against 1832.1 without (1.059x), and 189 of 200 greedy completions match the
LoRA path exactly (mean shared prefix 0.956).

## Table I: component stack

We run the lattice on LLaMA-7B, K=10, rank 32, saturated arrivals (rate 300),
400 requests, seeds 42 to 44. Each mode is one configuration of the stack
(`backends/backend_adapterslots.py`); WAR and GWAR(8) come from every logged
decode batch.

| Mode | Stack | tok/s | gain vs vLLM 0.5.0 | gain vs vLLM 0.6.3 | WAR | GWAR(8) |
|---|---|---|---|---|---|---|
| C1 | Buffer, threshold T_max | 1864.6 | 1.628x | 1.051x | 0.30 | 0.97 |
| C2 | Erlang T*(k) | 1876.9 | 1.639x | 1.058x | 0.28 | 0.98 |
| C3 | PI-controlled T_max | 1860.8 | 1.625x | 1.049x | 0.30 | 0.99 |
| C4 | CASH order | 1885.2 | 1.646x | 1.063x | 0.30 | 0.97 |
| C5 | CASH order + PredictiveLFU | 1858.1 | 1.623x | 1.048x | 0.33 | 0.97 |
| C6 | WGKP holdback, vLLM LoRA kernels | 1869.6 | 1.633x | 1.054x | 0.31 | 0.97 |
| C7 | Full: WGKP holdback, grouped kernels | 2514.5 | 2.196x | 1.418x | 0.28 | 0.95 |

Every mode makes the decode batches adapter-aligned (GWAR(8) 0.95 to 0.99), but
on vLLM's LoRA kernels alignment alone stays within 7% of vLLM 0.6.3 (C1 to C6).
The WGKP kernel tier turns the alignment into throughput: C7, the same
scheduling as C6 on the grouped kernels, is 1.418x vLLM 0.6.3 on the same engine
flags.

We also add the mechanisms one at a time on LLaMA-13B (one A6000, rank 64,
K=16, 64-token outputs, seed 42; `results/runs/table1_ablation_13b/`):

| Step | tok/s | gain vs vLLM 0.6.3 | GWAR(8) |
|---|---|---|---|
| vLLM 0.6.3 | 730.6 | 1.000x | |
| WGKP scheduling, vLLM LoRA kernels | 677.8 | 0.928x | 0.51 |
| + KV-aware admission | 801.1 | 1.096x | 0.63 |
| + grouped LoRA kernels (WGKP L2) | 997.3 | 1.365x | 0.63 |
| + fused QKV and gate_up projections | 1017.0 | 1.392x | 0.63 |
| + grouped prefill | 1034.0 | 1.415x | 0.63 |
| + adapter-aligned admission (full) | 1102.2 | 1.508x | 0.86 |

On LLaMA-13B the KV cache of one card is the binding limit, so admission decides
how aligned a batch can get: adapter-aligned admission raises
GWAR(8) from 0.63 to 0.86 and adds 7% throughput on top of the kernels. The same
steps on the C1 configuration give 1.000x, 1.320x, 1.337x, 1.340x and 1.358x
over vLLM 0.6.3 (`results/runs/table1_ablation_7b/`).

## C1 and Table II: K=10 against other multi-LoRA systems

We compare against four systems on LLaMA-7B, K=10, rank 32, 128-token outputs,
saturated arrivals, 30 warmup and 400 measured requests (`results/runs/c1_k10/`).

| System | s42 | s43 | s44 | mean tok/s | AdapterSlots speedup |
|---|---|---|---|---|---|
| AdapterSlots | 2490.7 | 2527.3 | 2472.4 | 2496.8 | |
| vLLM 0.5.0 | 1141.0 | 1172.4 | 1122.7 | 1145.4 | 2.180x |
| vLLM 0.6.3 | 1850.8 | 1623.5 | 1871.7 | 1782.0 | 1.408x |
| S-LoRA | 1856.2 | 1925.2 | 1820.1 | 1867.2 | 1.338x |
| Punica | 1793.7 | 1812.6 | 1797.3 | 1801.2 | 1.386x |
| dLoRA | 1059.1 | 1066.4 | 1086.0 | 1070.5 | 2.333x |

vLLM 0.5.0, S-LoRA and dLoRA run as servers on the same trace; Punica runs its
own batching engine in process on the trace's prompt lengths and adapter ids.
S-LoRA applies the adapters to the attention projections, which is what its
kernels support. Per-system setup is in
[`benchmarks/sota/REPRODUCE_SOTA.md`](../benchmarks/sota/REPRODUCE_SOTA.md).

At lambda = 7 req/s, with 384-token outputs so that the
offered load exceeds every system's capacity (1000 requests,
`results/runs/c1_lam7_mt384/`), AdapterSlots serves 1836.6, 1812.9 and 1814.9
tok/s: 1.673x vLLM 0.5.0 (seeds 42, 43) and 1.236x vLLM 0.6.3.

## C2: APIS on two GPUs

We serve K=50 rank-128 adapters for LLaMA-7B from two A6000 over PCIe, saturated
arrivals, 1000 requests (`results/runs/c2_apis/`).

| Arm | s42 | s43 | s44 |
|---|---|---|---|
| APIS (two TP=1 AdapterSlots shards) | 2728.7 | 2660.0 | 2733.6 |
| vLLM TP=2 | 1010.4 | 1008.6 | 1012.2 |
| two TP=1 vLLM servers, round-robin, 25 LoRA slots each | 945.6 | 1317.1 | 1376.5 |

APIS assigns the adapters to the shards by rate-sorted round-robin and reaches
2.680x vLLM TP=2 (2.637x to 2.701x). The two-server vLLM arm has the TP=1 topology and the
per-shard LoRA budget of APIS without its adapter assignment and scheduling: on
its own it gives 1.201x over TP=2, and APIS is 2.297x over it.

## C3: alignment drives throughput

We sweep the alignment deadline of adapter-aligned admission (how long a request
may wait so that admitted waves stay aligned) over six points from 0 to 10000 s.
LLaMA-13B, one A6000, rank 64, K=16, 64-token outputs, 2000 requests, two runs
per point (`results/runs/c3_deadline/`).

| deadline (s) | 0 | 15 | 30 | 60 | 100 | 10000 |
|---|---|---|---|---|---|---|
| seed 42, tok/s | 1151.5 | 1180.8 | 1196.7 | 1235.5 | 1245.3 | 1237.5 |
| seed 42, GWAR(8) | 0.585 | 0.625 | 0.688 | 0.859 | 0.936 | 0.921 |
| seed 43, tok/s | 1115.3 | 1162.3 | 1175.7 | 1217.3 | 1246.7 | 1234.5 |
| seed 43, GWAR(8) | 0.552 | 0.607 | 0.669 | 0.847 | 0.930 | 0.930 |

GWAR(8) and throughput rise together: Pearson r = 0.974 on seed 42 and 0.974 on
seed 43. A longer deadline admits waves from fewer adapters, the grouped kernels
then read fewer weight sets per step, and throughput follows GWAR.

## Table III: K=4 latency

We measure latency at K=4, rank 16, lambda = 7 req/s, 500 requests, LLaMA-7B
(`results/runs/table3_k4/`). Both systems run with vLLM's default engine flags;
AdapterSlots adds `--disable-async-output-proc`, so the first token leaves right
after the prefill step. Engine TTFT is measured inside the server from arrival to first token;
client TTFT includes the HTTP stream.

| Run | tok/s | engine TTFT p50 / p99 (ms) | client TTFT p50 / p99 (ms) |
|---|---|---|---|
| AdapterSlots, seed 42 | 796.2 | 42.5 / 69.9 | 70.0 / 126.8 |
| AdapterSlots, seed 43 | 841.3 | 43.1 / 73.0 | 72.2 / 144.4 |
| AdapterSlots, seed 44 | 872.7 | 43.4 / 96.3 | 75.9 / 149.2 |
| vLLM 0.6.3, seed 42 | 770.7 | | 250.3 / 480.6 |

## Table V: LLaMA-13B on one GPU

We serve LLaMA-13B from one A6000 with rank-64 adapters, 64-token outputs,
saturated arrivals, 2000 requests, seeds 42 to 44 (`results/runs/table5_13b/`).
Both systems get the same number of GPU LoRA slots.

| K | LoRA slots | vLLM tok/s | AdapterSlots tok/s | speedup |
|---|---|---|---|---|
| 4 | 4 | 915.5 | 1381.3 | 1.509x |
| 8 | 8 | 854.6 | 1309.7 | 1.533x |
| 16 | 16 | 756.4 | 1187.3 | 1.570x |
| 4 | 16 | 765.4 | 1249.9 | 1.633x |
| 8 | 16 | 772.1 | 1217.9 | 1.577x |

With one slot per adapter the speedup rises with adapter diversity, from 1.509x
at K=4 to 1.570x at K=16: from K=4 to K=16 vLLM drops from 915.5 to 756.4 tok/s
and AdapterSlots from 1381.3 to 1187.3.

With TP=2 on both A6000s (rank 32, 128-token outputs, 400 requests,
`results/runs/table5_13b_tp2/`), AdapterSlots is 1.301x vLLM at K=4 (seeds 42 to
44), 1.290x at K=8 and 1.306x at K=16.
