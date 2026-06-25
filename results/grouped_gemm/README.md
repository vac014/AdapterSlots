# Grouped-GEMM Level-2 Promotion

Level 2 of WGKP implemented as a grouped GEMM over the adapter-contiguous decode
batch the AlignmentBuffer already produces, with the measurements behind it.

## Method

`vllm_scheduler.py` sorts `scheduled_seq_groups` by adapter id, so an AS decode
batch consists of contiguous same-adapter segments. vLLM's SGMV kernel is a
GroupGEMM+SPLIT-K built on `tl.dot`, but it is only reached on the prefill path,
because a vanilla decode batch is not sorted. `segmented_punica_wrapper.py`
routes decode through that kernel by overriding `shrink_decode`,
`expand_decode` and `expand_slice_decode`; `tiled_segment_gemm.py` adds a
tile-scheduled variant whose grid covers only the occupied work tiles.

The SGMV kernels read segment length and adapter index from device tensors and
mask their stores, so the launch grid can be sized to a static worst case and
captured into a CUDA graph. Segment metadata is rebuilt once per step in
`update_metadata()`, outside any captured region and without a device-to-host
sync.

Promotion is gated on token count (`AS_SEGGEMM_MIN_TOKENS`, default 192); below
the crossover the stock BGMV path is used. The crossover is shape-dependent:
roughly 192 tokens at rank 16, 48-64 at rank 32 and above. Block sizes are
rank-dependent; vLLM's defaults are tuned for rank 16.

    AS_SEGGEMM=1   static grid (better at rank 64+ with large batches)
    AS_SEGGEMM=2   tile-scheduled grid (better at lower rank)

## Setup

RTX A6000, LLaMA FP16, vLLM 0.6.3, ShareGPT prompts with Zipf adapter routing,
K=10, `--max-num-seqs 256`, `--num-scheduler-steps 8`. Both arms run the same
base weights with identical flags; only `AS_SEGGEMM` differs. 7B uses one GPU;
13B uses two (TP=2), since one 48 GB card cannot hold 13B weights, rank-64+
LoRA slots and a batch large enough for the grouped path. Each rank measures the
vLLM baseline twice. `ceiling` is throughput with LoRA disabled divided by
throughput with LoRA, i.e. the most any LoRA-side change can gain.

## Rank sweep (`rank_sweep/`)

| rank | 7B vLLM | 7B AS | ratio | ceiling | 13B vLLM | 13B AS | ratio | ceiling |
|-----:|--------:|------:|------:|--------:|---------:|-------:|------:|--------:|
|  16  | 1937.5 | 2029.2 | 1.047 | 1.270 | 1529.7 | 1660.4 | 1.085 | 1.233 |
|  32  | 1708.8 | 1950.7 | 1.142 | 1.440 | 1362.6 | 1603.7 | 1.177 | 1.385 |
|  64  | 1419.4 | 2038.1 | 1.436 | 1.733 | 1143.8 | 1622.0 | 1.418 | 1.650 |
| 128  |  872.4 | 1823.0 | 2.090 | 2.820 |  699.9 | 1460.3 | 2.087 | 2.696 |

No-LoRA baseline: 7B 2460.5, 13B 1886.9 tok/s. Every ratio is below its ceiling.

AS throughput is close to rank-invariant (7B 2029 to 1823, -10%) while vLLM falls
by more than half (1937 to 872, -55%), on both models. BGMV re-gathers an `r x h`
weight slab per token, so its cost scales with rank; the grouped GEMM loads each
adapter's weights once per segment. The LoRA share of iteration time therefore
rises with rank (7B: 21.3% at rank 16, 64.5% at rank 128), which raises both the
ceiling and the achievable gain. Speedups are rank-specific and should be quoted
with their rank.

## Replication (`replication/`)

Three seeds per point, ratio computed within each seed.

| config | mean | min | max | CV |
|--------|-----:|----:|----:|---:|
| 7B rank 32   | 1.163 | 1.136 | 1.208 | 2.72% |
| 7B rank 64   | 1.542 | 1.531 | 1.550 | 0.54% |
| 7B rank 128  | 2.090 | 2.050 | 2.130 | 1.55% |
| 13B rank 64  | 1.400 | 1.371 | 1.426 | 1.64% |
| 13B rank 128 | 2.073 | 1.991 | 2.123 | 2.82% |

## Kernel microbenchmarks (`microbench/`)

CUDA-graph timings of the LoRA operation alone, h=4096, N=256, K=10:

| rank | BGMV | grouped | speedup |
|-----:|-----:|--------:|--------:|
|  64  | 274.4 us | 38.4 us |  7.14x |
| 128  | 733.6 us | 53.4 us | 13.74x |

`lora_share.csv` holds the with/without-LoRA runs used for the ceilings.

## Tensor parallelism

`aligned_mp_engine.py` injects `AlignmentAwareModelRunner` for both
`GPUExecutor` and `MultiprocessingGPUExecutor`. Only the former was handled
previously, so a TP>1 server ran without the kernel layer and performed like
stock vLLM while the scheduler still reordered batches. Check the install with

    grep -c "install_segmented_punica_wrapper: installed=True" <server log>

which should print one line per rank.

## Running

    AS_SEGGEMM=1 AS_SEGGEMM_MIN_TOKENS=64 BENCH_CONN_LIMIT=1024 \
      python benchmarks/ablations/bench.py --backend adapterslots --mode C7 \
        --model ./models/llama-7b --num-adapters 10 --rank 64 \
        --request-rate 300 --num-prompts 400 \
        --extra-args --max-num-seqs 256 --num-scheduler-steps 8

`BENCH_CONN_LIMIT` raises the aiohttp connector limit. The default
`TCPConnector(limit=100)` caps in-flight requests at 100 regardless of
`--max-num-seqs`, which leaves the server under-driven.
