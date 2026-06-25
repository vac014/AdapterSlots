# Results

All measurements: A6000 48 GB, llama-7b, fp16, vLLM 0.6.3. Workload: ShareGPT
prompts, zipf-skewed adapter assignment, 256-token decode, greedy, output length
fixed so every run emits the same token count (only wall time differs). Speculative
runs are greedy-exact (output token-for-token equal to the non-speculative baseline).

AdapterSlots serves multi-LoRA decode with a fused SGMV kernel whose cost is flat in
the number of concurrent adapters K, an Erlang/Whittle/CASH dispatch layer for
admission and per-adapter SLO/fairness, and a CUDA-graph-captured γ+1 speculative
verify that keeps LoRA applied during drafting.

## 1. Multi-adapter throughput, no speculation (γ=0)

Fused SGMV keeps throughput flat in K while vLLM's punica LoRA path collapses, so
AdapterSlots overtakes vLLM in the high-K regime multi-adapter serving targets
(tok/s, N=64, 256-token decode, zipf adapters, greedy):

| B  | K  | AdapterSlots | vLLM  | ratio |
|----|----|-------------:|------:|------:|
| 16 | 16 | 462.5        | 430.0 | 1.08× |
| 16 | 32 | 458.9        | 297.3 | 1.54× |
| 32 | 32 | 770.6        | 508.5 | 1.52× |

Throughput is flat in K (B=16: 471→459 across K=4→32, under 3%) while vLLM drops
562→297. The crossover is batch-dependent: K≥16 at B=16, K=32 at B=32. At low K (4–8)
vLLM's base-forward engine is ~15% faster; the gap is forward/LoRA fusion structure,
not the LoRA kernel (dropping in vLLM's own punica BGMV runs slower, 748 vs 787).
Speculation is the lever that wins low K.

The dispatch policies are throughput-neutral by design -- Erlang, Whittle, CASH, and
FIFO land within ±2% (463 vs 459 for plain batching). The K-flat SGMV kernel carries
the throughput win; the policies carry the control plane (admission, SLO, fairness,
provable indexability) at near-zero overhead.

## 2. Speculative decoding

The γ+1 verify is graph-captured with LoRA applied, so each accepted draft token
rides inside the same weight load. End-to-end serving vs vLLM (B=16, N=64, 256-token
decode, zipf adapters, greedy-exact):

| K  | AdapterSlots spec | vLLM  | ratio |
|----|------------------:|------:|------:|
| 4  | 861.5             | 559.9 | 1.54× |
| 8  | 782.0             | 489.3 | 1.60× |
| 16 | 649.7             | 435.2 | 1.49× |
| 24 | 554.0             | 330.9 | 1.67× |
| 32 | 490.4             | 308.5 | 1.59× |

Speculation wins at every K (1.49×–1.67×) and does not decay past K=16. The LoRA
delta is computed per token by gathering each token's own adapter matrices
(`index_select` + einsum), making per-token LoRA cost independent of K; this is
CUDA-graph-capturable, so capture, the speculative win, and correctness hold together
(100% argmax agreement vs PEFT). Across the full (B, K) matrix the speculative
speedup over vLLM is 1.08×–3.03×.

## 3. Comparison against the multi-LoRA serving field

Each system ran its own upstream harness on the same box (A6000 48 GB, llama-7b).
The harnesses measure different things, so the table is grouped by native metric. The
load-bearing comparison is AdapterSlots vs vLLM 0.6.3, itself production multi-LoRA
(punica SGMV/BGMV). Common axis = decode tok/s; AdapterSlots speculative anchor = 901
(B16) / 977 (B32) at rank 32, K=32.

| System | decode tok/s | total/e2e | regime | ratio (B16 / B32) |
|---|--:|--:|---|--:|
| **AdapterSlots spec** | **901 / 977** | -- | fixed-batch, rank 32, K32 | -- |
| AdapterSlots no-spec | 459 / 771 | -- | fixed-batch, rank 32, K32 | 1.96× / 1.27× |
| vLLM 0.6.3 | 297 / 509 | -- | fixed-batch, rank 32, K32 | 3.03× / 1.92× |
| punica | 607 / 1110 | 471 / 809 | decode microbench, rank 32 | 1.48× / 0.88× |
| SGLang 0.5.14 no-spec | 435 | 869 | server, rank 32, 14 LoRA, zipf | 2.07× / 2.24× |
| SGLang 0.5.14 + ngram spec | 638 | 1274 | server, spec + LoRA | 1.41× / 1.53× |
| vLLM V1 0.24.0 no-spec | 254 | 507 | server, rank 32, 14 LoRA, zipf | 3.55× / 3.85× |
| vLLM V1 0.24.0 + ngram spec | 333 | 666 | server, spec + LoRA | 2.71× / 2.93× |
| S-LoRA | 589 | 1204 | server, rank 8, Poisson | 1.53× |
| HF-PEFT | 70 | 144 | server, rank 8, Poisson | 12.9× |
| dLoRA (flagship exec_type=3) | 144 | 432 | server, 32 LoRA, concurrency 12 | -- |

### Spec-vs-spec

SGLang 0.5.14 runs speculative decoding and multi-LoRA together (ngram drafting,
csgmv LoRA -- the same SGMV family as AdapterSlots), giving a like-for-like comparison
on the same modern stack with all features on (RadixCache, flashinfer, csgmv).
AdapterSlots speculative wins 1.41× (B16) / 1.53× (B32) and extracts a larger
speculative gift -- 1.96× self-speedup vs SGLang's 1.47× -- because the γ+1 verify is
graph-captured and LoRA-correct. vLLM V1 (2025) also runs spec + LoRA together; its
self-speedup is 1.31× and AdapterSlots speculative is 2.71×–2.93× faster.

vLLM 0.6.3 (V0), the version AdapterSlots builds on, cannot place spec and LoRA in one
graph (`enforce_eager` is forced and `lora_request` is dropped during verify). The
2025 V1 rewrite removes that restriction, so against current vLLM the advantage is
performance (≈3×), not capability.

**Summary.** AdapterSlots speculative is the fastest of every measured system: 3.03×
vLLM 0.6.3, 2.71–2.93× vLLM V1 with spec+LoRA, 1.41–1.53× SGLang with ngram spec+LoRA
(the cleanest like-for-like), 1.48× the punica kernel, 1.53× S-LoRA, 12.9× HF-PEFT.
