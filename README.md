# AdapterSlots

Warp-aligned multi-LoRA serving.

When a decode batch mixes tokens from many LoRA adapters, the batched LoRA
kernel loads one weight set per adapter segment, and throughput falls as the
number of active adapters grows. AdapterSlots schedules requests so that decode
batches are adapter-aligned and then runs the LoRA projections over the aligned
segments with faster kernels. The scheduling stack (alignment buffer, Erlang
deadlines, PI control of the Warp Alignment Ratio, CASH dispatch, adapter-aligned
admission) raises GWAR, and WAR-Gated Kernel Promotion (WGKP) turns aligned
segments into throughput. APIS extends this to two GPUs by giving each GPU its
own share of the adapters. AdapterSlots runs inside vLLM 0.6.3 as a scheduler
subclass and a LoRA kernel wrapper.

This repository holds our implementation, the drivers we ran every experiment
with, and the measured runs behind each result.

## Results

| Claim | Result |
|---|---|
| C1: LLaMA-7B, K=10, rank 32, one RTX A6000 | 2.180x over vLLM 0.5.0, 1.408x over vLLM 0.6.3 |
| Table II: same setting | 1.338x over S-LoRA, 1.386x over Punica, 2.333x over dLoRA |
| C2: APIS, K=50, two RTX A6000 over PCIe | 2.680x over vLLM TP=2 |
| C3: GWAR(8) against throughput | Pearson r = 0.974 and 0.974 |
| Table V: LLaMA-13B, one RTX A6000, K = 4, 8, 16 | 1.509x, 1.533x, 1.570x over vLLM 0.6.3 |

Speedups are means over three request traces, and C3 gives r on each of two
traces. [`docs/results.md`](docs/results.md) has every table with its setup, and [`docs/design.md`](docs/design.md) maps each mechanism
to the code.

## Install

```bash
conda env create -f envs/adapter_env.yml
conda activate adapter_env
pip install -e .
```

Models, adapters and the ShareGPT prompt set do not ship with the repository;
[`REPRODUCE.md`](REPRODUCE.md) shows how to fetch and generate them and how to
run every experiment.

## Layout

| Path | Contents |
|---|---|
| `adapter_slots/` | the package: scheduler integration, alignment buffer, dispatch policies, PI control, kernels, prefetch, metrics |
| `backends/` | server wrappers for AdapterSlots, vLLM and S-LoRA |
| `benchmarks/paper/` | one driver per experiment, `run_all.sh`, `summarize.py` |
| `benchmarks/ablations/` | the serving harness (`bench.py`) and the two-GPU APIS harness (`bench_apis.py`) |
| `benchmarks/micro/` | SGMV microbenchmarks for E1 |
| `scripts/` | model, dataset and adapter setup; E1 layouts, fused-kernel crossover and CASH timing; the server entry point |
| `workloads/` | prompt loading and arrival-pattern generation |
| `results/` | measured outputs (`results/README.md`) |
| `deps/` | vendored baselines: Punica, S-LoRA, dLoRA |
| `envs/` | environment files and setup scripts |
| `docker/` | container recipes for the AdapterSlots environment |
| `gpu_setup/` | topology of the two-A6000 PCIe machine |
| `paper/` | the paper and its figures |
| `tests/` | unit tests |

## Tests

```bash
pip install -e ".[dev]"
pytest
```
