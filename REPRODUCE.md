# Reproducing the results

We produced every result in [`docs/results.md`](docs/results.md) with one command
in `benchmarks/paper/run_all.sh`. The drivers write their rows to `results/runs/`,
and `benchmarks/paper/summarize.py` builds `results/claims/` from them.

## 1. Hardware and software

- We used one NVIDIA RTX A6000 48 GB for everything except C2 and the TP=2
  Table V setting, which ran on two A6000s over PCIe.
- CUDA 12.4, Python 3.10, PyTorch 2.4.0, vLLM 0.6.3.post1; `requirements.txt`
  pins every package of the environment we measured with.

```bash
conda env create -f envs/adapter_env.yml
conda activate adapter_env
pip install -e .
```

`envs/adapter_env.yml` also notes the CUDA paths for building the vendored
baselines. `docker/Dockerfile` builds the same environment as an image, with
Punica and S-LoRA compiled from `deps/`.

## 2. Models, adapters and prompts

None of these ship with the repository; all live at the repository root.

```bash
python scripts/download_models.py --models llama-7b llama-13b      # models/llama-7b, models/llama-13b
python scripts/download_datasets.py                                # data/sharegpt/sharegpt.jsonl

python scripts/gen_adapters.py --model ./models/llama-7b  --K 10 --rank 32  --base-seed 100 --output-dir ./adapters
python scripts/gen_adapters.py --model ./models/llama-7b  --K 32 --rank 16  --output-dir ./adapters_r16d
python scripts/gen_adapters.py --model ./models/llama-7b  --K 10 --rank 128 --output-dir ./adapters_r128
python scripts/gen_adapters.py --model ./models/llama-13b --K 16 --rank 32  --output-dir ./adapters_13b
python scripts/gen_adapters.py --model ./models/llama-13b --K 10 --rank 64  --output-dir ./adapters_13b_r64
```

Adapters are LoRA on all seven linear projections with alpha = 2 x rank. When an
experiment uses more adapter ids than a directory holds (K=16 on
`adapters_13b_r64`, K=50 on `adapters_r128`), `benchmarks/ablations/bench.py`
reuses the weight sets in order; every id is still its own LoRA in the server,
with its own slot and weight copy.

The request traces are drawn from the prompt file by seed, so the traces match
ours only with the same file: `md5sum data/sharegpt/sharegpt.jsonl` should print
`6c9c187e5d0708502c7db53256eb9500`.

## 3. Baseline environments

| Baseline | Setup |
|---|---|
| vLLM 0.6.3 | the AdapterSlots environment |
| vLLM 0.5.0 | `bash envs/setup_vllm050.sh` (creates `venvs/vllm050`) |
| S-LoRA | `cd deps/slora && pip install -e . --no-build-isolation --no-deps` |
| Punica | `cd deps/punica && TORCH_CUDA_ARCH_LIST=8.6 pip install -e . --no-build-isolation --no-deps` |
| dLoRA | `bash envs/setup_dlora.sh` (creates `venvs/dlora`) |

[`benchmarks/sota/REPRODUCE_SOTA.md`](benchmarks/sota/REPRODUCE_SOTA.md) describes
how each one is driven.

## 4. Running

```bash
bash benchmarks/paper/run_all.sh c1 table2     # one or more experiments
bash benchmarks/paper/run_all.sh all           # everything, then summarize
```

| Experiment | Result | GPUs | Time |
|---|---|---|---|
| `e1` | SGMV on mixed batches, Nsight Compute profile (needs `ncu`) | 1 | 20 min |
| `dispatch_cost` | CASH dispatch cost per tick against K | none | 1 min |
| `kernel` | per-layer LoRA timings, tile sweep, output check, L3 served end to end, psi_fuse and psi_promote | 1 | 25 min |
| `c1` | C1 | 1 | 15 min |
| `table2` | Table II baselines | 1 | 40 min |
| `lam7` | C1 at lambda = 7 | 1 | 40 min |
| `lattice` | Table I lattice | 1 | 45 min |
| `ablation13`, `ablation7` | Table I mechanism by mechanism | 1 | 15 min each |
| `c2` | C2 | 2 | 40 min |
| `c3` | C3 | 1 | 2 h |
| `table3` | Table III | 1 | 10 min |
| `table5` | Table V | 1 | 2 h |
| `table5_tp2` | Table V with TP=2 | 2 | 30 min |
| `summarize` | `results/claims/` | none | seconds |

Drivers skip a run whose tag is already recorded in their output directory, so
an interrupted experiment continues where it stopped. To measure into a fresh
tree, set `OUT_ROOT` (default `results/runs`). `GPU` selects the device for
single-GPU experiments (default 0), and `PY` the Python interpreter.

Each driver also runs on its own with environment settings, for example one
Table V cell:

```bash
MODEL=./models/llama-13b ADIR=./adapters_13b_r64 RANK=64 K=16 NP=2000 SEED=42 \
SEGMODE=3 AS_KV_ADMISSION=1 AS_ADMIT_GROUPED=1 AS_MAX_LORAS=16 \
EXTRA_ARGS="--max-loras 16 --max-cpu-loras 16" CELLS=64:256:C7 \
OUT=/tmp/table5 python benchmarks/paper/cell.py
```

The docstring at the top of each file in `benchmarks/paper/` lists its settings,
and [`results/runs/MANIFEST.md`](results/runs/MANIFEST.md) maps every result
directory to its experiment.

## 5. Tests

```bash
pip install -e ".[dev]"
pytest
```
