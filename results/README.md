# results

| Path | Contents | Produced by |
|---|---|---|
| `claims/claims.json`, `claims/tables.md` | every number in `docs/results.md` | `benchmarks/paper/summarize.py` |
| `runs/` | serving and kernel runs behind C1 to C3 and Tables I, III, IV and V; `runs/MANIFEST.md` lists each directory | `benchmarks/paper/run_all.sh` |
| `e1/scale/` | SGMV throughput, adapter-aligned against mixed batches, N = 512..4096, K = 2..16 | `run_all.sh e1` |
| `e1/ncu/a6000/` | Nsight Compute export for the five E1 layouts (K=4, N=4096) | `run_all.sh e1` |
| `cash_dispatch/a6000_single/` | CASH dispatch cost per scheduler tick against K | `run_all.sh dispatch_cost` |
