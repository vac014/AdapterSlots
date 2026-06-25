"""Build results/claims/ from the measured runs in results/runs/.

Every number in results/claims/claims.json and results/claims/tables.md is
computed here from the per-run rows. Ratios over seeds are means of per-seed
ratios (each seed is one request trace that both arms replay).

Usage: python benchmarks/paper/summarize.py   (from anywhere; no GPU needed)
"""
import csv
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
RUNS = RES / "runs"
OUT = RES / "claims"
SEEDS = (42, 43, 44)


def rows(path):
    return json.load(open(RUNS / path / "results.json"))


def mean(v):
    return st.mean(v) if v else None


def cv(v):
    return st.pstdev(v) / st.mean(v) if len(v) > 1 else 0.0


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return num / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5


def ratio_block(num, den):
    """num/den: {seed: tput}. Mean, min, max and CV of the per-seed ratios."""
    r = {s: num[s] / den[s] for s in num if s in den}
    v = list(r.values())
    return {"per_seed": {str(s): round(x, 4) for s, x in r.items()},
            "mean": round(mean(v), 4), "min": round(min(v), 4), "max": round(max(v), 4),
            "cv": round(cv(v), 4)}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


claims, tables = {}, []

# E1
e1 = {(r["N"], r["K"]): r for r in read_csv(RES / "e1/scale/scale_sweep_a6000_tp1.csv") if r["condition"] == "D"}
gaps = [float(r["gap_pct"]) for r in e1.values()]
claims["E1"] = {"mixed_batch_gap_K16_N2048": float(e1[("2048", "16")]["gap_pct"]) / 100,
                "gap_range": [min(gaps) / 100, max(gaps) / 100]}
ncu = {}
for c in "ASBCD":
    vals = {}
    for r in read_csv(RES / f"e1/ncu/a6000/ncu_condition_{c}_a6000_latest.csv"):
        if "shrink" not in r["Kernel Name"]:
            continue
        try:
            vals.setdefault(r["Metric Name"].split(".")[0], []).append(float(r["Metric Value"].replace(",", "")))
        except ValueError:
            pass
    ncu[c] = {"dram_pct": round(mean(vals["gpu__dram_throughput"]), 2),
              "warps_active_pct": round(mean(vals["sm__warps_active"]), 2),
              "l2_hit_pct": round(mean(vals["lts__t_sector_hit_rate"]), 2),
              "thread_blocks": int(mean(vals["launch__grid_size"]))}
claims["E1"]["ncu_shrink_K4_N4096"] = ncu
claims["E1"]["dram_one_vs_four_weight_sets"] = round(ncu["S"]["dram_pct"] / ncu["A"]["dram_pct"], 3)
tables += ["## E1: SGMV on mixed batches (LLaMA-7B shapes, rank 16)", "",
           f"SGMV throughput of a fully interleaved batch against the adapter-aligned batch at K=16, N=2048: "
           f"{claims['E1']['mixed_batch_gap_K16_N2048'] * 100:.1f}% lower "
           f"({min(gaps):.1f}% to {max(gaps):.1f}% over N = 512..4096, K = 2..16).", "",
           "Nsight Compute, SGMV shrink, K=4, N=4096:", "",
           "| Layout | thread blocks | DRAM throughput (% of peak) | warps active (%) | L2 hit (%) |", "|---|---|---|---|---|"]
for c, label in (("A", "A: one adapter"), ("B", "B: two adapters, contiguous"), ("S", "S: four adapters, sorted"),
                 ("C", "C: two adapters, 16-token blocks"), ("D", "D: four adapters, token-by-token interleaved")):
    n = ncu[c]
    tables.append(f"| {label} | {n['thread_blocks']} | {n['dram_pct']:.1f} | {n['warps_active_pct']:.2f} | {n['l2_hit_pct']:.1f} |")
tables += [""]

# CASH dispatch cost
cash = [r for r in read_csv(RES / "cash_dispatch/a6000_single/overhead_scaling.csv")
        if r["method"] == "cash_rank"]
ovh = {int(r["K"]): float(r["overhead_mean_ms"]) for r in cash}
claims["cash_tick_us"] = {k: round(v * 1e3, 1) for k, v in sorted(ovh.items())}
claims["cash_tick_r2_linear"] = float(cash[0]["r2_linear"])
tables += ["## CASH dispatch cost", "",
           "CASH ranking per scheduler tick (CPU): "
           + ", ".join(f"K={k}: {v * 1e3:.0f} us" for k, v in sorted(ovh.items()))
           + f"; linear fit against K, R^2 = {claims['cash_tick_r2_linear']:.4f}.", ""]

# C1 / Table II (K=10, saturated)
cells = rows("c1_k10/cells")
base = rows("c1_k10/baselines")
as_ = {s: cells[f"r32_mt128_seqs256_C7_s{s}_n400"]["tput"] for s in SEEDS}
v063 = {s: cells[f"r32_mt128_seqs256_C0_s{s}_n400"]["tput"] for s in SEEDS}
v050 = {s: base[f"v050_s{s}"]["tput"] for s in SEEDS}
slora = {s: base[f"slora_s{s}"]["tput"] for s in SEEDS}
punica = {s: rows("c1_k10/punica")[f"punica_s{s}"]["tput"] for s in SEEDS}
dlora = {s: rows("c1_k10/dlora")[f"dlora_s{s}"]["tput"] for s in SEEDS}
claims["C1_vs_vllm050"] = ratio_block(as_, v050)
claims["C1_vs_vllm063"] = ratio_block(as_, v063)
claims["TableII_vs_slora"] = ratio_block(as_, slora)
claims["TableII_vs_punica"] = ratio_block(as_, punica)
claims["TableII_vs_dlora"] = ratio_block(as_, dlora)
t = ["## C1 and Table II: LLaMA-7B, K=10, rank 32, saturated arrivals, 400 requests", "",
     "| System | s42 | s43 | s44 | mean tok/s | AdapterSlots speedup |", "|---|---|---|---|---|---|"]
for name, d, key in (("AdapterSlots", as_, None), ("vLLM 0.5.0", v050, "C1_vs_vllm050"),
                     ("vLLM 0.6.3", v063, "C1_vs_vllm063"), ("S-LoRA", slora, "TableII_vs_slora"),
                     ("Punica", punica, "TableII_vs_punica"), ("dLoRA", dlora, "TableII_vs_dlora")):
    g = f"{claims[key]['mean']:.3f}x (CV {claims[key]['cv'] * 100:.1f}%)" if key else ""
    t.append(f"| {name} | " + " | ".join(f"{d[s]:.1f}" for s in SEEDS) + f" | {mean(list(d.values())):.1f} | {g} |")
tables += t + [""]

# C1 at lambda = 7, 384-token outputs
lam = rows("c1_lam7_mt384")
la = {s: lam[f"as_s{s}"]["tput"] for s in SEEDS}
lv63 = {s: lam[f"v063_s{s}"]["tput"] for s in SEEDS}
lv50 = {s: lam[f"v050_s{s}"]["tput"] for s in (42, 43)}
claims["C1_lambda7_vs_vllm050"] = ratio_block(la, lv50)
claims["C1_lambda7_vs_vllm063"] = ratio_block(la, lv63)
tables += ["## C1 at lambda = 7 req/s: 384-token outputs, 1000 requests", "",
           "| System | s42 | s43 | s44 | AdapterSlots speedup |", "|---|---|---|---|---|",
           "| AdapterSlots | " + " | ".join(f"{la[s]:.1f}" for s in SEEDS) + " | |",
           "| vLLM 0.5.0 | " + " | ".join(f"{lv50[s]:.1f}" if s in lv50 else "" for s in SEEDS)
           + f" | {claims['C1_lambda7_vs_vllm050']['mean']:.3f}x |",
           "| vLLM 0.6.3 | " + " | ".join(f"{lv63[s]:.1f}" for s in SEEDS)
           + f" | {claims['C1_lambda7_vs_vllm063']['mean']:.3f}x |", ""]

# Table I lattice
lat = rows("table1_lattice")
stacks = {1: "Buffer, threshold T_max", 2: "Erlang T*(k)", 3: "PI-controlled T_max", 4: "CASH order",
          5: "CASH order + PredictiveLFU", 6: "WGKP holdback, vLLM LoRA kernels",
          7: "Full: WGKP holdback, grouped kernels"}
claims["TableI_lattice"] = {}
t = ["## Table I: LLaMA-7B, K=10, rank 32, saturated, 3 seeds", "",
     "| Mode | Stack | mean tok/s | gain vs vLLM 0.5.0 | gain vs vLLM 0.6.3 | WAR | GWAR(8) |",
     "|---|---|---|---|---|---|---|"]
for m in range(1, 8):
    d = {s: lat[f"asC{m}_s{s}"]["tput"] for s in SEEDS}
    g50, g63 = ratio_block(d, v050), ratio_block(d, v063)
    war = mean([lat[f"asC{m}_s{s}"]["war"] for s in SEEDS])
    g8 = mean([lat[f"asC{m}_s{s}"]["gwar8"] for s in SEEDS])
    claims["TableI_lattice"][f"C{m}"] = {"stack": stacks[m], "tput_mean": round(mean(list(d.values())), 1),
                                         "gain_vs_vllm050": g50["mean"], "gain_vs_vllm063": g63["mean"],
                                         "war": round(war, 3), "gwar8": round(g8, 3)}
    t.append(f"| C{m} | {stacks[m]} | {mean(list(d.values())):.1f} | {g50['mean']:.3f}x | {g63['mean']:.3f}x "
             f"| {war:.2f} | {g8:.2f} |")
tables += t + [""]

# Table I as a cumulative ablation
label = {"L1_wgkp": "WGKP scheduling, vLLM LoRA kernels", "L2_kvadm": "+ KV-aware admission",
         "L3_grouped": "+ grouped LoRA kernels (WGKP L2)", "L4_multislice": "+ fused QKV and gate_up projections",
         "L5_prefill": "+ grouped prefill", "L6_full": "+ adapter-aligned admission (full)"}
for model, base_tput, steps, cfg in (
        ("13b", rows("table1_ablation_13b/L0_vllm")["r64_mt64_seqs256_C0_s42_n400"]["tput"],
         ("L1_wgkp", "L2_kvadm", "L3_grouped", "L4_multislice", "L5_prefill", "L6_full"),
         "LLaMA-13B, one A6000, rank 64, K=16, 64-token outputs"),
        ("7b", v063[42], ("L1_wgkp", "L3_grouped", "L4_multislice", "L5_prefill", "L6_full"),
         "LLaMA-7B, rank 32, K=10, 128-token outputs")):
    t = [f"## Table I, mechanism by mechanism: {cfg}, saturated, seed 42", "",
         "| Step | tok/s | gain vs vLLM 0.6.3 | GWAR(8) |", "|---|---|---|---|",
         f"| vLLM 0.6.3 | {base_tput:.1f} | 1.000x | |"]
    claims[f"TableI_ablation_{model}"] = {"vllm063": round(base_tput, 1)}
    for s in steps:
        r = list(rows(f"table1_ablation_{model}/{s}").values())[0]
        claims[f"TableI_ablation_{model}"][s] = {"tput": round(r["tput"], 1), "gain": round(r["tput"] / base_tput, 4),
                                                 "gwar8": round(r["gwar8"], 3) if "gwar8" in r else None}
        g8 = f"{r['gwar8']:.2f}" if "gwar8" in r else ""
        t.append(f"| {label[s]} | {r['tput']:.1f} | {r['tput'] / base_tput:.3f}x | {g8} |")
    tables += t + [""]

# C2 APIS
c2 = rows("c2_apis")
by = {arm: {int(k[5:]): v[arm] for k, v in c2.items() if arm in v} for arm in ("apis", "tp2", "topo_rr25")}
claims["C2_apis_vs_tp2"] = ratio_block(by["apis"], by["tp2"])
claims["C2_apis_vs_topology_matched"] = ratio_block(by["apis"], by["topo_rr25"])
claims["C2_topology_alone"] = ratio_block(by["topo_rr25"], by["tp2"])
t = ["## C2 APIS: LLaMA-7B, K=50, rank 128, two A6000 over PCIe, saturated, 1000 requests", "",
     "| Arm | s42 | s43 | s44 |", "|---|---|---|---|"]
for arm, name in (("apis", "APIS (two TP=1 AdapterSlots shards, rate-sorted round-robin)"), ("tp2", "vLLM TP=2"),
                  ("topo_rr25", "two TP=1 vLLM, round-robin, 25 LoRA slots each")):
    t.append(f"| {name} | " + " | ".join(f"{by[arm][s]:.1f}" for s in SEEDS) + " |")
t += ["", "| Ratio | mean | min | max |", "|---|---|---|---|"]
for key, name in (("C2_apis_vs_tp2", "APIS / vLLM TP=2"), ("C2_apis_vs_topology_matched", "APIS / topology-matched vLLM"),
                  ("C2_topology_alone", "topology alone (TP=1 pair / TP=2)")):
    c = claims[key]
    t.append(f"| {name} | {c['mean']:.3f}x | {c['min']:.3f}x | {c['max']:.3f}x |")
tables += t + [""]

# C3
claims["C3"] = {}
t = ["## C3: GWAR(8) against throughput over the alignment-deadline sweep", "",
     "LLaMA-13B, one A6000, rank 64, K=16, 64-token outputs, 2000 requests, two runs per point.", "",
     "| Seed | deadline (s) | tok/s | GWAR(8) |", "|---|---|---|---|"]
for s in (42, 43):
    runs = [rows(f"c3_deadline/s{s}_run{i}") for i in (0, 1)]
    keys = list(runs[0])
    tp = [mean([r[k]["tput"] for r in runs]) for k in keys]
    g8 = [mean([r[k]["gwar8"] for r in runs]) for k in keys]
    claims["C3"][f"s{s}"] = {"deadline_s": [runs[0][k]["deadline_s"] for k in keys],
                             "tput": [round(x, 1) for x in tp], "gwar8": [round(x, 3) for x in g8],
                             "pearson_r": round(pearson(g8, tp), 4)}
    for k, a, b in zip(keys, tp, g8):
        t.append(f"| {s} | {runs[0][k]['deadline_s']:g} | {a:.1f} | {b:.3f} |")
claims["C3"]["pearson_r_mean"] = round(mean([claims["C3"][f"s{s}"]["pearson_r"] for s in (42, 43)]), 4)
t += ["", f"Pearson r (n = 6): seed 42 {claims['C3']['s42']['pearson_r']:.3f}, seed 43 "
      f"{claims['C3']['s43']['pearson_r']:.3f}."]
tables += t + [""]

# Table III
k4 = rows("table3_k4")
sync = [k4[f"k4_sync_as_s{s}"] for s in SEEDS]
v = k4["k4_engine_vllm_s42"]
claims["TableIII"] = {"tput": [round(r["tput"], 1) for r in sync],
                     "engine_ttft_p50_ms": [round(r["eng_ttft_p50"], 1) for r in sync],
                     "engine_ttft_p99_ms": [round(r["eng_ttft_p99"], 1) for r in sync],
                     "client_ttft_p50_ms": [round(r["ttft_p50"], 1) for r in sync],
                     "client_ttft_p99_ms": [round(r["ttft_p99"], 1) for r in sync],
                     "vllm_s42": {"tput": round(v["tput"], 1), "client_ttft_p50_ms": round(v["ttft_p50"], 1),
                                  "client_ttft_p99_ms": round(v["ttft_p99"], 1)}}
tables += ["## Table III: LLaMA-7B, K=4, rank 16, lambda = 7 req/s, 500 requests", "",
           "| Run | tok/s | engine TTFT p50 / p99 (ms) | client TTFT p50 / p99 (ms) |", "|---|---|---|---|"]
for s, r in zip(SEEDS, sync):
    tables.append(f"| AdapterSlots s{s} | {r['tput']:.1f} | {r['eng_ttft_p50']:.1f} / {r['eng_ttft_p99']:.1f} "
                  f"| {r['ttft_p50']:.1f} / {r['ttft_p99']:.1f} |")
tables += [f"| vLLM 0.6.3 s42 | {v['tput']:.1f} | | {v['ttft_p50']:.1f} / {v['ttft_p99']:.1f} |", ""]

# Table V
t5 = {}
for k in (4, 8, 16):
    for slots, tag in ((k, "slots_K"), (16, "slots_16")):
        a = rows(f"table5_13b/k{k}_a_ml{slots}")
        b = rows(f"table5_13b/k{k}_v_ml{slots}")
        an = {s: a[f"r64_mt64_seqs256_C7_s{s}_n2000"]["tput"] for s in SEEDS}
        bn = {s: b[f"r64_mt64_seqs256_C0_s{s}_n2000"]["tput"] for s in SEEDS}
        t5.setdefault(tag, {})[f"K{k}"] = dict(ratio_block(an, bn), as_tput=round(mean(list(an.values())), 1),
                                                vllm_tput=round(mean(list(bn.values())), 1))
claims["TableV_13b"] = t5
t = ["## Table V: LLaMA-13B on one A6000, rank 64, 64-token outputs, saturated, 2000 requests, 3 seeds", "",
     "| K | LoRA slots (both systems) | vLLM tok/s | AdapterSlots tok/s | speedup |", "|---|---|---|---|---|"]
for tag, name in (("slots_K", "K"), ("slots_16", "16")):
    for k in (4, 8, 16):
        if tag == "slots_16" and k == 16:
            continue
        c = t5[tag][f"K{k}"]
        t.append(f"| {k} | {name} | {c['vllm_tput']:.1f} | {c['as_tput']:.1f} | {c['mean']:.3f}x |")
tables += t + [""]

tp2 = rows("table5_13b_tp2")
blk = {}
for k in (4, 8, 16):
    seeds = SEEDS if k == 4 else (42,)
    suf = {42: ""} | {s: f"_s{s}" for s in (43, 44)}
    an = {s: tp2[f"r32_sat_K{k}_as{suf[s]}"]["tput"] for s in seeds}
    bn = {s: tp2[f"r32_sat_K{k}_vllm{suf[s]}"]["tput"] for s in seeds}
    blk[f"K{k}"] = dict(ratio_block(an, bn), as_tput=round(mean(list(an.values())), 1),
                        vllm_tput=round(mean(list(bn.values())), 1))
claims["TableV_13b_tp2"] = blk
t = ["## Table V setting with TP=2: LLaMA-13B on two A6000s, rank 32, 128-token outputs, 400 requests", "",
     "| K | seeds | vLLM tok/s | AdapterSlots tok/s | speedup |", "|---|---|---|---|---|"]
for k in (4, 8, 16):
    c = blk[f"K{k}"]
    t.append(f"| {k} | {', '.join(c['per_seed'])} | {c['vllm_tput']:.1f} | {c['as_tput']:.1f} | {c['mean']:.3f}x |")
tables += t + [""]

# kernels
kl = json.load(open(RUNS / "kernel/kernel_layer_r32_13b_mode3.json"))
row = [r for r in kl["rows"] if r["K"] == 16 and r["B"] == 100][0]
fused = json.load(open(RUNS / "kernel/fused_crossover/a6000_tp1_r16.json"))
fused32 = json.load(open(RUNS / "kernel/fused_crossover/a6000_tp1_r32.json"))
oc = rows("kernel")["output_check_mode3"]
l3c = rows("kernel")["l3_check"]
claims["kernel"] = {
    "lora_ms_per_13b_decode_step_K16_B100_r32": {"adapterslots": round(row["as_us"] * 40 / 1e3, 2),
                                                "vllm_bgmv_sorted": round(row["stock_sorted_us"] * 40 / 1e3, 2),
                                                "vllm_bgmv_mixed": round(row["stock_mixed_us"] * 40 / 1e3, 2)},
    "psi_fuse_n32_r16": fused["crossover_curve"]["32"]["psi_fuse"],
    "psi_promote_n8_r32": fused32["crossover_curve"]["8"]["psi_gemm"],
    "output_check": oc,
    "l3_check": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in l3c.items()}}
tables += ["## Kernels", "",
           f"LoRA time per LLaMA-13B decode step (K=16, batch 100, rank 32, 40 layers): grouped kernels "
           f"{row['as_us'] * 40 / 1e3:.2f} ms; vLLM's BGMV path {row['stock_sorted_us'] * 40 / 1e3:.2f} ms on the "
           f"same adapter-sorted batch and {row['stock_mixed_us'] * 40 / 1e3:.2f} ms in arrival order.", "",
           f"psi_fuse (fused Triton over SGMV) at n = 32, rank 16: {fused['crossover_curve']['32']['psi_fuse']:.3f}; "
           f"psi_promote (merged-weight GEMM over SGMV) at n = 8, rank 32: "
           f"{fused32['crossover_curve']['8']['psi_gemm']:.3f}.", "",
           f"Greedy outputs against vLLM (160 prompts, 48 tokens): {oc['exact']}/{oc['n']} identical, "
           f"mean shared prefix {oc['mean_prefix_frac']:.3f}.", "",
           f"L3 served end to end (LLaMA-7B, one adapter, eager, 75% memory for vLLM): "
           f"{l3c['tput_l3']:.1f} tok/s with L3 against {l3c['tput_no_l3']:.1f} without "
           f"({l3c['speedup']:.3f}x), at least {l3c['promoted_steps_min']} promoted steps, {l3c['identical']}/{l3c['n']} "
           f"identical greedy completions (mean shared prefix {l3c['mean_shared_prefix']:.3f}).", ""]

OUT.mkdir(exist_ok=True)
json.dump(claims, open(OUT / "claims.json", "w"), indent=1)
open(OUT / "tables.md", "w").write(
    "# Results\n\nGenerated by `benchmarks/paper/summarize.py` from `results/`. Do not edit by hand.\n\n"
    + "\n".join(tables))
print(f"wrote {OUT / 'claims.json'} and {OUT / 'tables.md'}")
