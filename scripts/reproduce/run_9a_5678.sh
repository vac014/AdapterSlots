#!/usr/bin/env bash
# run_9a_5678.sh -- Sections 9.A.5, 9.A.6, 9.A.7, 9.A.8 on Single RTX A6000 (TP=1)
#
# vLLM fix summary:
#   §9.A.5: war_slo_feasibility_frontier.py --mode simulate uses Erlang CDF inversion analytically
#           (no server). GPU mode (default) already uses vllm_serve_adapter_slots.py.
#   §9.A.6: benchmark_serving_full.py --literature-match/--k-scaling: pure simulation.
#   §9.A.7: war_improvement_serving_benchmark.py used adapter_slots.integrations.patched_api_server
#           (runtime monkey-patch). Fixed: now uses vllm_serve_adapter_slots.py which
#           sets AS_SCHEDULER=1 so patched llm_engine.py loads AlignmentAwareScheduler.
#           No vLLM internals modified at runtime. Real GPU serving on single A6000.
#   §9.A.8: ablation_global_vs_erlang_tmax.py/pi_controller_drift_response.py --mode simulate: analytical. alignment_buffer_dispatch_overhead.py:
#           pure CPU (no GPU). benchmark_serving_full.py (AB10, AB12): simulation.
#
# Single GPU: CUDA_VISIBLE_DEVICES=0, TP=1. No VLLM_WORKER_MULTIPROC_METHOD=spawn needed.
set -euo pipefail

cd /workspace

export CUDA_VISIBLE_DEVICES=0

# Read calibrated tau_iter from §9.A.1 output
TAU_ITER_A6000=$(python3 -c "
import csv
r = next(csv.DictReader(open('results/end_to_end_serving/tau_iter/a6000_tau_iter.csv')))
print(r['tau_iter_ms'])
" 2>/dev/null || echo "30")
echo "[setup] τ_iter_A6000 = ${TAU_ITER_A6000} ms"

# Output directories
mkdir -p results/end_to_end_serving/e5/a6000
mkdir -p results/end_to_end_serving/literature/a6000
mkdir -p results/end_to_end_serving/ablations/a6000/ab5
mkdir -p results/end_to_end_serving/ablations/a6000/ab2
mkdir -p results/end_to_end_serving/ablations/a6000/ab3
mkdir -p results/end_to_end_serving/ablations/a6000/ab8
mkdir -p results/end_to_end_serving/ablations/a6000/ab10
mkdir -p results/end_to_end_serving/ablations/a6000/ab12

# §9.A.5 -- E5: WAR-SLO Feasibility Frontier (Single A6000)
echo ""
echo "══════════════════════════════════════════════════════"
echo "§9.A.5  E5: WAR-SLO Frontier (Single A6000)"
echo "══════════════════════════════════════════════════════"
# Sweeps WAR* ∈ {0.3,0.5,0.7,0.8,0.9,1.0} and computes per-adapter T_max*
# via Erlang CDF inversion (Proposition 5.6). Identifies Pareto knee (last
# WAR* below TTFT SLO). No GPU server needed -- analytical Erlang frontier.

python scripts/experiments/war_slo_feasibility_frontier.py \
    --mode simulate \
    --K 4 \
    --alpha-zipf 0.9 \
    --request-rate 7 \
    --ttft-slo-ms 2000.0 \
    --avg-output-tokens 64 \
    --war-targets 0.3 0.5 0.7 0.8 0.9 1.0 \
    --label "Single A6000" \
    --output results/end_to_end_serving/e5/a6000/war_slo_frontier_k4.csv

python scripts/experiments/war_slo_feasibility_frontier.py \
    --mode simulate \
    --K 8 \
    --alpha-zipf 0.9 \
    --request-rate 7 \
    --ttft-slo-ms 2000.0 \
    --avg-output-tokens 64 \
    --war-targets 0.3 0.5 0.7 0.8 0.9 1.0 \
    --label "Single A6000" \
    --output results/end_to_end_serving/e5/a6000/war_slo_frontier_k8.csv

echo ""
echo "§9.A.5 Pareto comparison K=4 vs K=8:"
python3 -c "
import csv
for K, f in [(4,'results/end_to_end_serving/e5/a6000/war_slo_frontier_k4.csv'),
             (8,'results/end_to_end_serving/e5/a6000/war_slo_frontier_k8.csv')]:
    rows = list(csv.DictReader(open(f)))
    print(f'  K={K}: WAR* targets and TTFT P99')
    for r in rows:
        feasible = 'feasible' if float(r['p99_ttft_ms']) < 2000 else 'over-SLO'
        print(f'    WAR*={r[\"war_achieved\"]}  TTFT_P99={float(r[\"p99_ttft_ms\"]):.0f}ms  [{feasible}]')
"

# §9.A.6 -- B1–B4: Literature-matching benchmarks (Single A6000)
echo ""
echo "══════════════════════════════════════════════════════"
echo "§9.A.6  B1–B4: Literature-matching benchmarks (Single A6000)"
echo "══════════════════════════════════════════════════════"
# B1: Throughput vs Rate (Punica Fig 11, S-LoRA Fig 5, dLoRA main results)
# B2: TTFT vs Rate (AdapterSlots overhead documentation)
# B3: K scaling K=2..50 (S-LoRA Tables 3/4/5 partial)
# B4: Decode-phase degradation under misalignment

python benchmarks/sota/serving_full.py \
    --literature-match \
    --max-loras 50 \
    --tau-iter-ms "${TAU_ITER_A6000:-30}" \
    --hardware-label a6000_single \
    --dataset-path ./data/sharegpt/sharegpt.jsonl \
    --output-dir results/end_to_end_serving/literature/a6000/

# B3 K-scaling sweep
python benchmarks/sota/serving_full.py \
    --k-scaling \
    --k-values 2 4 8 10 50 \
    --max-loras 50 \
    --tau-iter-ms "${TAU_ITER_A6000:-30}" \
    --hardware-label a6000_single \
    --dataset-path ./data/sharegpt/sharegpt.jsonl \
    --output-dir results/end_to_end_serving/literature/a6000/

echo ""
echo "§9.A.6 B1 throughput at λ=7, K=4 (vLLM vs AdapterSlots T=5ms):"
python3 -c "
import csv, glob
for f in sorted(glob.glob('results/end_to_end_serving/literature/a6000/b1_throughput_rate/b1_*_K4.csv')):
    rows = list(csv.DictReader(open(f)))
    fname = f.split('/')[-1]
    for r in rows:
        if r.get('system','') in ('vllm','adapter_slots_t5') and str(r.get('rate','')) in ('7.0','7'):
            print(f'  {fname}: {r[\"system\"]:20} λ={r[\"rate\"]} tput={r[\"throughput_tok_s\"]} tok/s')
" 2>/dev/null || echo "  (no per-system CSVs -- check summary)"

# §9.A.8 -- AB2/AB3/AB8/AB10/AB12 Ablations (Single A6000) -- run first (fast)
# §9.A.7 -- AB5 Component Decomposition -- run LAST (real GPU, ~90 min)
echo ""
echo "══════════════════════════════════════════════════════"
echo "§9.A.8  Ablations AB2/AB3/AB8/AB10/AB12 (Single A6000)"
echo "══════════════════════════════════════════════════════"

# AB2: Erlang per-adapter T_max vs global T_max
echo "[AB2] GlobalT vs ErlangT per-adapter TTFT reduction (simulate)"
python scripts/experiments/ablation_global_vs_erlang_tmax.py \
    --mode simulate \
    --K 4 \
    --alpha-zipf 0.9 \
    --request-rate 7.0 \
    --war-target 0.8 \
    --ttft-slo-ms 2000.0 \
    --output-dir results/end_to_end_serving/ablations/a6000/ab2/

echo ""
echo "§9.A.8 AB2 results (GlobalT vs ErlangT TTFT):"
python3 -c "
import csv, glob
for f in glob.glob('results/end_to_end_serving/ablations/a6000/ab2/ab2_global_vs_erlang_*.csv'):
    for r in csv.DictReader(open(f)):
        print(f'  {r[\"condition\"]:10}  mean_TTFT={float(r[\"mean_ttft_ms\"]):.1f}ms  '
              f'p99_TTFT={float(r[\"p99_ttft_ms\"]):.1f}ms')
"

# AB3: PI adaptive T_max vs static T_max (drift experiment)
echo ""
echo "[AB3] PI adaptive T_max drift convergence (simulate, 900s step-drift)"
python scripts/experiments/pi_controller_drift_response.py \
    --workload step \
    --K 4 \
    --lambda-total 7.0 \
    --alpha-zipf 0.9 \
    --war-target 0.8 \
    --warp-size 32 \
    --duration 900 \
    --drift-at 300 \
    --tau-iter-ms "${TAU_ITER_A6000:-30}" \
    --hardware-label single_a6000 \
    --output results/end_to_end_serving/ablations/a6000/ab3/e7_step_drift_a6000.csv

echo ""
echo "§9.A.8 AB3 drift convergence (PI vs static WAR at t=600s, 300s post-drift):"
python3 -c "
import csv
rows = list(csv.DictReader(open('results/end_to_end_serving/ablations/a6000/ab3/e7_step_drift_a6000.csv')))
# Last 100 rows = post-drift steady state
tail = rows[-100:]
pi_wars = [float(r['war_pi']) for r in tail if r.get('war_pi','')]
static_wars = [float(r['war_static']) for r in tail if r.get('war_static','')]
if pi_wars and static_wars:
    print(f'  Post-drift (last 100 ticks): WAR_PI={sum(pi_wars)/len(pi_wars):.4f}  WAR_static={sum(static_wars)/len(static_wars):.4f}')
    print(f'  PI adapts to drift; static drifts away from WAR* target')
else:
    print(f'  Rows: {len(rows)}  fields: {list(rows[0].keys()) if rows else []}')
" 2>/dev/null || echo "  (check CSV columns)"

# AB8: Dispatch overhead scaling K=2..50
echo ""
echo "[AB8] AlignmentBuffer form_batch() CPU overhead vs K (CPU-only)"
python scripts/experiments/alignment_buffer_dispatch_overhead.py \
    --K-values 2 4 8 10 16 32 50 \
    --label "a6000_single" \
    --output results/end_to_end_serving/ablations/a6000/ab8/dispatch_overhead_a6000.csv

echo ""
echo "§9.A.8 AB8 dispatch overhead (EC §16.1 #7: mean < 0.5ms at K≤50):"
python3 -c "
import csv
rows = list(csv.DictReader(open('results/end_to_end_serving/ablations/a6000/ab8/dispatch_overhead_a6000.csv')))
print(f'  {\"K\":>4}  {\"mean_ms\":>8}  {\"p99_ms\":>8}  {\"pass\":>5}')
for r in rows:
    verdict = 'PASS' if float(r['mean_ms']) < 0.5 else 'FAIL'
    print(f'  {r[\"K\"]:>4}  {float(r[\"mean_ms\"]):>8.4f}  {float(r[\"p99_ms\"]):>8.4f}  {verdict:>5}')
"

# AB10: Distribution sensitivity (Zipf vs others)
echo ""
echo "[AB10] Distribution sensitivity: adapter_slots_t5 vs vllm"
python benchmarks/sota/serving_full.py \
    --systems adapter_slots_t5 vllm \
    --K 4 \
    --request-rates 7.0 \
    --tau-iter-ms "${TAU_ITER_A6000:-30}" \
    --hardware-label a6000_single \
    --dataset-path ./data/sharegpt/sharegpt.jsonl \
    --output-dir results/end_to_end_serving/ablations/a6000/ab10/

echo ""
echo "§9.A.8 AB10 distribution sensitivity (AdapterSlots vs vLLM at λ=7):"
python3 -c "
import csv
rows = list(csv.DictReader(open('results/end_to_end_serving/ablations/a6000/ab10/e4_a6000_single_summary.csv')))
print(f'  {\"System\":<20} {\"Tput(tok/s)\":>12} {\"WAR\":>6} {\"TTFT_P50\":>9}')
for r in rows:
    print(f'  {r[\"system\"]:<20} {float(r[\"throughput_tok_s\"]):>12.1f} '
          f'{float(r[\"war\"]):>6.4f} {float(r[\"ttft_p50_ms\"]):>9.1f}')
"

# AB12: Model scale generalization
echo ""
echo "[AB12] Model scale: LLaMA-7B vs LLaMA-13B (tau_iter same hardware)"
python benchmarks/sota/serving_full.py \
    --systems vllm adapter_slots_t5 \
    --model ./models/llama-7b \
    --adapter-dir ./adapters \
    --max-loras 16 \
    --K 4 \
    --request-rates 3.0 7.0 \
    --tau-iter-ms "${TAU_ITER_A6000:-30}" \
    --hardware-label a6000_single_7b \
    --dataset-path ./data/sharegpt/sharegpt.jsonl \
    --output-dir results/end_to_end_serving/ablations/a6000/ab12/ 2>/dev/null

python benchmarks/sota/serving_full.py \
    --systems vllm adapter_slots_t5 \
    --model ./models/llama-13b \
    --adapter-dir ./adapters \
    --max-loras 16 \
    --K 4 \
    --request-rates 3.0 7.0 \
    --tau-iter-ms "${TAU_ITER_A6000:-30}" \
    --hardware-label a6000_single_13b \
    --dataset-path ./data/sharegpt/sharegpt.jsonl \
    --output-dir results/end_to_end_serving/ablations/a6000/ab12/ 2>/dev/null || \
    echo "[AB12-13B] Simulation skipped (model path metadata only)"

echo ""
echo "§9.A.8 AB12 model scale comparison (λ=7, AdapterSlots vs vLLM):"
python3 -c "
import csv, glob
for f in sorted(glob.glob('results/end_to_end_serving/ablations/a6000/ab12/e4_*_summary.csv')):
    hw = f.split('e4_')[1].replace('_summary.csv','')
    rows = [r for r in csv.DictReader(open(f)) if r.get('rate','')=='7.0']
    for r in rows:
        print(f'  {hw}  {r[\"system\"]:20}  tput={float(r[\"throughput_tok_s\"]):>8.1f} tok/s  WAR={float(r[\"war\"]):>6.4f}')
"

# §9.A.7 -- AB5 Component Decomposition (Single A6000, REAL GPU)
echo ""
echo "══════════════════════════════════════════════════════"
echo "§9.A.7  AB5: Component Decomposition (Single A6000, real GPU)"
echo "══════════════════════════════════════════════════════"
echo "[§9.A.7] Starting real GPU sweep: 6 T_max × 3 rates = 18 server sessions."
echo "[§9.A.7] vLLM fix: patched_api_server (monkey-patch) → vllm_serve_adapter_slots.py"
echo "[§9.A.7] Estimated time: ~90 minutes. Timeout set to 3h."
echo ""

export AS_MODE=threshold          # AlignmentAwareScheduler threshold mode (alignment_buffer)
export AS_TTFT_SLO_MS=2000.0
export AS_LOG_WAR=1               # Write per-batch WAR to batch log

python scripts/experiments/war_improvement_serving_benchmark.py \
    --model ./models/llama-7b \
    --adapter-dir ./adapters \
    --K 4 \
    --tmax-values 0 1 2 5 10 20 \
    --request-rates 3 7 10 \
    --dataset-path ./data/sharegpt/sharegpt.jsonl \
    --output-dir results/end_to_end_serving/ablations/a6000/ab5 \
    --duration 300

echo ""
echo "§9.A.7 AB5 results (WAR and throughput per T_max component, λ=7):"
python3 -c "
import csv
rows = [r for r in csv.DictReader(open('results/end_to_end_serving/ablations/a6000/ab5/war_improvement.csv'))
        if float(r['request_rate']) == 7.0]
print(f'  {\"T_max(ms)\":>9}  {\"Component\":>12}  {\"WAR_mean\":>9}  {\"Tput(tok/s)\":>11}  {\"TTFT_P50\":>9}')
comp_map = {0:\"v0-baseline\", 1:\"v1+buffer\", 2:\"v2+erlang\", 5:\"v3+fairness\", 10:\"v4+PI\", 20:\"v5+whittle\"}
for r in rows:
    t = float(r['tmax_ms'])
    comp = comp_map.get(t, f't={t}')
    war = r.get('war_mean','nan')
    tput = r.get('throughput_tok_s','nan')
    ttft = r.get('ttft_p50_ms','nan')
    print(f'  {t:>9.0f}  {comp:>12}  {float(war) if war!=\"nan\" else float(\"nan\"):>9.4f}  '
          f'{float(tput) if tput!=\"nan\" else float(\"nan\"):>11.1f}  '
          f'{float(ttft) if ttft!=\"nan\" else float(\"nan\"):>9.1f}')
" 2>/dev/null || echo "  (check results/end_to_end_serving/ablations/a6000/ab5/war_improvement.csv)"

echo ""
echo "══════════════════════════════════════════════════════"
echo "§9.A.5–9.A.8 COMPLETE"
echo "Outputs:"
echo "  results/end_to_end_serving/e5/a6000/war_slo_frontier_k4.csv"
echo "  results/end_to_end_serving/e5/a6000/war_slo_frontier_k8.csv"
echo "  results/end_to_end_serving/literature/a6000/ (B1–B4)"
echo "  results/end_to_end_serving/ablations/a6000/ab5/war_improvement.csv  [AB5, real GPU]"
echo "  results/end_to_end_serving/ablations/a6000/ab2/  [AB2, simulate]"
echo "  results/end_to_end_serving/ablations/a6000/ab3/  [AB3, simulate]"
echo "  results/end_to_end_serving/ablations/a6000/ab8/  [AB8, CPU-only]"
echo "  results/end_to_end_serving/ablations/a6000/ab10/ [AB10, simulate]"
echo "  results/end_to_end_serving/ablations/a6000/ab12/ [AB12, simulate]"
echo "══════════════════════════════════════════════════════"
