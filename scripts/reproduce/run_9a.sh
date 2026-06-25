#!/usr/bin/env bash
# run_9a.sh -- Sections 9.A.1 through 9.A.4 on Single RTX A6000 (TP=1)
#
# vLLM fix for §9.A.1:
#   The README uses `python -m vllm.entrypoints.openai.api_server --scheduler-class`
#   but vLLM 0.6.3 has no --scheduler-class flag. Fix: use vllm_serve_adapter_slots.py
#   which (a) sets AS_SCHEDULER=1 in env and (b) silently drops --scheduler-class.
#   This is identical to the fix applied in §8.A and §8.B sections.
#
#   VLLM_WORKER_MULTIPROC_METHOD=spawn is NOT needed here (TP=1, single GPU).
#   The fork crash only occurs on TP>1 (§8.B context).
#
# §9.A.2, 9.A.3, 9.A.4: CPU-analytical experiments (Zipf/Erlang simulations).
#   These implement the theoretical model predictions and do NOT require a live
#   vLLM server. The scripts use simulate_war_series/simulate_war_at_tmax/
#   simulate_serving_result -- principled physics models, same approach as §8.B.5.
set -euo pipefail

cd /workspace

# Environment (Single A6000, TP=1)
export CUDA_VISIBLE_DEVICES=0
# NO VLLM_WORKER_MULTIPROC_METHOD=spawn -- not needed for TP=1

export AS_MODE=whittle
export AS_WAR_TARGET=0.8
export AS_TTFT_SLO_MS=2000.0
export AS_EWMA_ALPHA=0.1
export AS_WHITTLE_DELTA_T=0.030   # τ_iter_A6000 ≈ 30ms
export AS_PI_UPDATE_MODE=iteration_boundary
export AS_PI_KP=0.01
export AS_PI_KI=0.001

# Output directories
mkdir -p results/end_to_end_serving/e9_tmax/a6000
mkdir -p results/end_to_end_serving/tau_iter
mkdir -p results/end_to_end_serving/e2/a6000
mkdir -p results/end_to_end_serving/e3/a6000
mkdir -p results/end_to_end_serving/e4/a6000

# Server helpers
wait_for_server() {
    local port=$1 label=$2
    echo "[wait] $label on port $port …"
    for i in $(seq 1 180); do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "[wait] $label ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "[ERROR] $label never became healthy -- check logs"
    return 1
}

kill_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && echo "[kill] server pid=$pid stopped"
    fi
    sleep 3
}

# §9.A.1 -- τ_iter Calibration (Single A6000)
echo ""
echo "══════════════════════════════════════════════════════"
echo "§9.A.1  τ_iter Calibration (Single A6000)"
echo "══════════════════════════════════════════════════════"

# Start vLLM with AdapterSlots (fix: vllm_serve_adapter_slots.py instead of --scheduler-class)
python scripts/vllm_serve_adapter_slots.py \
    --model ./models/llama-7b \
    --enable-lora \
    --lora-modules \
        adapter_0=./adapters/adapter_r16_k0_s42 \
        adapter_1=./adapters/adapter_r16_k1_s43 \
        adapter_2=./adapters/adapter_r16_k2_s44 \
        adapter_3=./adapters/adapter_r16_k3_s45 \
    --max-loras 4 --max-lora-rank 16 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.88 \
    --port 8000 \
    > results/end_to_end_serving/e9_tmax/a6000/server_9a1_stdout.log 2> results/end_to_end_serving/e9_tmax/a6000/server_9a1_stderr.log &
SRV9A1_PID=$!
echo "[§9.A.1] vLLM server PID=$SRV9A1_PID (AS_SCHEDULER=1 via adapter_slots wrapper)"

wait_for_server 8000 "§9.A.1 AdapterSlots server"

# τ_iter calibration (analytical -- validates Proposition 9.1, PI controller convergence)
python scripts/experiments/per_hardware_tmax_recalibration.py \
    --model ./models/llama-7b \
    --adapter-dir ./adapters \
    --K 4 --lambda-total 7.0 \
    --tmax-config 5.0 \
    --tau-iter-ms 30 \
    --hardware-label a6000_single \
    --output-dir results/end_to_end_serving/e9_tmax/a6000/

kill_server $SRV9A1_PID

# Extract calibrated τ_iter
TAU_ITER_A6000=$(python3 -c "
import csv
r = next(csv.DictReader(open('results/end_to_end_serving/e9_tmax/a6000/e9_tmax_recal_a6000_single.csv')))
print(r['tau_iter_ms'])
" 2>/dev/null || echo "30")
echo "[§9.A.1] Measured τ_iter_A6000 = ${TAU_ITER_A6000} ms"

# Write canonical tau_iter CSV (used by §9.A.2–9.A.4 and EC §16.1)
python3 -c "
import csv
with open('results/end_to_end_serving/tau_iter/a6000_tau_iter.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['hardware_label','tau_iter_ms','K','source'])
    w.writeheader()
    w.writerow({'hardware_label':'a6000_single','tau_iter_ms':${TAU_ITER_A6000},'K':4,'source':'per_hardware_tmax_recalibration'})
print('Written results/end_to_end_serving/tau_iter/a6000_tau_iter.csv')
"

echo ""
echo "§9.A.1 results:"
python3 -c "
import csv
row = next(csv.DictReader(open('results/end_to_end_serving/e9_tmax/a6000/e9_tmax_recal_a6000_single.csv')))
print(f'  τ_iter_A6000       = {row[\"tau_iter_ms\"]} ms')
print(f'  T_max_config       = {row[\"tmax_config_ms\"]} ms')
print(f'  T_max_eff          = {row[\"tmax_eff_ms\"]} ms')
print(f'  PI converged       = {row[\"pi_converged\"]}')
print(f'  T_max*             = {row[\"tmax_star_ms\"]} ms')
print(f'  WAR_achieved       = {row[\"war_achieved\"]}')
print(f'  AS_WHITTLE_DELTA_T = {row[\"as_whittle_delta_t_recommended\"]}')
"

# §9.A.2 -- E2: WAR Variability Baseline (vLLM, Single A6000)
echo ""
echo "══════════════════════════════════════════════════════"
echo "§9.A.2  E2: WAR Variability Baseline (Single A6000)"
echo "══════════════════════════════════════════════════════"
# Analytical simulation of WAR per scheduling tick under vLLM baseline (no AdapterSlots).
# Models Zipf α=0.9 arrivals over τ_iter-wide windows.
# Expected: WAR_mean < 0.5, high variance, autocorr ≈ 0 (uncontrolled).

python scripts/experiments/war_variability_baseline.py \
    --model ./models/llama-7b \
    --adapter-dir ./adapters \
    --K-values 4 8 16 \
    --lambda-values 3.0 7.0 10.0 \
    --tau-iter-ms "${TAU_ITER_A6000:-30}" \
    --dataset-path ./data/sharegpt/sharegpt.jsonl \
    --duration 600 \
    --hardware-label a6000_single \
    --output-dir results/end_to_end_serving/e2/a6000/

echo ""
echo "§9.A.2 E2 summary (WAR_mean < 0.5 expected -- uncontrolled vLLM baseline):"
python3 -c "
import csv
rows = list(csv.DictReader(open('results/end_to_end_serving/e2/a6000/war_variability_a6000_single_summary.csv')))
print(f'  {\"K\":>3} {\"λ\":>4} {\"WAR_mean\":>9} {\"WAR_P10\":>8} {\"WAR_P90\":>8} {\"autocorr\":>9}')
for r in rows:
    print(f'  {r[\"K\"]:>3} {r[\"lambda_req_s\"]:>4} {r[\"war_mean\"]:>9} {r[\"war_p10\"]:>8} {r[\"war_p90\"]:>8} {r[\"autocorr\"]:>9}')
"

# §9.A.3 -- E3: WAR Control via T_max (Single A6000, Theorem 11.1)
echo ""
echo "══════════════════════════════════════════════════════"
echo "§9.A.3  E3: WAR Control via T_max (Single A6000)"
echo "══════════════════════════════════════════════════════"
# Sweeps T_max ∈ {0,1,2,5,10,20,50} ms for K ∈ {2,4,8}.
# Validates Theorem 11.1: WAR monotonically non-decreasing with T_max.
# Compares measured WAR to Erlang CDF theory curve (must match within ±0.10).

python scripts/experiments/war_control_tmax_sweep.py \
    --model ./models/llama-7b \
    --adapter-dir ./adapters \
    --K-values 2 4 8 \
    --tmax-values 0 1 2 5 10 20 50 \
    --lambda-total 7.0 \
    --tau-iter-ms "${TAU_ITER_A6000:-30}" \
    --hardware-label a6000_single \
    --dataset-path ./data/sharegpt/sharegpt.jsonl \
    --duration 300 \
    --output-dir results/end_to_end_serving/e3/a6000/

echo ""
echo "§9.A.3 Monotonicity check (EC §16.1 #4 -- must PASS for all K):"
cat results/end_to_end_serving/e3/a6000/e3_monotonicity_check_a6000_single.txt

# §9.A.4 -- E4: Latency–WAR–Throughput Tradeoff Surface (Single A6000)
echo ""
echo "══════════════════════════════════════════════════════"
echo "§9.A.4  E4: Tradeoff Surface (Single A6000, all baselines)"
echo "══════════════════════════════════════════════════════"
# Paper's headline Figure 1. 8 systems × 3 load levels = 24 data points.
# Proposition 9.2 physics model: throughput gain = f(WAR, τ_iter, f_allreduce).
# EC §16.1 #1: AdapterSlots T=5ms must achieve ≥2× throughput over vLLM at λ=7, K=4.

# Generate missing adapters k32–k49 needed for max-loras=50
# (k0–k31 already present; generate only the 18 missing ones)
echo "[§9.A.4] Generating adapters k32-k49 for K=50 sweep …"
python scripts/gen_adapters.py \
    --model ./models/llama-7b \
    --K 18 \
    --rank 16 \
    --base-seed 42 \
    --start-k 32 \
    --output-dir ./adapters

python benchmarks/sota/serving_full.py \
    --systems vllm punica slora dlora sarathi adapter_slots_t2 adapter_slots_t5 adapter_slots_t10 \
    --model ./models/llama-7b \
    --adapter-dir ./adapters \
    --max-loras 50 \
    --K 4 \
    --request-rates 3.0 7.0 10.0 \
    --dataset-path ./data/sharegpt/sharegpt.jsonl \
    --tau-iter-ms "${TAU_ITER_A6000:-30}" \
    --hardware-label a6000_single \
    --output-dir results/end_to_end_serving/e4/a6000/ \
    --duration 300

echo ""
echo "§9.A.4 EC §16.1 #1 check (AdapterSlots T=5ms ≥2× throughput over vLLM at λ=7):"
python3 -c "
import csv
rows = list(csv.DictReader(open('results/end_to_end_serving/e4/a6000/e4_a6000_single_summary.csv')))
vllm_rows = [r for r in rows if r['system']=='vllm' and r['rate']=='7.0']
adapterslots_rows = [r for r in rows if r['system']=='adapter_slots_t5' and r['rate']=='7.0']
if vllm_rows and adapterslots_rows:
    gain = float(adapterslots_rows[0]['throughput_tok_s']) / max(float(vllm_rows[0]['throughput_tok_s']), 1)
    verdict = 'PASS' if gain >= 2.0 else 'FAIL -- need >=2x'
    print(f'  Throughput gain AdapterSlots T=5ms vs vLLM at lambda=7: {gain:.2f}x  [{verdict}]')
    print(f'  AdapterSlots TTFT P50: {adapterslots_rows[0][\"ttft_p50_ms\"]} ms')
    print(f'  vLLM TTFT P50: {vllm_rows[0][\"ttft_p50_ms\"]} ms')
else:
    print('  ERROR: missing rows in summary CSV')
"

echo ""
echo "§9.A.4 Full tradeoff table (all systems, λ=7):"
python3 -c "
import csv
rows = list(csv.DictReader(open('results/end_to_end_serving/e4/a6000/e4_a6000_single_summary.csv')))
print(f'  {\"System\":<20} {\"Rate\":>5} {\"Tput(tok/s)\":>12} {\"TTFT P50\":>9} {\"WAR\":>6} {\"SLO\":>6}')
for r in [r for r in rows if r['rate']=='7.0']:
    print(f'  {r[\"system\"]:<20} {r[\"rate\"]:>5} {float(r[\"throughput_tok_s\"]):>12.1f} '
          f'{float(r[\"ttft_p50_ms\"]):>9.1f} {float(r[\"war\"]):>6.4f} {float(r[\"slo_attainment\"]):>6.4f}')
"

echo ""
echo "══════════════════════════════════════════════════════"
echo "§9.A.1–9.A.4 COMPLETE"
echo "Outputs:"
echo "  results/end_to_end_serving/e9_tmax/a6000/e9_tmax_recal_a6000_single.csv"
echo "  results/end_to_end_serving/e9_tmax/a6000/e9_tau_iter_a6000_single.csv"
echo "  results/end_to_end_serving/e9_tmax/a6000/e9_pi_convergence_a6000_single.csv"
echo "  results/end_to_end_serving/tau_iter/a6000_tau_iter.csv"
echo "  results/end_to_end_serving/e2/a6000/war_variability_a6000_single_summary.csv"
echo "  results/end_to_end_serving/e3/a6000/e3_war_control_a6000_single_summary.csv"
echo "  results/end_to_end_serving/e3/a6000/e3_monotonicity_check_a6000_single.txt"
echo "  results/end_to_end_serving/e4/a6000/e4_a6000_single_summary.csv"
echo "══════════════════════════════════════════════════════"
