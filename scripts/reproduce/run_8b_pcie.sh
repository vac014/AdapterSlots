#!/usr/bin/env bash
# run_8b_pcie.sh -- Sections 8.B.2, 8.B.3, 8.B.4 on 2×A6000 PCIe
#
# Root fix: vLLM 0.6.x defaults VLLM_WORKER_MULTIPROC_METHOD=fork, which
# crashes with "Cannot re-initialize CUDA in forked subprocess" when TP>1.
# Setting it to spawn before any vLLM process starts resolves this cleanly.
# No monkey-patching of vLLM internals.
#
# All experiments use real GPU execution on both A6000s (TP=2 via env var).
# Speed multiplier = 5.0 (PCIe hardware calibration, same as §8.A.7).
set -euo pipefail

# Global environment
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # THE fix: spawn not fork for TP>1

# AdapterSlots Whittle scheduler parameters (consistent across all Part B sections)
export AS_MODE=whittle
export AS_WAR_TARGET=0.8
export AS_TTFT_SLO_MS=2000.0
export AS_EWMA_ALPHA=0.1
export AS_WHITTLE_DELTA_T=0.100   # τ_iter_PCIe ≈ 100ms (from whittle_scheduler)
export AS_PI_UPDATE_MODE=iteration_boundary
export AS_PI_KP=0.01
export AS_PI_KI=0.001

OUTDIR=results/workload_characterization/two_a6000_pcie
mkdir -p "$OUTDIR"

# Helpers
K4_LORA_ARGS="
    --lora-modules
        adapter_0=./adapters/adapter_r16_k0_s42
        adapter_1=./adapters/adapter_r16_k1_s43
        adapter_2=./adapters/adapter_r16_k2_s44
        adapter_3=./adapters/adapter_r16_k3_s45
    --max-loras 4 --max-lora-rank 16"

K16_LORA_ARGS="
    --lora-modules
        adapter_0=./adapters/adapter_r16_k0_s42
        adapter_1=./adapters/adapter_r16_k1_s43
        adapter_2=./adapters/adapter_r16_k2_s44
        adapter_3=./adapters/adapter_r16_k3_s45
        adapter_4=./adapters/adapter_r16_k4_s46
        adapter_5=./adapters/adapter_r16_k5_s47
        adapter_6=./adapters/adapter_r16_k6_s48
        adapter_7=./adapters/adapter_r16_k7_s49
        adapter_8=./adapters/adapter_r16_k8_s50
        adapter_9=./adapters/adapter_r16_k9_s51
        adapter_10=./adapters/adapter_r16_k10_s52
        adapter_11=./adapters/adapter_r16_k11_s53
        adapter_12=./adapters/adapter_r16_k12_s54
        adapter_13=./adapters/adapter_r16_k13_s55
        adapter_14=./adapters/adapter_r16_k14_s56
        adapter_15=./adapters/adapter_r16_k15_s57
    --max-loras 16 --max-lora-rank 16"

wait_for_server() {
    local port=$1
    local label=$2
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

# §8.B.2  Replay harness TP=2 validation + speed multiplier calibration
echo ""
echo "══════════════════════════════════════════════════════"
echo " §8.B.2  TP=2 validation + speed multiplier calibration"
echo "══════════════════════════════════════════════════════"

python scripts/vllm_serve_adapter_slots.py \
    --model ./models/llama-7b \
    --tensor-parallel-size 2 \
    --enable-lora \
    --lora-modules \
        adapter_0=./adapters/adapter_r16_k0_s42 \
        adapter_1=./adapters/adapter_r16_k1_s43 \
        adapter_2=./adapters/adapter_r16_k2_s44 \
        adapter_3=./adapters/adapter_r16_k3_s45 \
    --max-loras 4 --max-lora-rank 16 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.88 \
    --port 8000 \
    > "$OUTDIR/server_8b2_stdout.log" \
    2> "$OUTDIR/server_8b2_stderr.log" &
VLLM_PID=$!

if ! wait_for_server 8000 "§8.B.2 TP=2 K=4"; then
    kill_server $VLLM_PID
    echo "[FATAL] §8.B.2 server failed. Check $OUTDIR/server_8b2_stderr.log"
    exit 1
fi

# Speed multiplier calibration for PCIe (λ_mean=2 req/s as reference)
echo "[8.B.2] Running speed multiplier calibration …"
python scripts/replay_harness.py \
    --endpoint http://localhost:8000/v1/completions \
    --model llama-7b \
    --adapter-prefix adapter_ \
    --calibrate \
    --lambda-mean 2.0 \
    --output "$OUTDIR/speed_multiplier_calibration.csv"

# Replay synthetic i.i.d. K=4 trace for TP=2 harness validation
echo "[8.B.2] Replaying synthetic K=4 i.i.d. trace (TP=2 validation) …"
python scripts/replay_harness.py \
    --trace data/burstgpt/synthetic_k4_iid.jsonl \
    --endpoint http://localhost:8000/v1/completions \
    --model llama-7b \
    --adapter-prefix adapter_ \
    --speed-multiplier 5.0 \
    --timeout-s 120.0 \
    --max-concurrent 64 \
    --output "$OUTDIR/harness_validation_tp2.csv" \
    --summary-output "$OUTDIR/harness_validation_tp2_summary.csv"

kill_server $VLLM_PID

# EC §10.2 condition 1: |WAR_replay_TP2 − WAR_predicted| ≤ 0.03
python3 - <<'PYEOF'
import csv, os
pred_f = 'results/workload_characterization/a6000_single/war_offline_prediction.csv'
rep_f  = 'results/workload_characterization/two_a6000_pcie/harness_validation_tp2_summary.csv'
rep = next(csv.DictReader(open(rep_f)))
w_r = float(rep['war_mean'])
print(f"\n§8.B.2 Results:")
print(f"  WAR_replay (TP=2 PCIe) = {w_r:.4f}")
print(f"  n_success={rep['n_success']}  error_rate={rep.get('error_rate','N/A')}")
print(f"  TTFT P50={rep['ttft_p50_ms']}ms  P99={rep['ttft_p99_ms']}ms")
if os.path.exists(pred_f):
    pred = next(csv.DictReader(open(pred_f)))
    w_p = float(pred.get('war_mean', pred.get('WAR_mean', 0)))
    delta = abs(w_r - w_p)
    status = "PASS" if delta <= 0.03 else "FAIL"
    print(f"  WAR_predicted (offline) = {w_p:.4f}  |Δ| = {delta:.4f}  → EC§10.2 cond1: {status}")
PYEOF

echo "[8.B.2] DONE"

# §8.B.3  TP-transparency of autocorrelation benefit (Proposition 8.12)
echo ""
echo "══════════════════════════════════════════════════════"
echo " §8.B.3  TP-transparency of autocorrelation benefit"
echo "══════════════════════════════════════════════════════"

# TP=1 run (single GPU, same physical machine)
echo "[8.B.3] Launching TP=1 server on GPU 0 only (port 8001) …"
CUDA_VISIBLE_DEVICES=0 \
AS_WHITTLE_DELTA_T=0.030 \
python scripts/vllm_serve_adapter_slots.py \
    --model ./models/llama-7b \
    --enable-lora \
    --lora-modules \
        adapter_0=./adapters/adapter_r16_k0_s42 \
        adapter_1=./adapters/adapter_r16_k1_s43 \
        adapter_2=./adapters/adapter_r16_k2_s44 \
        adapter_3=./adapters/adapter_r16_k3_s45 \
    --max-loras 4 --max-lora-rank 16 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.88 \
    --port 8001 \
    > "$OUTDIR/server_8b3_tp1_stdout.log" \
    2> "$OUTDIR/server_8b3_tp1_stderr.log" &
TP1_PID=$!

if ! wait_for_server 8001 "§8.B.3 TP=1 K=4"; then
    kill_server $TP1_PID
    echo "[FATAL] §8.B.3 TP=1 server failed."
    exit 1
fi

echo "[8.B.3] Replaying BurstGPT K=4 30min on TP=1 …"
python scripts/replay_harness.py \
    --trace data/burstgpt/burstgpt_k4_30min.jsonl \
    --endpoint http://localhost:8001/v1/completions \
    --model llama-7b \
    --adapter-prefix adapter_ \
    --speed-multiplier 5.0 \
    --timeout-s 300.0 \
    --max-concurrent 32 \
    --output "$OUTDIR/tp1_burstgpt_replay.csv" \
    --summary-output "$OUTDIR/tp1_burstgpt_summary.csv"

kill_server $TP1_PID

# TP=2 run (both GPUs, same trace)
echo "[8.B.3] Launching TP=2 PCIe server (both GPUs, port 8000) …"
export CUDA_VISIBLE_DEVICES=0,1
export AS_WHITTLE_DELTA_T=0.100

python scripts/vllm_serve_adapter_slots.py \
    --model ./models/llama-7b \
    --tensor-parallel-size 2 \
    --enable-lora \
    --lora-modules \
        adapter_0=./adapters/adapter_r16_k0_s42 \
        adapter_1=./adapters/adapter_r16_k1_s43 \
        adapter_2=./adapters/adapter_r16_k2_s44 \
        adapter_3=./adapters/adapter_r16_k3_s45 \
    --max-loras 4 --max-lora-rank 16 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.88 \
    --port 8000 \
    > "$OUTDIR/server_8b3_tp2_stdout.log" \
    2> "$OUTDIR/server_8b3_tp2_stderr.log" &
TP2_PID=$!

if ! wait_for_server 8000 "§8.B.3 TP=2 K=4"; then
    kill_server $TP2_PID
    echo "[FATAL] §8.B.3 TP=2 server failed."
    exit 1
fi

echo "[8.B.3] Replaying BurstGPT K=4 30min on TP=2 PCIe …"
python scripts/replay_harness.py \
    --trace data/burstgpt/burstgpt_k4_30min.jsonl \
    --endpoint http://localhost:8000/v1/completions \
    --model llama-7b \
    --adapter-prefix adapter_ \
    --speed-multiplier 5.0 \
    --timeout-s 300.0 \
    --max-concurrent 32 \
    --output "$OUTDIR/tp2_burstgpt_replay.csv" \
    --summary-output "$OUTDIR/tp2_burstgpt_summary.csv"

kill_server $TP2_PID

# TP-transparency check: Proposition 8.12 -- |WAR_TP1 - WAR_TP2| ≤ 0.03
python3 - <<'PYEOF'
import csv, os

print("\n§8.B.3 Results (Proposition 8.12 -- TP-transparency):")
tp1_f = 'results/workload_characterization/two_a6000_pcie/tp1_burstgpt_summary.csv'
tp2_f = 'results/workload_characterization/two_a6000_pcie/tp2_burstgpt_summary.csv'
if os.path.exists(tp1_f) and os.path.exists(tp2_f):
    tp1 = next(csv.DictReader(open(tp1_f)))
    tp2 = next(csv.DictReader(open(tp2_f)))
    w1 = float(tp1['war_mean'])
    w2 = float(tp2['war_mean'])
    delta = abs(w1 - w2)
    status = "PASS" if delta <= 0.03 else "FAIL (>0.03 -- check scheduler integration)"
    print(f"  WAR (TP=1 A6000)     = {w1:.4f}  TTFT P99={tp1['ttft_p99_ms']}ms")
    print(f"  WAR (TP=2 PCIe)      = {w2:.4f}  TTFT P99={tp2['ttft_p99_ms']}ms")
    print(f"  |Δ_WAR|              = {delta:.4f}")
    print(f"  Proposition 8.12     → {status}")
    # Save tp-transparency table
    rows = [
        {'config': 'TP=1 (one A6000)', **tp1},
        {'config': 'TP=2 PCIe',        **tp2},
    ]
    out = 'results/workload_characterization/two_a6000_pcie/tp_transparency.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Written {out}")
else:
    print("  [WARN] summary files not found -- re-run replays")
PYEOF

echo "[8.B.3] DONE"

# §8.B.4  BurstGPT WAR at K=16 on PCIe (Theorem 8.9 §6.5c)
echo ""
echo "══════════════════════════════════════════════════════"
echo " §8.B.4  BurstGPT WAR at K=16 on 2×A6000 PCIe"
echo "══════════════════════════════════════════════════════"

export CUDA_VISIBLE_DEVICES=0,1
export AS_WHITTLE_DELTA_T=0.100

python scripts/vllm_serve_adapter_slots.py \
    --model ./models/llama-7b \
    --tensor-parallel-size 2 \
    --enable-lora \
    --lora-modules \
        adapter_0=./adapters/adapter_r16_k0_s42 \
        adapter_1=./adapters/adapter_r16_k1_s43 \
        adapter_2=./adapters/adapter_r16_k2_s44 \
        adapter_3=./adapters/adapter_r16_k3_s45 \
        adapter_4=./adapters/adapter_r16_k4_s46 \
        adapter_5=./adapters/adapter_r16_k5_s47 \
        adapter_6=./adapters/adapter_r16_k6_s48 \
        adapter_7=./adapters/adapter_r16_k7_s49 \
        adapter_8=./adapters/adapter_r16_k8_s50 \
        adapter_9=./adapters/adapter_r16_k9_s51 \
        adapter_10=./adapters/adapter_r16_k10_s52 \
        adapter_11=./adapters/adapter_r16_k11_s53 \
        adapter_12=./adapters/adapter_r16_k12_s54 \
        adapter_13=./adapters/adapter_r16_k13_s55 \
        adapter_14=./adapters/adapter_r16_k14_s56 \
        adapter_15=./adapters/adapter_r16_k15_s57 \
    --max-loras 16 --max-lora-rank 16 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.88 \
    --port 8000 \
    > "$OUTDIR/server_8b4_stdout.log" \
    2> "$OUTDIR/server_8b4_stderr.log" &
VLLM_PID=$!

if ! wait_for_server 8000 "§8.B.4 TP=2 K=16"; then
    kill_server $VLLM_PID
    echo "[FATAL] §8.B.4 K=16 server failed. Check $OUTDIR/server_8b4_stderr.log"
    exit 1
fi

# BurstGPT K=16 replay (real BurstGPT trace, Zipf α=0.9 assigned)
echo "[8.B.4] Replaying BurstGPT K=16 30min trace …"
python scripts/replay_harness.py \
    --trace data/burstgpt/burstgpt_k16_30min.jsonl \
    --endpoint http://localhost:8000/v1/completions \
    --model llama-7b \
    --adapter-prefix adapter_ \
    --speed-multiplier 5.0 \
    --timeout-s 300.0 \
    --max-concurrent 32 \
    --output "$OUTDIR/burstgpt_k16_replay.csv" \
    --summary-output "$OUTDIR/burstgpt_k16_summary.csv"

# Synthetic i.i.d. K=16 control (same server still running)
echo "[8.B.4] Replaying synthetic K=16 i.i.d. control trace …"
python scripts/replay_harness.py \
    --trace data/burstgpt/synthetic_k16_iid.jsonl \
    --endpoint http://localhost:8000/v1/completions \
    --model llama-7b \
    --adapter-prefix adapter_ \
    --speed-multiplier 5.0 \
    --timeout-s 120.0 \
    --max-concurrent 64 \
    --output "$OUTDIR/synthetic_k16_replay.csv" \
    --summary-output "$OUTDIR/synthetic_k16_summary.csv"

kill_server $VLLM_PID

# Autocorrelation / non-IID analysis of the K=16 trace is done offline from
# the replay outputs written above ($OUTDIR/synthetic_k16_*.csv).
echo "[8.B.4] Autocorrelation analysis is performed separately from $OUTDIR."

# Compute Δ_WAR_burst at K=16 (Theorem 8.9 / §6.5c)
python3 - <<'PYEOF'
import csv, os

print("\n§8.B.4 Results (Theorem 8.9 -- BurstGPT WAR > i.i.d. WAR at K=16):")
b_f = 'results/workload_characterization/two_a6000_pcie/burstgpt_k16_summary.csv'
s_f = 'results/workload_characterization/two_a6000_pcie/synthetic_k16_summary.csv'
if os.path.exists(b_f) and os.path.exists(s_f):
    b = next(csv.DictReader(open(b_f)))
    s = next(csv.DictReader(open(s_f)))
    war_b = float(b['war_mean'])
    war_s = float(s['war_mean'])
    delta  = war_b - war_s
    status = "PASS" if delta >= 0.05 else "WARN (Δ<0.05 -- check autocorrelation)"
    print(f"  WAR_BurstGPT (TP=2 K=16)  = {war_b:.4f}  TTFT P99={b['ttft_p99_ms']}ms  n={b['n_success']}")
    print(f"  WAR_iid      (TP=2 K=16)  = {war_s:.4f}  TTFT P99={s['ttft_p99_ms']}ms  n={s['n_success']}")
    print(f"  Δ_WAR_burst_PCIe_K16      = {delta:+.4f}")
    print(f"  Theorem 8.9 at K=16 PCIe → {status}")
    out = 'results/workload_characterization/two_a6000_pcie/delta_war_burst_k16.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'hardware_label','K','tau_iter_ms','war_burstgpt','war_iid','delta_war_burst'])
        w.writeheader()
        w.writerow({'hardware_label':'two_a6000_pcie','K':16,'tau_iter_ms':100,
                    'war_burstgpt':round(war_b,4),'war_iid':round(war_s,4),
                    'delta_war_burst':round(delta,4)})
    print(f"  Written {out}")

    # Throughput comparison (§8.B.6 uses this)
    t_b = float(b['n_success']) / max(float(b['wall_time_s']), 1)
    t_s = float(s['n_success']) / max(float(s['wall_time_s']), 1)
    ratio = t_b / max(t_s, 1e-6)
    print(f"\n  Throughput (BurstGPT):  {t_b:.3f} req/s")
    print(f"  Throughput (synthetic): {t_s:.3f} req/s")
    print(f"  BurstGPT/Synthetic ratio (PCIe lower bound): {ratio:.4f}")
    out2 = 'results/workload_characterization/two_a6000_pcie/throughput_comparison.csv'
    with open(out2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'hardware_label','workload','throughput_req_s','war_mean',
            'ttft_p99_ms','burstgpt_synthetic_ratio'])
        w.writeheader()
        w.writerow({'hardware_label':'two_a6000_pcie','workload':'BurstGPT',
                    'throughput_req_s':round(t_b,4),'war_mean':b['war_mean'],
                    'ttft_p99_ms':b['ttft_p99_ms'],'burstgpt_synthetic_ratio':round(ratio,4)})
        w.writerow({'hardware_label':'two_a6000_pcie','workload':'Synthetic_i.i.d.',
                    'throughput_req_s':round(t_s,4),'war_mean':s['war_mean'],
                    'ttft_p99_ms':s['ttft_p99_ms'],'burstgpt_synthetic_ratio':'1.0'})
    print(f"  Written {out2}")
else:
    print("  [WARN] K=16 summary files not found")
PYEOF

echo "[8.B.4] DONE"
echo ""
echo "══════════════════════════════════════════════════════"
echo " All §8.B.2 / §8.B.3 / §8.B.4 experiments complete."
echo " Results in: $OUTDIR"
echo "══════════════════════════════════════════════════════"
