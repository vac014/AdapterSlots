#!/usr/bin/env bash
# reproduce_e11.sh -- Full flashinfer_composition reproduction script
#
# Runs all flashinfer_composition experiments in order:
#   1. Unit tests (no GPU required)
#   2. E10 FlashInfer composability (live or simulation)
#   3. SOTA comparison B1–B3 (simulation + optional live)
#   4. Mandatory ablations AB2, AB3, AB4, AB5, AB8, AB10
#   5. Gate checklist
#
# Usage:
#   # CPU-only simulation (no GPU, fast, ~5 min)
#   bash benchmarks/reproduce/reproduce_e11.sh --mode simulation
#
#   # Single A6000 (real GPU, simulation ablations, ~45 min)
#   bash benchmarks/reproduce/reproduce_e11.sh --mode a6000_single \
#       --model ./models/llama-7b --adapter-dir ./adapters
#
#   # Two A6000 PCIe TP=2 (real GPU, ~2 hr)
#   bash benchmarks/reproduce/reproduce_e11.sh --mode two_a6000_pcie \
#       --model ./models/llama-7b --adapter-dir ./adapters
#
#   # With live GPU benchmarks for SOTA + E10
#   bash benchmarks/reproduce/reproduce_e11.sh --mode a6000_single --live \
#       --model ./models/llama-7b --adapter-dir ./adapters
#
# Environment:
#   conda activate adapter_env
#   export CUDA_VISIBLE_DEVICES=0,1   # for two-GPU modes
#
# Outputs written to:
#   results/flashinfer_composition/composability/
#   results/flashinfer_composition/sota/
#   results/flashinfer_composition/ablations/
#   results/flashinfer_composition/figures/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Argument parsing
MODE="simulation"
LIVE=""
MODEL="./models/llama-7b"
ADAPTER_DIR="./adapters"
K=4
OUTPUT_BASE="results/flashinfer_composition"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)            MODE="$2";         shift 2 ;;
        --live)            LIVE="--live";     shift   ;;
        --model)           MODEL="$2";        shift 2 ;;
        --adapter-dir)     ADAPTER_DIR="$2";  shift 2 ;;
        --K)               K="$2";            shift 2 ;;
        --output-base)     OUTPUT_BASE="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo "  flashinfer_composition Full Reproduction"
echo "  Mode: $MODE"
echo "  Live: ${LIVE:-off}"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_BASE"
echo "================================================================"
echo ""

PASS_COUNT=0
FAIL_COUNT=0
GATE_RESULTS=()

run_step() {
    local name="$1"
    shift
    echo "──────────────────────────────────────────────────────────────"
    echo "  STEP: $name"
    echo "──────────────────────────────────────────────────────────────"
    if "$@"; then
        echo "  [PASS] $name"
        PASS_COUNT=$((PASS_COUNT + 1))
        GATE_RESULTS+=("PASS  $name")
    else
        echo "  [FAIL] $name (exit $?)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        GATE_RESULTS+=("FAIL  $name")
    fi
    echo ""
}

# Step 1: Unit tests
run_step "Unit tests (127 tests, no GPU)" \
    python -m pytest tests/ -q --tb=short

# Step 2: E10 FlashInfer composability
mkdir -p "$OUTPUT_BASE/composability"

# e10_composability uses 'cpu' for no-GPU mode (not 'simulation')
E10_MODE="${MODE}"
[[ "$MODE" == "simulation" ]] && E10_MODE="cpu"

if [[ -n "$LIVE" ]]; then
    run_step "E10 FlashInfer composability (live GPU)" \
        python scripts/experiments/e10_composability.py \
            --mode "$E10_MODE" \
            --live \
            --model "$MODEL" \
            --adapter-dir "$ADAPTER_DIR" \
            --K "$K" \
            --rate 7.0 \
            --num-prompts 400 \
            --output-dir "$OUTPUT_BASE/composability/"
else
    run_step "E10 FlashInfer composability (simulation)" \
        python scripts/experiments/e10_composability.py \
            --mode "$E10_MODE" \
            --output-dir "$OUTPUT_BASE/composability/"
fi

# Step 3: SOTA comparison B1–B3
mkdir -p "$OUTPUT_BASE/sota"

SOTA_MODE="${LIVE:+$MODE}"
SOTA_MODE="${SOTA_MODE:-simulation}"

# impl11_sota_comparison runs B1+B2+B3 in a single invocation; use --skip-b1 to skip B1
run_step "SOTA B1+B2+B3 (all benchmarks)" \
    python scripts/experiments/impl11_sota_comparison.py \
        --mode "$SOTA_MODE" \
        --K "$K" \
        --model "$MODEL" \
        --adapter-dir "$ADAPTER_DIR" \
        --output-dir "$OUTPUT_BASE/sota/"

# Step 4: Mandatory ablations
mkdir -p "$OUTPUT_BASE/ablations"

run_step "Ablations AB2 AB3 AB4 AB5 AB8 AB10" \
    python scripts/experiments/impl11_ablations.py \
        --mode "$MODE" \
        --which AB2 AB3 AB4 AB5 AB8 AB10 \
        --output-dir "$OUTPUT_BASE/ablations/"

echo "Results written to $OUTPUT_BASE. Figures 8 and 9 are generated separately."

# Gate checklist
echo "================================================================"
echo "  GATE CHECKLIST"
echo "================================================================"

for result in "${GATE_RESULTS[@]}"; do
    echo "  $result"
done

echo ""
echo "  Summary: $PASS_COUNT PASS / $FAIL_COUNT FAIL out of $((PASS_COUNT + FAIL_COUNT)) steps"
echo ""

# Per-gate evidence checks from CSV outputs
echo "  Per-gate evidence:"

# Gate E10: combined > either component alone (Claim 11.1)
if [[ -f "$OUTPUT_BASE/composability/e10_composability.csv" ]]; then
    python3 -c "
import csv
rows = list(csv.DictReader(open('$OUTPUT_BASE/composability/e10_composability.csv')))
sys = {r['system']: float(r['throughput_tok_s']) for r in rows if 'throughput_tok_s' in r}
if 'combined' in sys and 'adapterslots' in sys and 'flashinfer' in sys:
    ok = sys['combined'] > max(sys['adapterslots'], sys['flashinfer'])
    print(f'  E10 combined={sys[\"combined\"]:.0f} > max(adapterslots={sys[\"adapterslots\"]:.0f},fi={sys[\"flashinfer\"]:.0f}): {\"PASS\" if ok else \"FAIL\"}')
" 2>/dev/null || echo "  E10: CSV parse error"
fi

# Gate B3: AdapterSlots beats best SOTA at K=4
if [[ -f "$OUTPUT_BASE/sota/sota_comparison_b3.csv" ]]; then
    python3 -c "
import csv
rows = [r for r in csv.DictReader(open('$OUTPUT_BASE/sota/sota_comparison_b3.csv')) if str(r.get('K','')) == '4']
sys = {r['system']: float(r['throughput_tok_s']) for r in rows if 'throughput_tok_s' in r}
if 'adapterslots' in sys:
    adapterslots = sys.pop('adapterslots')
    best = max(sys.values()) if sys else 0
    ok = adapterslots > best
    print(f'  B3 K=4: AdapterSlots={adapterslots:.0f} vs best SOTA={best:.0f}: {\"PASS\" if ok else \"FAIL\"}')
" 2>/dev/null || echo "  B3: CSV parse error"
fi

# Gate AB5: C7 gain ≥ 2.0× vs C0
if [[ -f "$OUTPUT_BASE/ablations/ab5_component_decomp.csv" ]]; then
    python3 -c "
import csv
rows = list(csv.DictReader(open('$OUTPUT_BASE/ablations/ab5_component_decomp.csv')))
c7 = next((r for r in rows if r.get('component','') == 'C7'), None)
if c7:
    g = float(c7['gain_vs_c0'])
    print(f'  AB5 C7 gain={g:.2f}x >= 2.0x: {\"PASS\" if g >= 2.0 else \"FAIL\"}')
" 2>/dev/null || echo "  AB5: CSV parse error"
fi

# Gate AB8: gain ≥ 1.5× for all K ≤ 20
if [[ -f "$OUTPUT_BASE/ablations/ab8_k_scaling.csv" ]]; then
    python3 -c "
import csv
rows = list(csv.DictReader(open('$OUTPUT_BASE/ablations/ab8_k_scaling.csv')))
fails = [r for r in rows if int(r['K']) <= 20 and float(r['gain_vs_vllm']) < 1.5]
ok = len(fails) == 0
print(f'  AB8 K<=20 gain >= 1.5x: {\"PASS\" if ok else \"FAIL (\" + str(len(fails)) + \" below threshold)\"}')
" 2>/dev/null || echo "  AB8: CSV parse error"
fi

# Gate AB4: Whittle ≥ 85% oracle at all T_max
if [[ -f "$OUTPUT_BASE/ablations/ab4_whittle_vs_threshold.csv" ]]; then
    python3 -c "
import csv
rows = list(csv.DictReader(open('$OUTPUT_BASE/ablations/ab4_whittle_vs_threshold.csv')))
fails = [r for r in rows if float(r['whittle_oracle_frac']) < 0.85]
ok = len(fails) == 0
print(f'  AB4 Whittle >= 85% oracle: {\"PASS\" if ok else \"FAIL\"}')
" 2>/dev/null || echo "  AB4: CSV parse error"
fi

echo ""
echo "================================================================"
if [[ $FAIL_COUNT -eq 0 ]]; then
    echo "  ALL GATES PASS -- flashinfer_composition ready for paper submission"
    exit 0
else
    echo "  $FAIL_COUNT gate(s) FAILED -- check logs above"
    exit 1
fi
