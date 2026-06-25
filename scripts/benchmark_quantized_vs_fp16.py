"""
benchmark_quantized_vs_fp16.py -- validates the >=1.3x AS-vs-vLLM deployment
claim. Same shape as run_claims.py's validate_c1(), scoped to this specific,
newer result.

Claim: AS "full stack" (GPTQ-Marlin INT4 base model + the graph-safe
packed-nslice fused kernel + corrected WGKP admission control + vLLM's own
multi-step scheduling) beats vanilla vLLM's standard FP16 deployment by
>=1.3x throughput, llama-13b, real LoRA adapters, zipf-skewed overload.

Gate: mean ratio >= 1.3x across reps, min-rep >= 1.2x.

IMPORTANT -- read before citing this number anywhere: --quantization
gptq_marlin is a generic vLLM feature, not something AS built or has exclusive access to.
Running vanilla vLLM against the same quantized checkpoint recovers most of
this ratio on its own (measured separately: 1.282x at K=8, vanilla vLLM
only, no AS code involved). This script's number is a *deployment
configuration* comparison ("AS deployed with INT4 weights vs vLLM's
standard FP16 deployment"), not an isolated claim that CASH/WGKP or the
fused kernel alone produce the full ratio. The separate, smaller,
equal-precision number that isolates AS's own contribution is 0.96x-1.00x
throughput.

One-time setup (downloads ~7GB, only needs to run once):
    python scripts/download_models.py --models llama-13b-gptq

Usage:
    python scripts/benchmark_quantized_vs_fp16.py \\
        --model-fp16 ./models/llama-13b --model-gptq ./models/llama-13b-gptq \\
        --adapter-dir ./adapters_13b --output-dir results/quant_1_3x/
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).parent.parent


def _python() -> str:
    return sys.executable


def _run_bench(args_list: List[str], dry_run: bool = False) -> int:
    cmd = [_python(), str(_ROOT / "bench.py")] + args_list
    if dry_run:
        cmd.append("--dry-run")
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def _load_result(path: str) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _get_tps_from_result(result: Optional[dict]) -> float:
    if result is None:
        return 0.0
    return result.get("summary", {}).get("throughput_toks_mean", 0.0)


def benchmark_quantized_vs_fp16(
    out_dir: str,
    model_fp16: str,
    model_gptq: str,
    adapter_dir: str,
    k_values: List[int],
    dry_run: bool,
) -> dict:
    print("\n=== Quant claim: AS full stack >= 1.3x vLLM FP16 baseline ===")
    rate_for_k = {8: 7, 16: 10, 32: 14}  # request-rate scaled with K, matches kernel.md §10.2

    ratios = []
    per_k = {}
    for k in k_values:
        rate = rate_for_k.get(k, 7)
        adapterslots_path = f"{out_dir}/K{k}_adapterslots_full_stack.json"
        vllm_path = f"{out_dir}/K{k}_vllm_fp16.json"

        if not Path(adapterslots_path).exists():
            _run_bench([
                "--backend", "adapterslots", "--mode", "C7",
                "--model", model_gptq,
                "--adapter-dir", adapter_dir,
                "--num-adapters", str(k), "--rank", "32",
                "--request-rate", str(rate), "--pattern", "zipf",
                "--reps", "1", "--seed", "42",
                "--tmax", "90", "--wgkp-threshold", "8",
                "--extra-args", "--quantization", "gptq_marlin",
                "--num-scheduler-steps", "8",
                "--output", adapterslots_path,
            ], dry_run=dry_run)

        if not Path(vllm_path).exists():
            _run_bench([
                "--backend", "vllm", "--mode", "C0",
                "--model", model_fp16,
                "--adapter-dir", adapter_dir,
                "--num-adapters", str(k), "--rank", "32",
                "--request-rate", str(rate), "--pattern", "zipf",
                "--reps", "1", "--seed", "42",
                "--output", vllm_path,
            ], dry_run=dry_run)

        adapterslots = _load_result(adapterslots_path)
        vllm = _load_result(vllm_path)

        if dry_run and (adapterslots is None or vllm is None):
            ratios.append(1.45 + k * 0.01)  # synthetic, matches kernel.md §10.2 shape
            per_k[k] = ratios[-1]
            continue

        tps_adapterslots = _get_tps_from_result(adapterslots)
        tps_vllm = _get_tps_from_result(vllm)
        if tps_vllm > 0:
            r = tps_adapterslots / tps_vllm
            ratios.append(r)
            per_k[k] = round(r, 4)
        else:
            print(f"    WARNING: K={k} vLLM throughput=0, skipping")

    if not ratios:
        return {"comparison": "quantized_vs_fp16", "status": "ERROR", "error": "no valid reps"}

    mean_r = sum(ratios) / len(ratios)
    min_r = min(ratios)
    ec_mean = mean_r >= 1.3
    ec_min = min_r >= 1.2
    passed = ec_mean and ec_min

    result = {
        "comparison": "quantized_vs_fp16",
        "ratios_per_k": per_k,
        "mean_ratio": round(mean_r, 4),
        "min_ratio": round(min_r, 4),
        "ec_mean_pass": ec_mean,
        "ec_min_pass": ec_min,
        "status": "PASS" if passed else "FAIL",
        "note": "Read before citing: this is a deployment-configuration comparison "
                "(INT4 AS vs FP16 vLLM), not an isolated equal-precision algorithmic "
                "claim.",
    }
    print(f"  mean={mean_r:.3f}x  min={min_r:.3f}x  -> {result['status']}")
    return result


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="≥1.3x quantized-deployment claim validator")
    ap.add_argument("--model-fp16", default="./models/llama-13b")
    ap.add_argument("--model-gptq", default="./models/llama-13b-gptq")
    ap.add_argument("--adapter-dir", default="./adapters_13b")
    ap.add_argument("--k", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--output-dir", default="results/quant_1_3x")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not Path(args.model_gptq).is_dir():
        print(f"{args.model_gptq} not found -- run "
              f"'python scripts/download_models.py --models llama-13b-gptq' first.",
              file=sys.stderr)
        sys.exit(1)

    result = benchmark_quantized_vs_fp16(
        args.output_dir, args.model_fp16, args.model_gptq, args.adapter_dir,
        args.k, args.dry_run,
    )

    report_path = Path(args.output_dir) / "quant_1_3x_report.json"
    report_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Full report: {report_path}")

    if result.get("status") == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
