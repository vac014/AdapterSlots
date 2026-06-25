#!/usr/bin/env python3
"""
generate_workload.py -- Thin wrapper around scripts/workload_gen.py.

Accepts all workload_gen.py arguments and additionally accepts --rate as
an alias for --arrival-rate, matching the conventions used in RUN_README.md
sections 2.7.3 (Two A6000 PCIe WARτ stress) and 2.8.3 (Two H100 NVLink stress).

Usage (as referenced in RUN_README.md §2.7.3):
    python scripts/generate_workload.py \
        --pattern zipf \
        --K 16 \
        --zipf-alpha 0.9 \
        --rate 2000 \
        --n-requests 20000 \
        --output workloads/stress_k16_rate2000_n20000.jsonl

Usage (as referenced in RUN_README.md §2.8.3):
    python scripts/generate_workload.py \
        --pattern zipf \
        --K 32 \
        --zipf-alpha 0.9 \
        --rate 5000 \
        --n-requests 50000 \
        --output workloads/stress_k32_rate5000_n50000.jsonl

All other flags are forwarded unchanged to workload_gen.py.
"""

import subprocess
import sys
import os


def main() -> None:
    raw_args = sys.argv[1:]

    # Translate --rate → --arrival-rate (workload_gen.py uses --arrival-rate).
    translated: list[str] = []
    i = 0
    while i < len(raw_args):
        if raw_args[i] == "--rate":
            if i + 1 >= len(raw_args):
                print("error: --rate requires a value", file=sys.stderr)
                sys.exit(1)
            translated.extend(["--arrival-rate", raw_args[i + 1]])
            i += 2
        elif raw_args[i].startswith("--rate="):
            translated.append("--arrival-rate=" + raw_args[i].split("=", 1)[1])
            i += 1
        else:
            translated.append(raw_args[i])
            i += 1

    # Resolve path to workload_gen.py relative to this script's location.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workload_gen = os.path.join(script_dir, "workload_gen.py")

    cmd = [sys.executable, workload_gen] + translated
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
