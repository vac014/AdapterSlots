"""
preprocess_burstgpt.py -- Download and preprocess the BurstGPT trace dataset.

Source: lzzmm/BurstGPT on Hugging Face (arxiv:2401.17644)
CSV columns: Timestamp, Model, Request tokens, Response tokens, Total tokens, Log Type

Output JSONL format (compatible with workload replay harness):
    {"request_id": 0, "adapter_id": 0, "arrival_time_ms": 0.0,
     "prompt_len": 472, "output_len": 18}

Adapter mapping:
    0 → ChatGPT (GPT-3.5-turbo)
    1 → GPT-4

Usage:
    python scripts/preprocess_burstgpt.py
    python scripts/preprocess_burstgpt.py --input data/burstgpt/BurstGPT_without_fails_1.csv
    python scripts/preprocess_burstgpt.py --no-download --input data/burstgpt/BurstGPT_without_fails_1.csv
"""

import argparse
import json
import os

import pandas as pd
import requests


HF_BASE = "https://huggingface.co/datasets/lzzmm/BurstGPT/resolve/main/data"
DEFAULT_FILES = [
    "BurstGPT_without_fails_1.csv",
    "BurstGPT_without_fails_2.csv",
]

MODEL_TO_ADAPTER = {
    "ChatGPT": 0,
    "GPT-4":   1,
}


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess BurstGPT trace dataset")
    p.add_argument("--data-dir", default="data/burstgpt",
                   help="Directory to store raw CSVs and output JSONL")
    p.add_argument("--input", nargs="+", default=None,
                   help="Specific CSV file(s) to process (skips download)")
    p.add_argument("--no-download", action="store_true",
                   help="Do not download CSVs; use existing files in --data-dir")
    p.add_argument("--output", default=None,
                   help="Output JSONL path (default: <data-dir>/burstgpt.jsonl)")
    p.add_argument("--min-prompt-len", type=int, default=1,
                   help="Drop requests with fewer prompt tokens than this")
    p.add_argument("--min-output-len", type=int, default=1,
                   help="Drop requests with fewer output tokens than this")
    return p.parse_args()


def download_file(url: str, dest: str):
    print(f"  Downloading {os.path.basename(dest)} ...", flush=True)
    r = requests.get(url, stream=True, allow_redirects=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f"  Saved -> {dest}")


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df


def preprocess(dfs: list[pd.DataFrame], min_prompt: int, min_output: int) -> list[dict]:
    df = pd.concat(dfs, ignore_index=True)

    # Drop rows with missing or zero token counts
    df = df.dropna(subset=["Timestamp", "Request tokens", "Response tokens", "Model"])
    df = df[df["Request tokens"] >= min_prompt]
    df = df[df["Response tokens"] >= min_output]

    # Sort by timestamp to get chronological order
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Timestamps are in seconds; normalize to ms relative to first request
    t0 = df["Timestamp"].iloc[0]
    df["arrival_time_ms"] = (df["Timestamp"] - t0) * 1000.0

    # Map model name to adapter id; unknown models get adapter 0
    df["adapter_id"] = df["Model"].map(MODEL_TO_ADAPTER).fillna(0).astype(int)

    ids          = range(len(df))
    adapter_ids  = df["adapter_id"].tolist()
    arrivals     = df["arrival_time_ms"].round(3).tolist()
    prompt_lens  = df["Request tokens"].astype(int).tolist()
    output_lens  = df["Response tokens"].astype(int).tolist()

    records = [
        {
            "request_id":      i,
            "adapter_id":      int(a),
            "arrival_time_ms": float(t),
            "prompt_len":      int(p),
            "output_len":      int(o),
        }
        for i, a, t, p, o in zip(ids, adapter_ids, arrivals, prompt_lens, output_lens)
    ]
    return records


def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    # Determine input CSV paths
    if args.input:
        csv_paths = args.input
    else:
        csv_paths = [os.path.join(args.data_dir, f) for f in DEFAULT_FILES]
        if not args.no_download:
            for path, fname in zip(csv_paths, DEFAULT_FILES):
                if os.path.exists(path):
                    print(f"  [skip] {path} already exists")
                else:
                    download_file(f"{HF_BASE}/{fname}", path)

    # Load and concatenate
    dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"  [warn] {path} not found, skipping")
            continue
        print(f"  Loading {path} ...")
        dfs.append(load_csv(path))

    if not dfs:
        print("[ERROR] No CSV files found. Run without --no-download to fetch them.")
        raise SystemExit(1)

    print(f"  Preprocessing {sum(len(d) for d in dfs):,} rows ...")
    records = preprocess(dfs, args.min_prompt_len, args.min_output_len)

    out_path = args.output or os.path.join(args.data_dir, "burstgpt.jsonl")
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"  Wrote {len(records):,} records -> {out_path}")

    # Summary
    import collections
    adapter_counts = collections.Counter(r["adapter_id"] for r in records)
    inv_map = {v: k for k, v in MODEL_TO_ADAPTER.items()}
    print("  Adapter distribution:")
    for aid in sorted(adapter_counts):
        name = inv_map.get(aid, f"unknown-{aid}")
        cnt = adapter_counts[aid]
        print(f"    {aid} ({name}): {cnt:,} ({cnt/len(records):.1%})")
    if records:
        span_s = records[-1]["arrival_time_ms"] / 1000.0
        print(f"  Trace span: {span_s:,.0f} s  ({len(records)/span_s:.2f} req/s avg)" if span_s > 0 else "")


if __name__ == "__main__":
    main()
