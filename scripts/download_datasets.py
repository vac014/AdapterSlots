"""
download_datasets.py -- Download benchmark datasets.

Datasets:
    sharegpt   Aeala/ShareGPT_Vicuna_unfiltered (primary workload)
    lmsys      lmsys/lmsys-chat-1m (workload_characterization workload characterization)
    mmlu       cais/mmlu (end_to_end_serving quality eval)
    mtbench    MT-Bench (manual download instructions printed)

Usage:
    python scripts/download_datasets.py --datasets sharegpt lmsys
    python scripts/download_datasets.py --datasets all
"""

import argparse
import json
import os
import sys


DATASETS = {
    "sharegpt": {
        "hf_name": "Aeala/ShareGPT_Vicuna_unfiltered",
        "filename": "ShareGPT_V3_unfiltered_cleaned_split.json",
        "local": "data/sharegpt",
    },
    "lmsys": {
        "hf_name": "lmsys/lmsys-chat-1m",
        "local": "data/lmsys",
    },
    "mmlu": {
        "hf_name": "cais/mmlu",
        "local": "data/mmlu",
        "subsets": ["all"],
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Download benchmark datasets")
    p.add_argument("--datasets", type=str, nargs="+",
                   default=["sharegpt"],
                   help=f"Datasets to download. Choices: {list(DATASETS.keys())} or 'all'")
    p.add_argument("--output-dir", type=str, default="./data",
                   help="Root data directory")
    p.add_argument("--hf-token", type=str, default=None)
    return p.parse_args()


def download_hf_dataset(hf_name: str, local_dir: str, token: str = None, subsets=None):
    """Download a HuggingFace dataset to local_dir."""
    from datasets import load_dataset
    os.makedirs(local_dir, exist_ok=True)
    print(f"  Downloading {hf_name} ...")

    if subsets:
        for subset in subsets:
            ds = load_dataset(hf_name, subset, token=token)
            subset_dir = os.path.join(local_dir, subset)
            ds.save_to_disk(subset_dir)
            print(f"    Saved {subset} -> {subset_dir}")
    else:
        ds = load_dataset(hf_name, token=token)
        ds.save_to_disk(local_dir)
        print(f"  Saved -> {local_dir}")


def convert_sharegpt_to_jsonl(data_dir: str):
    """Convert ShareGPT JSON to JSONL for the workload replay harness."""
    src = os.path.join(data_dir, "ShareGPT_V3_unfiltered_cleaned_split.json")
    dst = os.path.join(data_dir, "sharegpt.jsonl")
    if not os.path.exists(src):
        print(f"  [skip] {src} not found")
        return

    with open(src) as f:
        data = json.load(f)

    with open(dst, "w") as out:
        for item in data:
            # Extract first human turn as prompt
            human_turns = [t for t in item.get("conversations", [])
                           if t.get("from") == "human"]
            if human_turns:
                rec = {
                    "id": item.get("id", ""),
                    "prompt": human_turns[0]["value"][:2048],  # truncate
                    "source": "sharegpt",
                }
                out.write(json.dumps(rec) + "\n")

    print(f"  Converted -> {dst}")


def main():
    args = parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN")

    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = list(DATASETS.keys())

    for ds_key in datasets_to_download:
        if ds_key not in DATASETS:
            print(f"[ERROR] Unknown dataset: {ds_key}. "
                  f"Choose from: {list(DATASETS.keys())}")
            sys.exit(1)

        ds_info = DATASETS[ds_key]
        local_dir = os.path.join(args.output_dir, ds_key)
        print(f"\n[{ds_key}] {ds_info['hf_name']}")

        try:
            download_hf_dataset(
                hf_name=ds_info["hf_name"],
                local_dir=local_dir,
                token=token,
                subsets=ds_info.get("subsets"),
            )
            if ds_key == "sharegpt":
                convert_sharegpt_to_jsonl(local_dir)
        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\nMT-Bench: manual download required.")
    print("  See: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge")
    print("  Save questions to: data/mtbench/question.jsonl")
    print("\nBurstGPT: see scripts/preprocess_burstgpt.py for instructions.")
    print("\nDone.")


if __name__ == "__main__":
    main()
