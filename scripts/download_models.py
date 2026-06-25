"""
download_models.py -- Download the base models from the HuggingFace Hub.

Usage:
    python scripts/download_models.py --models llama-7b llama-13b --output-dir ./models

Each model lands in ./models/<name>/ in HuggingFace format. Both are the
huggyllama conversions of LLaMA-7B and LLaMA-13B and need no HF_TOKEN.
"""

import argparse
import os
import sys

MODEL_MAP = {
    "llama-7b": "huggyllama/llama-7b",
    "llama-13b": "huggyllama/llama-13b",
}


def parse_args():
    p = argparse.ArgumentParser(description="Download models from HuggingFace")
    p.add_argument("--models", type=str, nargs="+",
                   choices=list(MODEL_MAP.keys()),
                   default=["llama-7b"],
                   help="Models to download")
    p.add_argument("--output-dir", type=str, default="./models",
                   help="Local directory to save models")
    p.add_argument("--hf-token", type=str, default=None,
                   help="HuggingFace API token (or set HF_TOKEN env var)")
    p.add_argument("--revision", type=str, default="main",
                   help="Git revision / branch to download")
    return p.parse_args()


def download_model(hf_name: str, local_dir: str, token: str = None, revision: str = "main"):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    print(f"\nDownloading {hf_name} -> {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=hf_name,
        local_dir=local_dir,
        token=token,
        revision=revision,
        ignore_patterns=["*.pt", "*.bin.index.json"],  # prefer safetensors
    )

    # Loading the tokenizer checks the download.
    print("  Verifying tokenizer ...")
    try:
        tok = AutoTokenizer.from_pretrained(local_dir)
        print(f"  Vocab size: {tok.vocab_size}")
    except Exception as e:
        print(f"  [WARNING] Tokenizer load failed: {e}")

    print(f"  Done -> {local_dir}")


def main():
    args = parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN")

    for model_key in args.models:
        hf_name = MODEL_MAP[model_key]
        local_dir = os.path.join(args.output_dir, model_key)
        try:
            download_model(hf_name, local_dir, token, args.revision)
        except Exception as e:
            print(f"[ERROR] Failed to download {hf_name}: {e}")
            sys.exit(1)

    print(f"\nAll models downloaded to {args.output_dir}")


if __name__ == "__main__":
    main()
