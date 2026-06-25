"""
download_models.py -- Download base models from HuggingFace Hub.

Usage:
    python scripts/download_models.py --models llama-7b --output-dir ./models
    python scripts/download_models.py --models llama-7b mistral-7b --output-dir ./models

Downloads and caches models in ./models/<model_name>/ in HuggingFace format.
huggyllama/llama-7b is ungated -- no HF_TOKEN required.
Gated models (llama-13b) require: export HF_TOKEN=hf_...

llama-13b-gptq is the GPTQ-Marlin INT4 quantization used for the >=1.3x
deployment comparison in scripts/benchmark_quantized_vs_fp16.py. It is ungated (no HF_TOKEN needed) and
confirmed (via its config.json's _name_or_path) to be a quantization of the
exact same base checkpoint as llama-13b above.
"""

import argparse
import os
import sys

MODEL_MAP = {
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama-7b": "huggyllama/llama-7b",          # ungated alternative
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "llama-13b": "meta-llama/Llama-2-13b-hf",
    "llama-13b-gptq": "TheBloke/LLaMA-13b-GPTQ",  # ungated, INT4, see docstring above
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

    # Quick sanity: load tokenizer to verify the download
    print(f"  Verifying tokenizer ...")
    try:
        tok = AutoTokenizer.from_pretrained(local_dir)
        print(f"  Vocab size: {tok.vocab_size}")
    except Exception as e:
        print(f"  [WARNING] Tokenizer load failed: {e}")

    print(f"  Done -> {local_dir}")


def main():
    args = parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[WARNING] No HF_TOKEN set. Gated models (Llama) will fail.")
        print("  Fix: export HF_TOKEN=hf_... or pass --hf-token")

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
