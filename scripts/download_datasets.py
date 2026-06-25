"""Download the ShareGPT prompt set the benchmarks read.

The drivers load data/sharegpt/sharegpt.jsonl, which is the ShareGPT V3 JSON
array (ShareGPT_V3_unfiltered_cleaned_split.json) under that name;
workloads/sharegpt_loader.py reads either a JSON array or one JSON object per
line. Prompts are drawn from it by seed, so the file must be this exact one for
the request traces to match ours (md5 in REPRODUCE.md).

Usage:
    python scripts/download_datasets.py [--output-dir ./data] [--hf-token TOKEN]
"""
import argparse
import os
import shutil

REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", default="./data")
    p.add_argument("--hf-token", default=None)
    args = p.parse_args()

    from huggingface_hub import hf_hub_download
    src = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset",
                          token=args.hf_token or os.environ.get("HF_TOKEN"))
    dst_dir = os.path.join(args.output_dir, "sharegpt")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "sharegpt.jsonl")
    shutil.copyfile(src, dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
