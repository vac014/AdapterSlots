"""
gen_adapters.py -- Deterministic LoRA adapter generator.

Usage:
    python scripts/gen_adapters.py --model ./models/llama-7b --K 4 --rank 16
    python scripts/gen_adapters.py --model ./models/llama-7b --K 8 --rank 8 16 32 64
    python scripts/gen_adapters.py --dry-run  # Preview without loading model

    # Generate adapters k8..k15 (for K=16 experiments):
    python scripts/gen_adapters.py --model ./models/llama-7b --K 8 --rank 16 --start-k 8

Generates K adapters starting at k-index --start-k (default 0).
Adapter k_index i gets seed = base_seed + i (global index).
Saves each adapter under --output-dir and writes adapters/manifest.json.
"""

import argparse
import json
import os

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Generate LoRA adapters deterministically")
    p.add_argument("--model", type=str, default="./models/llama-7b",
                   help="HuggingFace model path or local directory")
    p.add_argument("--K", type=int, default=4,
                   help="Number of adapters to generate")
    p.add_argument("--rank", type=int, nargs="+", default=[16],
                   help="LoRA rank(s). If multiple, generates one set per rank.")
    p.add_argument("--base-seed", type=int, default=42,
                   help="Starting seed. Adapter at global k-index i gets seed = base_seed + i.")
    p.add_argument("--start-k", type=int, default=0,
                   help="First k-index to generate. Useful for adding adapters k8..k15 "
                        "when k0..k7 already exist: --K 8 --start-k 8.")
    p.add_argument("--output-dir", type=str, default="./adapters",
                   help="Root directory to save adapter checkpoints")
    p.add_argument("--lora-alpha-factor", type=float, default=2.0,
                   help="lora_alpha = lora_alpha_factor * rank")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", type=str, nargs="+",
                   default=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without loading model or writing files")
    return p.parse_args()


def generate_single_adapter(base_model_path: str, adapter_name: str, rank: int,
                             seed: int, output_dir: str, target_modules: list,
                             lora_alpha_factor: float, lora_dropout: float,
                             dry_run: bool = False) -> str:
    """Generate one LoRA adapter with a fixed seed. Returns save path."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    save_path = os.path.join(output_dir, adapter_name)

    if dry_run:
        print(f"  [dry-run] {adapter_name} (rank={rank}, seed={seed}) -> {save_path}")
        return save_path

    print(f"\n[gen] {adapter_name}  rank={rank}  seed={seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"  Loading {base_model_path} on CPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",        # keep on CPU; we only care about weight init
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=int(lora_alpha_factor * rank),
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"  Saved -> {save_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return save_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = {
        "base_model": args.model,
        "K": args.K,
        "start_k": args.start_k,
        "base_seed": args.base_seed,
        "lora_alpha_factor": args.lora_alpha_factor,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "adapters": [],
    }

    for rank in args.rank:
        for offset in range(args.K):
            k = args.start_k + offset
            seed = args.base_seed + k
            name = f"adapter_r{rank}_k{k}_s{seed}"
            path = generate_single_adapter(
                base_model_path=args.model,
                adapter_name=name,
                rank=rank,
                seed=seed,
                output_dir=args.output_dir,
                target_modules=args.target_modules,
                lora_alpha_factor=args.lora_alpha_factor,
                lora_dropout=args.lora_dropout,
                dry_run=args.dry_run,
            )
            manifest["adapters"].append({
                "name": name,
                "rank": rank,
                "k_index": k,
                "seed": seed,
                "path": os.path.abspath(path),
            })


    manifest_path = os.path.join(args.output_dir, "manifest.json")
    if not args.dry_run:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest -> {manifest_path}")
    else:
        print("\n[dry-run] Manifest would contain:")
        print(json.dumps(manifest, indent=2))

    print(f"\nDone. Total adapters: {len(manifest['adapters'])}")


if __name__ == "__main__":
    main()
