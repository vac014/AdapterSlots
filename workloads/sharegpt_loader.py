"""
workloads/sharegpt_loader.py -- ShareGPT dataset loader + prompt preprocessor.

Loads ShareGPT flat JSONL (data/sharegpt/sharegpt.jsonl), extracts user turns,
filters by token length cap, and serialises deterministic 1000-prompt shards per seed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional


_DEFAULT_SHAREGPT = "data/sharegpt/sharegpt.jsonl"
_DEFAULT_ALPACA = "data/alpaca/alpaca_raw.jsonl"
_MAX_TOKENS_FALLBACK = 128


def _estimate_tokens(text: str) -> int:
    """Rough whitespace tokenisation -- avoids requiring transformers at load time."""
    return max(1, len(text.split()))


def load_sharegpt(
    path: str = _DEFAULT_SHAREGPT,
    max_tokens: int = _MAX_TOKENS_FALLBACK,
    min_tokens: int = 4,
) -> List[str]:
    """Load and filter ShareGPT prompts.

    Supports two formats:
    - Flat JSONL: each line is {"conversations": [...]} with role/value pairs
    - Single JSON array of conversation objects
    Returns list of user-turn strings that satisfy the token length filter.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ShareGPT dataset not found: {path}")

    prompts: List[str] = []
    with p.open() as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            records = json.load(f)
        else:
            records = [json.loads(line) for line in f if line.strip()]

    for rec in records:
        convs = rec.get("conversations") or rec.get("conversation") or []
        for turn in convs:
            role = turn.get("from", turn.get("role", ""))
            value = turn.get("value", turn.get("content", ""))
            if role.lower() in ("human", "user") and isinstance(value, str):
                tok_count = _estimate_tokens(value)
                if min_tokens <= tok_count <= max_tokens:
                    prompts.append(value)

    return prompts


def load_alpaca(
    path: str = _DEFAULT_ALPACA,
    max_tokens: int = _MAX_TOKENS_FALLBACK,
    min_tokens: int = 4,
) -> List[str]:
    """Load Alpaca instruction dataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Alpaca dataset not found: {path}")

    prompts: List[str] = []
    with p.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    for rec in records:
        instruction = rec.get("instruction", "")
        inp = rec.get("input", "")
        text = (instruction + " " + inp).strip() if inp else instruction
        if isinstance(text, str):
            tok_count = _estimate_tokens(text)
            if min_tokens <= tok_count <= max_tokens:
                prompts.append(text)

    return prompts


def make_shard(
    prompts: List[str],
    n: int,
    seed: int,
    output_path: Optional[str] = None,
) -> List[str]:
    """Sample n prompts with given seed; optionally write to JSONL shard file."""
    rng = random.Random(seed)
    shard = rng.sample(prompts, min(n, len(prompts)))
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            for text in shard:
                f.write(json.dumps({"prompt": text}) + "\n")
    return shard


def load_shard(path: str) -> List[str]:
    """Load a previously saved prompt shard JSONL."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    return prompts


def get_prompts(
    dataset: str = "sharegpt",
    max_tokens: int = _MAX_TOKENS_FALLBACK,
    n: int = 1000,
    seed: int = 42,
    shard_dir: Optional[str] = None,
) -> List[str]:
    """High-level helper used by bench.py: load or generate a prompt shard.

    If shard_dir is given, looks for a cached shard file at
    {shard_dir}/{dataset}_n{n}_seed{seed}.jsonl before reloading the full dataset.
    """
    if shard_dir:
        shard_path = str(
            Path(shard_dir) / f"{dataset}_n{n}_seed{seed}.jsonl"
        )
        if Path(shard_path).exists():
            return load_shard(shard_path)
    else:
        shard_path = None

    if dataset == "sharegpt":
        all_prompts = load_sharegpt(max_tokens=max_tokens)
    elif dataset == "alpaca":
        all_prompts = load_alpaca(max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'sharegpt' or 'alpaca'.")

    shard = make_shard(all_prompts, n, seed, output_path=shard_path)
    return shard
