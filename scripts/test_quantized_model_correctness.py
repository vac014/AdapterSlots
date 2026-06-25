"""
test_quantized_model_correctness.py -- sanity check that LoRA adapters applied
on top of the GPTQ-Marlin INT4 base model still produce coherent, on-topic
output, before any throughput claim is made about that configuration.

This does NOT assert exact-match against the FP16 baseline -- INT4
quantization is expected to introduce wording-level differences (it changed
the model's actual weights, not just its execution speed). What it checks,
by eye, is that outputs stay fluent and factually on-topic rather than
becoming garbage. This is the same check that validated the §10 result:
one prompt came back an exact token-for-token match, the other three
differed in wording only while staying coherent and correct.

Requires both ./models/llama-13b (FP16) and ./models/llama-13b-gptq
(download via: python scripts/download_models.py --models llama-13b llama-13b-gptq)
and at least 2 real LoRA adapters under ./adapters_13b/.

Usage:
    python scripts/test_quantized_model_correctness.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.backend_vllm import VLLMBackend

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, there was a",
    "The three laws of robotics are",
]


def query(port: int, adapter_idx: int, prompt: str, max_tokens: int = 40) -> str:
    url = f"http://localhost:{port}/v1/completions"
    payload = {"model": f"adapter_{adapter_idx}", "prompt": prompt,
               "max_tokens": max_tokens, "temperature": 0.0, "logprobs": 1}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["text"]


def run_one(model: str, port: int, adapter_dirs, extra_args=None) -> list:
    bkd = VLLMBackend(model=model, adapter_dirs=adapter_dirs, port=port, tp=1,
                       max_lora_rank=32, max_loras=16, extra_args=extra_args)
    bkd.start()
    outs = []
    try:
        for i, p in enumerate(PROMPTS):
            out = query(port, i % len(adapter_dirs), p)
            outs.append(out)
            print(f"adapter={i % len(adapter_dirs)} prompt={p!r}\n  -> {out!r}", flush=True)
    finally:
        bkd.stop()
    return outs


def main() -> None:
    adapter_dirs = ["adapters_13b/adapter_r32_k0_s42", "adapters_13b/adapter_r32_k1_s43"]
    if not all(Path(d).is_dir() for d in adapter_dirs):
        print(f"Adapter dirs not found: {adapter_dirs} -- generate with "
              f"scripts/gen_adapters.py first.", file=sys.stderr)
        sys.exit(1)

    print("=== FP16 baseline ===", flush=True)
    fp16_outs = run_one("./models/llama-13b", 9201, adapter_dirs)

    print("\n=== GPTQ-Marlin INT4 + LoRA ===", flush=True)
    gptq_outs = run_one("./models/llama-13b-gptq", 9202, adapter_dirs,
                         extra_args=["--quantization", "gptq_marlin"])

    print("\n=== COMPARISON ===", flush=True)
    for p, a, b in zip(PROMPTS, fp16_outs, gptq_outs):
        match = "EXACT MATCH" if a == b else "DIFFERS (expected for quantization -- check it's still coherent above)"
        print(f"prompt={p!r}\n  fp16: {a!r}\n  gptq: {b!r}\n  {match}", flush=True)


if __name__ == "__main__":
    main()
