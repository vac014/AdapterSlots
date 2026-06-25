"""
backend_vllm.py -- Vanilla vLLM baseline backend.

Starts vLLM with standard PagedAttention scheduling (no adapter alignment).
Used as the primary baseline in all ablation and SOTA comparisons.
"""

from __future__ import annotations

import sys
from typing import List, Tuple

from backends.base import BaseBackend


class VLLMBackend(BaseBackend):
    """Vanilla vLLM backend -- no alignment scheduler."""

    def _build_server_cmd(self) -> List[str]:
        lora_modules = [
            f"adapter_{i}={d}" for i, d in enumerate(self.adapter_dirs)
        ]
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tp),
            "--enable-lora",
            "--max-lora-rank", str(self.max_lora_rank),
            "--max-loras", str(self.max_loras),
        ]
        if lora_modules:
            # vLLM's --lora-modules takes nargs='+': one flag, all pairs
            # space-separated. Repeating the flag per adapter (the previous
            # code here) silently overwrites all but the last registration.
            cmd += ["--lora-modules", *lora_modules]
        cmd += self.extra_args
        return cmd

    def build_request_payload(
        self, prompt: str, adapter_id: str, max_tokens: int
    ) -> Tuple[str, dict]:
        url = f"http://localhost:{self.port}/v1/completions"
        # Workload generator hands us the raw int adapter index (Request.adapter_id);
        # the registered LoRA name from _build_server_cmd is "adapter_{i}".
        payload = {
            "model": f"adapter_{adapter_id}",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        return url, payload
