"""
backend_slora.py -- S-LoRA baseline backend (OSDI '24).

S-LoRA ships a FastAPI HTTP server at deps/slora/slora/server/api_server.py.
This backend starts that server as a subprocess on the specified port.

Setup (benchmarks/sota/REPRODUCE_SOTA.md):
    cd deps/slora && pip install -e . --no-build-isolation --no-deps

S-LoRA server CLI (from slora/server/api_server.py):
    python -m slora.server.api_server \
        --model_dir <model> \
        --port <port> \
        --tp <tp> \
        --lora-dirs <adapter_dir1> --lora-dirs <adapter_dir2> ...
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from backends.base import BaseBackend


_SLORA_ROOT = Path(__file__).parent.parent / "deps" / "slora"


class SLoRABackend(BaseBackend):
    """S-LoRA HTTP backend (slora.server.api_server)."""

    def _build_server_cmd(self) -> List[str]:
        cmd = [
            sys.executable, "-m", "slora.server.api_server",
            "--model_dir", self.model,
            "--port", str(self.port),
            "--tp", str(self.tp),
            "--max_req_input_len", "2048",
            "--max_req_total_len", "2560",
            # KV token pool. S-LoRA's default (6000), or max_loras*512, gives it a
            # fraction of the ~55K tokens vLLM gets on the same card.
            "--max_total_token_num", os.environ.get("SLORA_MAX_TOTAL_TOKENS", "50000"),
            "--scheduler", "slora",
            # S-LoRA's unified paging (adapters swapped into the KV memory pool);
            # its loader only builds the GPU-side adapter buffers on this path.
            "--swap",
        ]
        for d in self.adapter_dirs:
            cmd += ["--lora-dirs", d]
        return cmd

    def _server_env(self) -> dict:
        # deps/slora on PYTHONPATH (SLORA_ROOT overrides the location), and torch's
        # own libs on LD_LIBRARY_PATH for S-LoRA's compiled kernels.
        import torch
        slora_src = os.environ.get("SLORA_ROOT", str(_SLORA_ROOT))
        torch_lib = str(Path(torch.__file__).parent / "lib")
        join = lambda new, var: f"{new}:{os.environ[var]}" if os.environ.get(var) else new  # noqa: E731
        return {"PYTHONPATH": join(slora_src, "PYTHONPATH"),
                "LD_LIBRARY_PATH": join(torch_lib, "LD_LIBRARY_PATH")}

    def build_request_payload(
        self, prompt: str, adapter_id: str, max_tokens: int
    ) -> Tuple[str, dict]:
        # S-LoRA selects the adapter by the top-level "lora_dir" (the path it was
        # registered with via --lora-dirs); a request without it runs on the
        # base model. Greedy, like the temperature-0 requests the vLLM backends send.
        lora_dir = self.adapter_dirs[int(adapter_id) % len(self.adapter_dirs)]
        url = f"http://localhost:{self.port}/generate_stream"
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_tokens, "do_sample": False},
            "lora_dir": lora_dir,
        }
        return url, payload

    async def stream_request(self, session, url: str, payload: dict, timeout_s: float,
                             collector, req_id: str) -> int:
        """S-LoRA streams one 'data:{"token": {"text": ...}, ...}' chunk per token."""
        import aiohttp

        n = 0
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            resp.raise_for_status()
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("token") is None:
                    continue
                now = time.perf_counter()
                if n == 0:
                    collector.record_first_token(req_id, now)
                else:
                    collector.record_token(req_id, now)
                n += 1
        return n
