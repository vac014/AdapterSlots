"""
backend_slora.py -- S-LoRA baseline backend (OSDI '24).

S-LoRA ships a FastAPI HTTP server at deps/slora/slora/server/api_server.py.
This backend starts that server as a subprocess on the specified port.

Setup:
    pip install -e deps/slora      (or conda activate slora)
    Requires: uvicorn, uvloop, fastapi, and S-LoRA's custom MBGMV CUDA kernels.

S-LoRA server CLI (from slora/server/api_server.py):
    python -m slora.server.api_server \
        --model_dir <model> \
        --port <port> \
        --tp <tp> \
        --lora-dirs <adapter_dir1> --lora-dirs <adapter_dir2> ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

from backends.base import BaseBackend


_SLORA_ROOT = Path(__file__).parent.parent / "deps" / "slora"


class SLoRABackend(BaseBackend):
    """S-LoRA HTTP backend -- starts slora.server.api_server."""

    def _build_server_cmd(self) -> List[str]:
        cmd = [
            sys.executable, "-m", "slora.server.api_server",
            "--model_dir", self.model,
            "--port", str(self.port),
            "--tp", str(self.tp),
            "--max_req_input_len", "2048",
            "--max_req_total_len", "2560",
            "--max_total_token_num", str(self.max_loras * 512),
            "--scheduler", "slora",
        ]
        for d in self.adapter_dirs:
            cmd += ["--lora-dirs", d]
        return cmd

    def _server_env(self) -> dict:
        # Ensure deps/slora is on PYTHONPATH so slora package is importable
        import os
        existing = os.environ.get("PYTHONPATH", "")
        slora_src = str(_SLORA_ROOT)
        new_path = f"{slora_src}:{existing}" if existing else slora_src
        return {"PYTHONPATH": new_path}

    def build_request_payload(
        self, prompt: str, adapter_id: str, max_tokens: int
    ) -> Tuple[str, dict]:
        # S-LoRA uses a generate endpoint with lora_id as integer index
        # adapter_id is expected as "adapter_N" -- extract N
        try:
            lora_idx = int(adapter_id.split("_")[-1])
        except (ValueError, IndexError):
            lora_idx = 0
        url = f"http://localhost:{self.port}/generate"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "lora_id": lora_idx,
            },
        }
        return url, payload
