"""
backend_dlora.py -- dLoRA baseline backend (OSDI '24).

dLoRA implements credit-based macro-batching + adapter migration for
multi-LoRA serving. Source: deps/dlora/ (cloned from
https://github.com/LLMServe/dLoRA-artifact, the paper's official artifact
repo -- note the upstream URL is dLoRA-artifact, not dLoRA/dLoRA).

Setup (one-time):
    git clone https://github.com/LLMServe/dLoRA-artifact deps/dlora
    cd deps/dlora && git clone https://github.com/LLMServe/PEFT-Dist PEFT-Dist
    # Needs its own env: Python 3.9, CUDA 12.2, torch 2.1.0 (see
    # deps/dlora/ae_scripts/README.md) -- distinct from this project's
    # adapter_env (CUDA 12.4, torch 2.4). Build in that env:
    #   pip install -e deps/dlora
    #   pip install -e deps/dlora/PEFT-Dist
    # Building deps/dlora requires a GPU + nvcc; it vendors its own vLLM
    # fork with CUDA extensions under deps/dlora/vllm/csrc.

dLoRA server interface (confirmed by reading
deps/dlora/vllm/entrypoints/api_server.py and
deps/dlora/vllm/engine/arg_utils.py directly):
  - Module: `python -m vllm.entrypoints.api_server` run with PYTHONPATH
    pointed at deps/dlora/ so the *vendored* `vllm` package resolves
    (NOT vLLM upstream -- dLoRA forked vLLM ~0.1.x and modified the engine/
    scheduler internals, so the package name collides with real vllm).
  - CLI args of interest: --model, --port, --tensor-parallel-size, --max-r
    (LoRA rank), --num-models (total adapter pool size), --policy credit
    (dLoRA's signature scheduling policy), --use-dummy-weights.
  - There is no /health route anywhere in this vendored vLLM fork --
    _wait_for_health() is overridden below to use a raw TCP probe instead.
  - Request: POST /generate with JSON {"prompt", "model_id" (int),
    "stream": false, plus SamplingParams fields (max_tokens, temperature,
    ...)}. model_id is an integer index into the --num-models pool, not an
    adapter path or name.
  - Response (non-streaming): {"text": ["<prompt><completion>", ...]} --
    note the completion is concatenated onto the prompt, unlike vLLM
    upstream's OpenAI-style {"choices": [...]} schema. send_request() is
    overridden below to parse this correctly instead of silently returning
    empty text via the base class's OpenAI-shaped parser.

KNOWN LIMITATION (real, not a stand-in): dLoRA-artifact's engine only
supports synthetic/dummy LoRA weights sized by --max-r -- there is no CLI
path to load real adapter checkpoints from disk (use_dummy_weights=True is
the only mode wired up in this artifact release). self.adapter_dirs is
therefore only used for its *length* (adapter pool size), not as real
checkpoint paths, unlike the punica/slora/vllm/adapterslots backends. This is an
upstream constraint of the released artifact, not a simplification made
here -- flag it in any comparison that claims dLoRA ran on the same
checkpoints as the other systems.

Reference:
    Wu et al. "dLoRA: Dynamically Orchestrating Requests and Adapters for
    LoRA LLM Serving." OSDI 2024.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import List, Tuple

from backends.base import BaseBackend


_DLORA_ROOT = Path(__file__).parent.parent / "deps" / "dlora"
_HEALTH_POLL_S = 2
_HEALTH_TIMEOUT_S = 480


def _check_dlora_installed() -> None:
    if not _DLORA_ROOT.exists():
        raise FileNotFoundError(
            "dLoRA source not found. Fetch it first:\n"
            "  git clone https://github.com/LLMServe/dLoRA-artifact deps/dlora\n"
            "  cd deps/dlora && git clone https://github.com/LLMServe/PEFT-Dist PEFT-Dist\n"
            "  pip install -e deps/dlora && pip install -e deps/dlora/PEFT-Dist\n"
            "(in a separate Python 3.9 / CUDA 12.2 / torch 2.1.0 env -- see "
            "deps/dlora/ae_scripts/README.md)."
        )


class DLoRABackend(BaseBackend):
    """dLoRA baseline backend -- credit-based macro-batching + migration.

    IMPORTANT: This backend requires deps/dlora/ to be cloned and installed
    before use. See module docstring for setup instructions and the known
    dummy-weights limitation.
    """

    def start(self) -> bool:
        _check_dlora_installed()
        return super().start()

    def _build_server_cmd(self) -> List[str]:
        _check_dlora_installed()
        return [
            "python", "-m", "vllm.entrypoints.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tp),
            "--max-r", str(self.max_lora_rank),
            "--num-models", str(max(self.max_loras, len(self.adapter_dirs) or 1)),
            "--policy", "credit",
            "--use-dummy-weights",
        ]

    def _server_env(self) -> dict:
        # Prepend deps/dlora so its vendored `vllm` fork shadows any real
        # vllm install on PYTHONPATH for this subprocess only.
        existing = os.environ.get("PYTHONPATH", "")
        peft_dist = str(_DLORA_ROOT / "PEFT-Dist")
        dlora_src = str(_DLORA_ROOT)
        new_path = os.pathsep.join(p for p in (dlora_src, peft_dist, existing) if p)
        return {"PYTHONPATH": new_path}

    def _wait_for_health(self) -> bool:
        # dLoRA's vendored api_server.py has no /health route -- the model
        # finishes loading before uvicorn binds the port (synchronous
        # EngineManager construction precedes uvicorn.run()), so a plain
        # TCP-connect probe is an accurate readiness signal here.
        deadline = time.time() + _HEALTH_TIMEOUT_S
        while time.time() < deadline:
            try:
                with socket.create_connection(("localhost", self.port), timeout=2):
                    return True
            except OSError:
                if self._proc is not None and self._proc.poll() is not None:
                    return False  # server process exited early
                time.sleep(_HEALTH_POLL_S)
        return False

    def build_request_payload(
        self, prompt: str, adapter_id: str, max_tokens: int
    ) -> Tuple[str, dict]:
        # model_id is an integer index into --num-models; adapter_id is
        # expected as "adapter_N" (same convention as backend_slora.py).
        try:
            model_id = int(adapter_id.split("_")[-1])
        except (ValueError, IndexError):
            model_id = 0
        url = f"http://localhost:{self.port}/generate"
        payload = {
            "prompt": prompt,
            "model_id": model_id,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        return url, payload

    def send_request(
        self, prompt: str, adapter_id: str, max_tokens: int
    ) -> Tuple[str, int, float]:
        # Overridden because dLoRA's response shape ({"text": [prompt +
        # completion]}) is not the OpenAI-style {"choices": [...]} schema
        # the base class's send_request() parses -- using the base
        # implementation would silently return empty text instead of
        # raising, which is exactly the failure mode this project forbids.
        import urllib.request

        url, payload = self.build_request_payload(prompt, adapter_id, max_tokens)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        latency = time.perf_counter() - t0
        full_text = body["text"][0]
        completion = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
        return completion, len(completion.split()), latency
