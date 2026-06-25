"""
backend_punica.py -- Punica SGMV baseline backend.

Punica (ASPLOS '24) provides BGMV/SGMV kernels for batched LoRA inference
but does NOT ship a standalone HTTP server. Its serving model is in-process:
one Python process loads the model and runs batched token generation.

This backend wraps Punica's bench_textgen_lora.py as a subprocess that:
  1. Loads the model and LoRA adapters into GPU memory
  2. Runs a batched generation benchmark at the specified request rate
  3. Writes JSON metrics to a temp file and exits

Because Punica is not HTTP-based, start()/stop() manage a subprocess that
runs the full benchmark in one shot rather than serving individual requests.
build_request_payload() and send_request() are not used for Punica -- the
benchmark harness should call run_benchmark() directly.

Setup: pip install -e deps/punica    (or conda activate punica)
       Requires CUDA and compiled Punica C extensions.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

from backends.base import BaseBackend


# Path to Punica source relative to this file (two levels up → deps/punica)
_PUNICA_ROOT = Path(__file__).parent.parent / "deps" / "punica"
_BENCH_SCRIPT = _PUNICA_ROOT / "benchmarks" / "bench_textgen_lora.py"


class PunicaBackend(BaseBackend):
    """Punica SGMV in-process benchmark backend.

    Unlike HTTP-based backends, Punica runs as a single subprocess that
    performs the entire benchmark and writes results to a JSON file.
    The benchmark harness must call run_punica_benchmark() rather than
    the standard start/send/stop cycle used for HTTP backends.
    """

    def __init__(
        self,
        model: str,
        adapter_dirs: List[str],
        port: int = 8100,         # unused for Punica; kept for interface compat
        tp: int = 1,
        max_lora_rank: int = 32,
        max_loras: int = 16,
        num_requests: int = 500,
        max_seq_len: int = 256,
        lora_popularity: str = "zipf",
    ) -> None:
        super().__init__(model, adapter_dirs, port, tp, max_lora_rank, max_loras)
        self.num_requests = num_requests
        self.max_seq_len = max_seq_len
        self.lora_popularity = lora_popularity
        self._result_file: Optional[str] = None
        self._bench_proc: Optional[subprocess.Popen] = None

    # Lifecycle (not HTTP-based)

    def start(self) -> bool:
        """No persistent server to start; returns True immediately."""
        if not _BENCH_SCRIPT.exists():
            raise FileNotFoundError(
                f"Punica benchmark script not found at {_BENCH_SCRIPT}. "
                "Run: pip install -e deps/punica"
            )
        return True

    def stop(self) -> None:
        """Terminate benchmark subprocess if still running."""
        if self._bench_proc is not None and self._bench_proc.poll() is None:
            self._bench_proc.terminate()
            try:
                self._bench_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._bench_proc.kill()
            self._bench_proc = None

    def run_punica_benchmark(self) -> dict:
        """Run the full Punica in-process benchmark and return metrics dict.

        Returns keys: throughput_toks, ttft_p50_ms, ttft_p99_ms, n_completed
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        self._result_file = tmp.name

        cmd = [
            sys.executable, str(_BENCH_SCRIPT),
            "--model", self.model,
            "--num-requests", str(self.num_requests),
            "--maxlen", str(self.max_seq_len),
            "--lora-popularity", self.lora_popularity,
            "--num-lora-models", str(len(self.adapter_dirs)),
            "--output-json", self._result_file,
        ]
        # Pass adapter directories as positional or env var depending on Punica version
        env = {**__import__("os").environ, "PUNICA_LORA_DIRS": ":".join(self.adapter_dirs)}

        self._bench_proc = subprocess.Popen(cmd, env=env)
        self._bench_proc.wait()

        if self._bench_proc.returncode != 0:
            raise RuntimeError(
                f"Punica benchmark subprocess exited with code "
                f"{self._bench_proc.returncode}"
            )

        with open(self._result_file) as f:
            raw = json.load(f)

        # Normalise to the shared metrics schema
        return {
            "throughput_toks": raw.get("throughput_tok_s", 0.0),
            "throughput_reqs": raw.get("throughput_req_s", 0.0),
            "ttft_p50_ms": raw.get("latency_p50_ms", 0.0),
            "ttft_p99_ms": raw.get("latency_p99_ms", 0.0),
            "n_completed": raw.get("num_requests", self.num_requests),
        }

    # HTTP interface (not applicable for Punica)

    def _build_server_cmd(self) -> List[str]:
        raise NotImplementedError(
            "Punica does not run as an HTTP server. "
            "Use run_punica_benchmark() instead."
        )

    def build_request_payload(
        self, prompt: str, adapter_id: str, max_tokens: int
    ) -> Tuple[str, dict]:
        raise NotImplementedError(
            "Punica does not expose an HTTP API. "
            "Use run_punica_benchmark() for end-to-end measurements."
        )
