"""
base.py -- Abstract interface that every serving backend must implement.

Contract:
    1. Instantiate with model path, adapter dirs, port, and backend-specific kwargs.
    2. Call start() -- launches a real server subprocess, waits for /health, returns True.
    3. Send requests via build_request_payload() or send_request().
    4. Call stop() -- SIGTERM → wait → SIGKILL the process group.

No backend is allowed to return simulated or synthetic metrics.
If the real server fails to start, start() must raise RuntimeError.
"""

from __future__ import annotations

import abc
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple


_HEALTH_POLL_S = 2
_HEALTH_TIMEOUT_S = 480  # 4-GPU / 100-LoRA load can take several minutes


class BaseBackend(abc.ABC):
    """Abstract base for all serving backends."""

    def __init__(
        self,
        model: str,
        adapter_dirs: List[str],
        port: int = 8100,
        tp: int = 1,
        max_lora_rank: int = 32,
        max_loras: int = 16,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.adapter_dirs = adapter_dirs
        self.port = port
        self.tp = tp
        self.max_lora_rank = max_lora_rank
        self.max_loras = max_loras
        # Passthrough for one-off experimentation with vLLM server flags
        # (e.g. --num-scheduler-steps, --enable-chunked-prefill) without
        # needing a dedicated constructor param + CONFIG_MAP entry for
        # every flag under test. Empty by default -- no effect on any
        # existing benchmark unless a caller explicitly passes something.
        self.extra_args = list(extra_args) if extra_args else []
        self._proc: Optional[subprocess.Popen] = None

    # Lifecycle

    def start(self) -> bool:
        """Launch the server subprocess and wait until /health responds.

        Returns True on success, raises RuntimeError on timeout.
        """
        cmd = self._build_server_cmd()
        env = {**os.environ, **self._server_env()}
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,  # separate process group for clean SIGKILL
        )
        if not self._wait_for_health():
            self.stop()
            raise RuntimeError(
                f"{self.__class__.__name__}: server did not become healthy on "
                f"port {self.port} within {_HEALTH_TIMEOUT_S}s"
            )
        return True

    def stop(self) -> None:
        """Send SIGTERM to the process group; SIGKILL after 20 s if still alive."""
        if self._proc is None:
            return
        try:
            pgid = os.getpgid(self._proc.pid)
        except ProcessLookupError:
            pgid = None

        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        try:
            self._proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass

        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()

        self._proc = None
        # Allow GPU memory to fully release before next launch
        time.sleep(5)

    # Request interface

    @abc.abstractmethod
    def build_request_payload(
        self, prompt: str, adapter_id: str, max_tokens: int
    ) -> Tuple[str, dict]:
        """Return (url, json_payload) for an aiohttp POST."""

    def send_request(
        self, prompt: str, adapter_id: str, max_tokens: int
    ) -> Tuple[str, int, float]:
        """Synchronous single request. Returns (text, n_tokens, latency_s)."""
        import json
        import time

        url, payload = self.build_request_payload(prompt, adapter_id, max_tokens)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        latency = time.perf_counter() - t0
        text = body.get("choices", [{}])[0].get("text", "")
        n_tokens = body.get("usage", {}).get("completion_tokens", len(text.split()))
        return text, n_tokens, latency

    # Subclass hooks

    @abc.abstractmethod
    def _build_server_cmd(self) -> List[str]:
        """Return the subprocess command list to launch the server."""

    def _server_env(self) -> dict:
        """Extra environment variables to set for the server process."""
        return {}

    # Helpers

    def _wait_for_health(self) -> bool:
        url = f"http://localhost:{self.port}/health"
        deadline = time.time() + _HEALTH_TIMEOUT_S
        while time.time() < deadline:
            try:
                urllib.request.urlopen(url, timeout=2)
                return True
            except Exception:
                if self._proc is not None and self._proc.poll() is not None:
                    # Server exited early
                    return False
                time.sleep(_HEALTH_POLL_S)
        return False
