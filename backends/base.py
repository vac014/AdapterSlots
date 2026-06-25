"""
base.py -- Abstract interface that every serving backend must implement.

Contract:
    1. Instantiate with model path, adapter dirs, port, and backend-specific kwargs.
    2. start() launches the server subprocess, waits for /health and returns True.
    3. Send requests via build_request_payload() or send_request().
    4. stop() sends SIGTERM to the process group, then SIGKILL if it is still alive.

If the server fails to start, start() raises RuntimeError.
"""

from __future__ import annotations

import abc
import os
import signal
import socket
import subprocess
import time
import urllib.request
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
        gpu_id: Optional[int] = None,
    ) -> None:
        self.model = model
        self.adapter_dirs = adapter_dirs
        self.port = port
        self.tp = tp
        self.max_lora_rank = max_lora_rank
        self.max_loras = max_loras
        # Extra server flags appended to the command line (the drivers pass
        # e.g. --max-num-seqs and --num-scheduler-steps here).
        self.extra_args = list(extra_args) if extra_args else []
        # Pin this server to one device (APIS runs one TP=1 server per GPU).
        self.gpu_id = gpu_id
        # Optional open file for the server's stdout and stderr (default: inherited).
        self.log_file = None
        self._proc: Optional[subprocess.Popen] = None

    # Lifecycle

    def _wait_for_port_free(self, timeout_s: int = 180) -> None:
        """Block until self.port can be bound.

        Consecutive arms reuse one port. A previous arm's server can still hold
        it when the next one launches; the new server then dies with EADDRINUSE
        and the health wait burns its full timeout before reporting a failure
        that looks unrelated.
        """
        deadline = time.time() + timeout_s
        while True:
            # No SO_REUSEADDR: vLLM's api_server binds a bare socket, so a port
            # in TIME_WAIT is free to a SO_REUSEADDR probe but not to the server.
            # The probe has to be exactly as strict as the bind it predicts.
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("", self.port))
                return
            except OSError:
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"{self.__class__.__name__}: port {self.port} still in "
                        f"use after {timeout_s}s"
                    )
                time.sleep(2)
            finally:
                sock.close()

    def _wait_for_gpus_free(self, env: dict, max_used_mib: int = 1024,
                            timeout_s: int = 180) -> None:
        """Block until the server's GPUs hold no leftover allocations.

        vLLM sizes its KV cache from the memory free at startup, so a previous
        server still releasing memory would shrink the next one's
        batch capacity.
        """
        ids = [i for i in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if i.strip()]
        if not ids:
            return
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                     "-i", ",".join(ids)], capture_output=True, text=True, timeout=10).stdout
                used = [int(x) for x in out.split()]
            except (OSError, ValueError, subprocess.SubprocessError):
                return
            if used and max(used) <= max_used_mib:
                return
            time.sleep(2)

    def start(self) -> bool:
        """Launch the server subprocess and wait until /health responds.

        Returns True on success, raises RuntimeError on timeout.
        """
        self._wait_for_port_free()
        cmd = self._build_server_cmd()
        env = {**os.environ, **self._server_env()}
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        self._wait_for_gpus_free(env)
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT if self.log_file is not None else None,
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
