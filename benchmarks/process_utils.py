"""
benchmarks/process_utils.py -- Server lifecycle management for benchmark harness.

Provides start/stop helpers used by bench.py, bench_apis.py, and backend wrappers.
Pattern matches existing AdapterSlots scripts: Popen with start_new_session=True + killpg teardown.
"""

import os
import signal
import subprocess
import time
import urllib.request
import urllib.error
from typing import Optional


def start_server(
    cmd: list,
    env: dict,
    health_url: str,
    timeout_s: float = 180.0,
    poll_interval_s: float = 2.0,
) -> Optional[subprocess.Popen]:
    """Launch server subprocess; poll health_url until ready or timeout."""
    proc = subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None  # crashed
        try:
            urllib.request.urlopen(health_url, timeout=2)
            return proc
        except (urllib.error.URLError, OSError):
            time.sleep(poll_interval_s)
    kill_server(proc)
    return None


def kill_server(proc: subprocess.Popen) -> None:
    """Send SIGTERM to the process group; wait up to 10 s then SIGKILL."""
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def wait_for_server(url: str, timeout_s: float = 120.0, poll_interval_s: float = 2.0) -> bool:
    """Return True when url responds 200; False on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(poll_interval_s)
    return False


def build_env(base_env: Optional[dict] = None, **overrides: str) -> dict:
    """Return a copy of base_env (default: os.environ) with overrides applied."""
    env = dict(base_env or os.environ)
    env.update({k: str(v) for k, v in overrides.items()})
    return env
