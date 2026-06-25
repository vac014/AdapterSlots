"""Shared helpers for the experiment drivers in benchmarks/paper/.

Every driver runs from the repository root, reads its settings from environment
variables, and appends one row per run to <OUT>/results.json next to the
harness's per-run JSON. A tag that is already recorded is skipped, so an interrupted
sweep resumes where it stopped.
"""
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def setup(gpu: str = "0") -> None:
    """Run from the repo root on the GPUs named by $GPU."""
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", gpu)
    # aiohttp caps in-flight requests at 100 unless told otherwise.
    os.environ.setdefault("BENCH_CONN_LIMIT", "1024")
    # Saturated runs queue requests for minutes; the harness default (120 s per
    # request) would cut them short.
    os.environ.setdefault("BENCH_TIMEOUT", "3600")


def out_dir(default: str) -> str:
    path = os.environ.get("OUT", f"results/runs/{default}")
    os.makedirs(path, exist_ok=True)
    return path


def free_port(port: int) -> int:
    """First port >= port that a bare bind (as vLLM's API server does) accepts."""
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("", port))
            return port
        except OSError:
            port += 1
        finally:
            s.close()


def load(out: str) -> dict:
    path = f"{out}/results.json"
    return json.load(open(path)) if os.path.exists(path) else {}


def record(out: str, tag: str, row: dict) -> None:
    res = load(out)
    res[tag] = row
    json.dump(res, open(f"{out}/results.json", "w"), indent=1)


def rep_row(path: str) -> dict:
    """The fields every driver keeps from one harness run."""
    r = json.load(open(path))["reps"][0]
    return {"tput": r["throughput_toks"], "ttft_p50": r["ttft_p50_ms"],
            "ttft_p99": r["ttft_p99_ms"], "n": r["n_completed"], "wall": r["wall_time_s"],
            "tok_per_req": r["throughput_toks"] * r["wall_time_s"] / max(r["n_completed"], 1)}


def gwar(adapter_ids, threshold: int = 8) -> float:
    """GWAR(threshold): fraction of tokens in contiguous same-adapter runs of at
    least `threshold` tokens."""
    n, i, promoted = len(adapter_ids), 0, 0
    while i < n:
        j = i + 1
        while j < n and adapter_ids[j] == adapter_ids[i]:
            j += 1
        if j - i >= threshold:
            promoted += j - i
        i = j
    return promoted / n if n else 0.0


def batch_metrics(path: str) -> dict | None:
    """Mean WAR and GWAR(8) over the decode batches the scheduler logged
    (AS_LOG_WAR=1). GWAR is recomputed from each batch's token order, so it
    does not depend on the scheduler's WGKP threshold."""
    if not os.path.exists(path):
        return None
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r.get("tokens")]
    if not rows:
        return None
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    g8 = [gwar([t["adapter_id"] for t in r["tokens"]], 8) for r in rows]
    tok = [len(r["tokens"]) for r in rows]
    return {"war": mean([r["war"] for r in rows]), "gwar8": mean(g8),
            "gwar8_tok": sum(g * n for g, n in zip(g8, tok)) / sum(tok),
            "n_batches": len(rows)}


def vllm_nolora_backend():
    """vLLM serving the base model with LoRA disabled (the no-LoRA ceiling)."""
    from backends.backend_vllm import VLLMBackend

    class BaseNoLoRA(VLLMBackend):
        def _build_server_cmd(self):
            return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", self.model, "--port", str(self.port),
                    "--tensor-parallel-size", str(self.tp)] + self.extra_args

        def build_request_payload(self, prompt, adapter_id, max_tokens):
            return (f"http://localhost:{self.port}/v1/completions",
                    {"model": self.model, "prompt": prompt, "max_tokens": max_tokens,
                     "temperature": 0.0})
    return BaseNoLoRA


def vllm050_backend():
    """vLLM 0.5.0 from its own environment ($VLLM050_PYTHON), same CLI and
    request format as 0.6.3. vLLM 0.5.0 sends no usage chunk when streaming and
    merges tokens into one chunk under load, so requests go non-streaming and
    the token count comes from usage.completion_tokens."""
    from backends.backend_vllm import VLLMBackend
    python = os.environ.get("VLLM050_PYTHON", str(REPO / "venvs" / "vllm050" / "bin" / "python"))

    class VLLM050Backend(VLLMBackend):
        def _build_server_cmd(self):
            cmd = super()._build_server_cmd()
            cmd[0] = python
            return cmd

        async def stream_request(self, session, url, payload, timeout_s, collector, req_id):
            import time
            import aiohttp
            async with session.post(url, json=dict(payload, stream=False),
                                    timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                resp.raise_for_status()
                body = await resp.json()
            collector.record_first_token(req_id, time.perf_counter())
            return body["usage"]["completion_tokens"]
    return VLLM050Backend


def strip_scheduler_steps(extra):
    """vLLM 0.5.0 has no --num-scheduler-steps."""
    out, it = [], iter(extra or [])
    for a in it:
        if a == "--num-scheduler-steps":
            next(it, None)
            continue
        out.append(a)
    return out


# AdapterSlots settings shared by the drivers: K_hot = 5 for the L3 merged-weight
# cache (it holds as many adapters as fit in free GPU memory), grouped kernels
# from 64 tokens, kernel mode from $SEGMODE (3 = tile-scheduled, packed
# projections fused, prefill grouped).
def as_env() -> dict:
    return {"AS_MWC_K_HOT": "5", "AS_SEGGEMM": os.environ.get("SEGMODE", "3"),
            "AS_SEGGEMM_MIN_TOKENS": "64"}
