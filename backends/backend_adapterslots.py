"""
backend_adapterslots.py -- AdapterSlots server backend.

Modes C1 to C7 launch scripts/vllm_serve_adapter_slots.py, which runs the
engine with AlignmentAwareScheduler in its own subprocess
(adapter_slots/integrations/aligned_mp_engine.py), the same process layout
vLLM uses by default. C0 launches vLLM's own entry point with no AdapterSlots
code.
"""

from __future__ import annotations

import os

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from backends.base import BaseBackend

_REPO_ROOT = Path(__file__).parent.parent
_ALIGNED_SERVER = _REPO_ROOT / "scripts" / "vllm_serve_adapter_slots.py"


# Lattice modes: environment the scheduler and kernel layer read at server
# start. AS_SCHEDULER=1 selects AlignmentAwareScheduler, AS_MODE its dispatch
# rule (adapter_slots/integrations/vllm_scheduler.py), AS_SEGGEMM the grouped
# LoRA kernels in WGKP mode (0: vLLM's kernels). A driver's AS_SEGGEMM overrides
# the default here.
CONFIG_MAP = {
    "C0": {"AS_SCHEDULER": "0"},                                            # vLLM
    "C1": {"AS_SCHEDULER": "1", "AS_MODE": "threshold"},                    # alignment buffer
    "C2": {"AS_SCHEDULER": "1", "AS_MODE": "erlang"},                       # + Erlang T*(k)
    "C3": {"AS_SCHEDULER": "1", "AS_MODE": "pi_adaptive"},                  # + PI control of T_max
    "C4": {"AS_SCHEDULER": "1", "AS_MODE": "cash"},                      # + CASH order
    "C5": {"AS_SCHEDULER": "1", "AS_MODE": "cash",                       # + PredictiveLFU
           "AS_PREFETCH_POLICY": "predictive"},
    "C6": {"AS_SCHEDULER": "1", "AS_MODE": "wgkp", "AS_SEGGEMM": "0"},     # WGKP, vLLM LoRA kernels
    "C7": {"AS_SCHEDULER": "1", "AS_MODE": "wgkp", "AS_SEGGEMM": "3"},     # WGKP, grouped LoRA kernels
}


class AdapterSlotsBackend(BaseBackend):
    """AdapterSlots backend: vLLM with AlignmentAwareScheduler."""

    def __init__(
        self,
        model: str,
        adapter_dirs: List[str],
        port: int = 8100,
        tp: int = 1,
        max_lora_rank: int = 32,
        max_loras: int = 16,
        mode: str = "C6",
        tmax_ms: Optional[int] = None,
        wgkp_threshold: Optional[int] = None,
        extra_args: Optional[List[str]] = None,
        gpu_id: Optional[int] = None,
    ) -> None:
        # AS_MAX_LORAS: GPU-resident adapter slots. Fewer slots than adapters
        # leaves the freed memory to the KV cache; vLLM's admission then only
        # batches requests whose adapters are resident.
        if os.environ.get("AS_MAX_LORAS"):
            max_loras = int(os.environ["AS_MAX_LORAS"])
        super().__init__(model, adapter_dirs, port, tp, max_lora_rank, max_loras,
                         extra_args=extra_args, gpu_id=gpu_id)
        self.mode = mode
        self.tmax_ms = tmax_ms
        self.wgkp_threshold = wgkp_threshold
        # Per-batch WAR/GWAR log (written when AS_LOG_WAR=1). Drivers name it per
        # run through AS_METRICS_PATH; the port-based default can collide when
        # two runs reuse a port.
        self.metrics_path = (os.environ.get("AS_METRICS_PATH")
                             or f"/tmp/adapter_slots_metrics_{port}.jsonl")

    def _build_server_cmd(self) -> List[str]:
        lora_modules = [
            f"adapter_{i}={d}" for i, d in enumerate(self.adapter_dirs)
        ]
        if CONFIG_MAP[self.mode]["AS_SCHEDULER"] == "1":
            entrypoint = [sys.executable, str(_ALIGNED_SERVER)]
        else:
            entrypoint = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        cmd = entrypoint + [
            "--model", self.model,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tp),
            "--enable-lora",
            "--max-lora-rank", str(self.max_lora_rank),
            "--max-loras", str(self.max_loras),
        ]
        if lora_modules:
            # vLLM's --lora-modules takes nargs='+': one flag, all pairs
            # space-separated. Repeating the flag per adapter would keep only
            # the last registration.
            cmd += ["--lora-modules", *lora_modules]
        # No --enforce-eager: the grouped LoRA kernels launch on fixed grids
        # and are captured in CUDA graphs like vLLM's own.
        cmd += self.extra_args
        return cmd

    def _server_env(self) -> dict:
        env: dict = {
            # The scheduler writes per-batch WAR/GWAR here when AS_LOG_WAR=1.
            "AS_METRICS_PATH": self.metrics_path,
            **CONFIG_MAP[self.mode],
        }
        if "AS_PREFETCH_POLICY" in env:
            # The warm set mirrors vLLM's GPU LoRA slots.
            env["AS_K_WARM"] = str(self.max_loras)
        if self.tmax_ms is not None:
            env["AS_TMAX_MS"] = str(self.tmax_ms)
        if self.wgkp_threshold is not None:
            env["AS_WGKP_THRESHOLD"] = str(self.wgkp_threshold)
        # Per-run settings passed through from the driver's environment.
        for k in ("AS_MWC_K_HOT", "AS_SEGGEMM", "AS_DEADLINE", "AS_TMAX_MS",
                  "AS_SEGGEMM_MIN_TOKENS"):
            if os.environ.get(k) is not None:
                env[k] = os.environ[k]
        return env

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
