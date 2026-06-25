"""
backend_adapterslots.py -- AdapterSlots backend (AdapterSlotsBackend).

Starts vLLM with AlignmentAwareScheduler and the requested AS_MODE.
The "adapterslots" identifier is kept as a short CLI token; it does not appear
in user-facing strings (those say "AdapterSlots").

vLLM 0.6.x (the pinned baseline version, see benchmarks/sota/SOTA_VERSIONS.txt)
has no CLI flag to substitute a custom Scheduler class, so modes that need
AlignmentAwareScheduler (AS_SCHEDULER=1) launch through
scripts/vllm_serve_adapter_slots.py, which by default runs
AlignmentAwareMQLLMEngine in its own subprocess (see
adapter_slots/integrations/aligned_mp_engine.py) -- same process-isolation
architecture vanilla vLLM uses by default, restored here after direct
measurement showed it was the structural cause of AS's residual
decode-throughput gap. The in-process AlignmentAwareAsyncEngine
(aligned_engine.py) remains available via --disable-frontend-multiprocessing.
C0 (vanilla, no alignment) launches vLLM's own entrypoint directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from backends.base import BaseBackend

_REPO_ROOT = Path(__file__).parent.parent
_ALIGNED_SERVER = _REPO_ROOT / "scripts" / "vllm_serve_adapter_slots.py"


# Mapping from bench.py mode strings (C0-C7) to the env vars that
# AlignmentAwareScheduler and the kernel layer read at server start.
# AS_SCHEDULER selects vLLM's default scheduler ("0") vs.
# AlignmentAwareScheduler ("1"); AS_MODE selects the dispatch policy
# (see adapter_slots/integrations/vllm_scheduler.py); AS_FUSED_KERNEL
# selects the Level-2 fused Triton kernel (see kernel/fused_lora_kernel.py).
CONFIG_MAP = {
    "C0": {"AS_SCHEDULER": "0"},                                            # vanilla vLLM, no alignment
    "C1": {"AS_SCHEDULER": "1", "AS_MODE": "threshold"},                    # alignment_buffer
    "C2": {"AS_SCHEDULER": "1", "AS_MODE": "erlang"},                       # erlang_scheduler
    "C3": {"AS_SCHEDULER": "1", "AS_MODE": "pi_adaptive"},                  # pi_controller
    "C4": {"AS_SCHEDULER": "1", "AS_MODE": "whittle"},                      # whittle_scheduler
    "C5": {"AS_SCHEDULER": "1", "AS_MODE": "whittle"},
    "C6": {"AS_SCHEDULER": "1", "AS_MODE": "wgkp", "AS_FUSED_KERNEL": "0"}, # WGKP, fused kernel off (stock per-slice SGMV/BGMV path)
    "C7": {"AS_SCHEDULER": "1", "AS_MODE": "wgkp", "AS_FUSED_KERNEL": "1"}, # WGKP + graph-safe packed-nslice fusion (kernel/fused_punica_wrapper.py) -- no --enforce-eager needed
}


class AdapterSlotsBackend(BaseBackend):
    """AdapterSlots backend -- vLLM + AlignmentAwareScheduler."""

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
    ) -> None:
        super().__init__(model, adapter_dirs, port, tp, max_lora_rank, max_loras,
                         extra_args=extra_args)
        self.mode = mode
        self.tmax_ms = tmax_ms
        self.wgkp_threshold = wgkp_threshold
        self.metrics_path = f"/tmp/adapter_slots_metrics_{port}.jsonl"

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
            # space-separated. Repeating the flag per adapter (the previous
            # code here) silently overwrites all but the last registration.
            cmd += ["--lora-modules", *lora_modules]
        # No --enforce-eager here: the packed-nslice fused kernel
        # (kernel/fused_punica_wrapper.py) uses a fixed-grid, on-device
        # per-row index load -- same CUDA-graph-capture compatibility as
        # vLLM's own stock bgmv kernels. An earlier version of this fused
        # path (kernel/fused_lora_layers.py, now uninstalled by default)
        # did Python-side data-dependent dispatch and required
        # --enforce-eager, which measured 3.5x-10x SLOWER end-to-end
        # because it disabled graph capture for the entire model, not just
        # the LoRA layers. See docs/custom_kernel/kernel.md §1.
        cmd += self.extra_args
        return cmd

    def _server_env(self) -> dict:
        env: dict = {
            "AS_BATCH_LOG_PATH": self.metrics_path,
            **CONFIG_MAP[self.mode],
        }
        if self.tmax_ms is not None:
            env["AS_TMAX_MS"] = str(self.tmax_ms)
        if self.wgkp_threshold is not None:
            env["AS_WGKP_THRESHOLD"] = str(self.wgkp_threshold)
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
