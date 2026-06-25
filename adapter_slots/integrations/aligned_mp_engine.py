"""
aligned_mp_engine.py -- multiprocessing-frontend support for AlignmentAware*.

Why this exists:

vLLM's default OpenAI server runs the engine's scheduling/step loop in a
*separate* subprocess (vllm.engine.multiprocessing.engine.MQLLMEngine),
connected to the HTTP-facing process via ZMQ. That isolates per-token
detokenization, HTTP response streaming, and FastAPI/asyncio overhead from
the process that drives the GPU. The in-process extension point
(AlignmentAwareAsyncEngine, in aligned_engine.py) cannot survive that split on its
own: the multiprocessing frontend spawns a *fresh* subprocess that re-imports
vanilla vLLM and constructs a plain LLMEngine, bypassing the AS scheduler and
model_runner entirely, which is why serving through it needs this module rather
than --disable-frontend-multiprocessing.

Process isolation is the whole point. Without it AS runs its entire engine
(scheduler + HTTP + tokenizer + detokenizer) in one process while vanilla vLLM gets
that isolation for free from the frontend, and at 100 concurrent decode streams the
per-token detokenization and HTTP work for the whole batch shares a GIL with the
scheduling loop on the AS side and does not on the vanilla side. That GIL contention,
not scheduler CPU cost, buffer-wait latency, CASH batch-holdback, or the Level-2/3
kernel paths (dead code in the live serving path, see model_runner.py), is what makes
AS pay a ~10% decode-throughput tax against vanilla vLLM under saturation.

This module restores process isolation for AS by subclassing the *sync*
LLMEngine/GPUExecutor/MQLLMEngine trio (mirroring what aligned_engine.py
already does for the async/in-process trio) and re-implementing
run_mp_engine() with the engine class swapped. No vLLM module attribute is
ever reassigned -- same "subclass, don't monkeypatch" rule as aligned_engine.py.
"""

from __future__ import annotations

import signal

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.multiprocessing.engine import MQLLMEngine
from vllm.executor.gpu_executor import GPUExecutor
from vllm.usage.usage_lib import UsageContext

from adapter_slots.integrations.vllm_scheduler import AlignmentAwareScheduler
from adapter_slots.kernel.model_runner import AlignmentAwareModelRunner


class AlignmentAwareLLMEngine(LLMEngine):
    """Sync LLMEngine that schedules with AlignmentAwareScheduler.

    Same rebuild-after-super().__init__() trick as
    aligned_engine._AlignmentAwareAsyncLLMEngine -- vLLM 0.6.x's LLMEngine
    hardcodes the scheduler class with no constructor/CLI override, and
    _AsyncLLMEngine subclasses LLMEngine directly (confirmed:
    _AsyncLLMEngine.__bases__ == (LLMEngine,)), so this is the exact same
    rebuild applied to the class both engine flavors share.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduler = [
            AlignmentAwareScheduler(
                self.scheduler_config,
                self.cache_config,
                self.lora_config,
                self.parallel_config.pipeline_parallel_size,
                self.async_callbacks[v_id]
                if self.model_config.use_async_output_proc else None,
            )
            for v_id in range(self.parallel_config.pipeline_parallel_size)
        ]


class _AlignmentAwareGPUExecutor(GPUExecutor):
    """Sync GPUExecutor that injects model_runner_cls=AlignmentAwareModelRunner.

    Mirrors aligned_engine._AlignmentAwareGPUExecutorAsync for the
    subprocess-side sync executor used by MQLLMEngine.
    """

    def _get_worker_kwargs(self, local_rank=0, rank=0,
                            distributed_init_method=None):
        kwargs = super()._get_worker_kwargs(
            local_rank, rank, distributed_init_method
        )
        kwargs["model_runner_cls"] = AlignmentAwareModelRunner
        return kwargs


class AlignmentAwareMQLLMEngine(MQLLMEngine):
    """MQLLMEngine that wraps AlignmentAwareLLMEngine instead of LLMEngine.

    MQLLMEngine.__init__ constructs `self.engine = LLMEngine(...)` inline --
    there is no factory method to override, so __init__ is reimplemented
    here in full (copied from vllm.engine.multiprocessing.engine.MQLLMEngine.
    __init__, vllm==0.6.3) with that one line changed. Everything else
    (ZMQ sockets, heartbeat thread, error state) is identical to upstream.
    """

    def __init__(self, ipc_path: str, use_async_sockets: bool, *args,
                 log_requests: bool = True, **kwargs) -> None:
        import threading
        import time as _time

        import zmq  # type: ignore[import]
        from vllm.engine.multiprocessing import (
            IPC_DATA_EXT, IPC_HEALTH_EXT, IPC_INPUT_EXT, IPC_OUTPUT_EXT,
        )
        from vllm.envs import VLLM_RPC_TIMEOUT

        use_cached_outputs = True

        self.engine = AlignmentAwareLLMEngine(
            *args, **kwargs, use_cached_outputs=use_cached_outputs
        )
        self.log_requests = log_requests

        self.use_async_sockets = use_async_sockets
        if self.use_async_sockets:
            self.engine.process_request_outputs_callback = \
                self._async_socket_engine_callback

        self.ctx = zmq.Context()

        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.bind(f"{ipc_path}{IPC_INPUT_EXT}")

        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.bind(f"{ipc_path}{IPC_OUTPUT_EXT}")

        self.heartbeat_socket = self.ctx.socket(zmq.constants.PUSH)
        self.heartbeat_socket.bind(f"{ipc_path}{IPC_HEALTH_EXT}")

        self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"

        self._errored_with = None

        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True)
        self._heartbeat_stop_event = threading.Event()
        self.heartbeat_interval_seconds = VLLM_RPC_TIMEOUT / 5000.0

        self._last_alive_time = _time.time()
        self.last_alive_threshold = VLLM_RPC_TIMEOUT * 3.0 / 1000.0

    @classmethod
    def from_engine_args(cls, engine_args, usage_context: UsageContext,
                         ipc_path: str):
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        engine_config = engine_args.create_engine_config()

        executor_class = LLMEngine._get_executor_cls(engine_config)
        if executor_class is GPUExecutor:
            executor_class = _AlignmentAwareGPUExecutor

        return cls(
            ipc_path=ipc_path,
            use_async_sockets=engine_config.model_config.use_async_output_proc,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
        )


def run_mp_engine_aligned(engine_args, usage_context: UsageContext,
                          ipc_path: str) -> None:
    """Subprocess entrypoint -- same shape as vllm.engine.multiprocessing.
    engine.run_mp_engine, with the engine class swapped to
    AlignmentAwareMQLLMEngine. Spawned via multiprocessing.get_context(
    "spawn"), so this only ever runs in the fresh child process.
    """

    def signal_handler(*_) -> None:
        raise KeyboardInterrupt("MQLLMEngine terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    engine = AlignmentAwareMQLLMEngine.from_engine_args(
        engine_args=engine_args, usage_context=usage_context, ipc_path=ipc_path
    )
    engine.start()
