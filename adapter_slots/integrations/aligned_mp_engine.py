"""
aligned_mp_engine.py -- AlignmentAware engine for vLLM's multiprocessing frontend.

vLLM's default OpenAI server runs the engine loop in a subprocess
(MQLLMEngine) connected to the HTTP process over ZMQ, which keeps
detokenization and streaming off the process that drives the GPU. That
subprocess builds a plain LLMEngine, so this module subclasses the sync
LLMEngine, GPU executors and MQLLMEngine and provides run_mp_engine_aligned()
with the engine class swapped. No vLLM module attribute is reassigned.
"""

from __future__ import annotations

import signal

from vllm.engine.llm_engine import LLMEngine
from vllm.engine.multiprocessing.engine import MQLLMEngine
from vllm.executor.gpu_executor import GPUExecutor
from vllm.executor.multiproc_gpu_executor import MultiprocessingGPUExecutor
from vllm.usage.usage_lib import UsageContext

from adapter_slots.integrations.vllm_scheduler import AlignmentAwareScheduler
from adapter_slots.kernel.model_runner import AlignmentAwareModelRunner


class AlignmentAwareLLMEngine(LLMEngine):
    """Sync LLMEngine that schedules with AlignmentAwareScheduler.

    vLLM 0.6.3 builds its scheduler inside LLMEngine.__init__ with no override,
    so the scheduler is rebuilt right after it returns, from the same configs.
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


class _AlignmentAwareMultiprocGPUExecutor(MultiprocessingGPUExecutor):
    """The same model_runner_cls injection for tensor parallelism (TP > 1).

    vLLM selects MultiprocessingGPUExecutor once tensor_parallel_size > 1. It
    builds the driver and every spawned worker through _get_worker_kwargs(), so
    this one override installs the kernel wrappers on all ranks.
    AlignmentAwareModelRunner is a module-level class and pickles by reference
    into the spawned processes.
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

    MQLLMEngine.__init__ constructs its LLMEngine inline, so __init__ is
    vLLM 0.6.3's with that one line changed; sockets, heartbeat and error
    state are as upstream.
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
        elif executor_class is MultiprocessingGPUExecutor:
            executor_class = _AlignmentAwareMultiprocGPUExecutor

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
    """Subprocess entry point: vLLM's run_mp_engine with
    AlignmentAwareMQLLMEngine as the engine class."""

    def signal_handler(*_) -> None:
        raise KeyboardInterrupt("MQLLMEngine terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    engine = AlignmentAwareMQLLMEngine.from_engine_args(
        engine_args=engine_args, usage_context=usage_context, ipc_path=ipc_path
    )
    engine.start()
