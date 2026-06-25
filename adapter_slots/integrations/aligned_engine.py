"""
aligned_engine.py -- AsyncLLMEngine subclass that schedules with
AlignmentAwareScheduler, and wires AlignmentAwareModelRunner into the
worker so Level-3 WGKP kernel promotion actually executes on the GPU.

vLLM 0.6.x's LLMEngine.__init__ builds self.scheduler directly from the
module-level `Scheduler` name in its own namespace, with no constructor
parameter or CLI flag to substitute a different scheduler class (confirmed
against vllm==0.6.3 source: vllm/engine/llm_engine.py, vllm/engine/arg_utils.py
-- no --scheduler-cls flag exists before vllm 0.8.0). Rather than rebind that
module global at import time, this subclasses the engine and rebuilds
self.scheduler immediately after the base constructor returns, using the
same config objects vLLM already validated.

There is no "--worker-cls" CLI flag for vllm==0.6.3 either, so it cannot be used to
activate AlignmentAwareModelRunner: vllm/executor/gpu_executor.py hardcodes worker
selection in _get_worker_module_and_class(), and an AlignmentAwareWorker's
_init_model_runner() override is never called by vllm/worker/worker.py, which has no
such method. A model runner wired that way would never reach any real server process.
The working extension point is instead vllm.worker.worker.Worker.__init__'s
model_runner_cls keyword argument (vllm/worker/worker.py:58,92-94), which
GPUExecutor._get_worker_kwargs() never populates. _AlignmentAwareGPUExecutorAsync
below injects it, and _get_executor_cls() swaps it in only for the plain single-GPU
async case (TP=1, no Ray, no speculative/multi-step); all other configurations fall
back to vLLM's normal executor selection unchanged.
"""

from __future__ import annotations

from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from vllm.executor.gpu_executor import GPUExecutorAsync

from adapter_slots.integrations.vllm_scheduler import AlignmentAwareScheduler
from adapter_slots.kernel.model_runner import AlignmentAwareModelRunner


class _AlignmentAwareAsyncLLMEngine(_AsyncLLMEngine):
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


class _AlignmentAwareGPUExecutorAsync(GPUExecutorAsync):
    """GPUExecutorAsync that constructs its worker with
    model_runner_cls=AlignmentAwareModelRunner, so WGKP Level-3 promotion
    (the merged-weight swap) actually runs instead of being silently inert.
    """

    def _get_worker_kwargs(self, local_rank=0, rank=0,
                            distributed_init_method=None):
        kwargs = super()._get_worker_kwargs(
            local_rank, rank, distributed_init_method
        )
        kwargs["model_runner_cls"] = AlignmentAwareModelRunner
        return kwargs


class AlignmentAwareAsyncEngine(AsyncLLMEngine):
    """AsyncLLMEngine that schedules with AlignmentAwareScheduler and runs
    promoted segments through AlignmentAwareModelRunner."""

    _engine_class = _AlignmentAwareAsyncLLMEngine

    @classmethod
    def _get_executor_cls(cls, engine_config):
        executor_class = super()._get_executor_cls(engine_config)
        if executor_class is GPUExecutorAsync:
            return _AlignmentAwareGPUExecutorAsync
        return executor_class
