"""
aligned_engine.py -- in-process AsyncLLMEngine that schedules with
AlignmentAwareScheduler and runs AlignmentAwareModelRunner.

vLLM 0.6.3 builds self.scheduler inside LLMEngine.__init__ from its
module-level Scheduler name and has no flag to substitute another class, so
the engine subclass rebuilds the scheduler right after the base constructor
returns, from the configs vLLM already validated. The model runner is passed
through Worker's model_runner_cls keyword, which _AlignmentAwareGPUExecutorAsync
fills in; it replaces the executor only in the single-GPU async case, and every
other configuration keeps vLLM's executor selection.
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
    model_runner_cls=AlignmentAwareModelRunner, so the AdapterSlots kernel
    wrappers are installed in the worker.
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
