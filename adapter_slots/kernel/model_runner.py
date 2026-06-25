"""
model_runner.py -- AlignmentAwareModelRunner: vLLM ModelRunner subclass (kernel_promotion Phase 8).

This subclass overrides execute_model() to implement the three-level WGKP kernel
dispatch hierarchy for promoted adapter segments:

    Level 3 (WGKP):  cuBLAS GEMM with merged weight W_k = W + alpha*B@A
                      (zero-copy .data swap, no LoRA branch at all)
    Level 2 (Fused): FusedLoRAKernel -- fused Triton base+LoRA (AS_FUSED_KERNEL=1)
    Level 1 (SGMV):  Standard vLLM SGMV path (fallback; always correct)

Level 2 is wired in for real via load_model() below (see
fused_lora_layers.py) -- it reassigns each installed LoRA layer's __class__
to a subclass that overrides apply() to call FusedLoRAKernel directly,
instead of vLLM's stock two-launch SGMV path. Level 3 is not (see
execute_model()'s docstring for why it's a measured dead end).

Integration:
    On vllm==0.6.3, which this repo pins, there is no --worker-cls flag: the runner is
    wired in through Worker's model_runner_cls keyword by _AlignmentAwareGPUExecutorAsync
    (integrations/aligned_engine.py), which is what the AS servers use. AlignmentAwareWorker
    below is the equivalent entry point for vLLM builds that do expose --worker-cls:
        --worker-cls adapter_slots.kernel.model_runner.AlignmentAwareWorker

    Without either, the scheduler-only mode still works (no kernel promotion):
        export AS_MODE=wgkp AS_MWC_K_HOT=0 AS_FUSED_KERNEL=0

    The merged weight swap (Level 3) requires the MergedWeightCache instance
    from the scheduler. It is shared via a module-level singleton set by
    AlignmentAwareScheduler.__init__():
        adapter_slots.kernel.model_runner._mwc_singleton = mwc_instance

No monkey patching: all vLLM interaction is via standard subclassing of
ModelRunner and GPUWorker. The weight swap uses .data assignment -- a standard
PyTorch in-place operation, not a monkey patch.

TP=2 correctness: APIS has no tensor parallelism (each GPU runs independently),
so AlignmentAwareModelRunner can be used for each GPU's independent instance.
For TP=2 with merged weights, sharding correctness is validated in
results/kernel_promotion/e13_7_mwc:
    W_k[local_rank] = W[local_rank] + (alpha*B@A)[local_rank]
    since alpha*B@A is a linear operation that shards along the same axis as W.
"""

import os
import sys

# Module-level MergedWeightCache singleton.
# Set by AlignmentAwareScheduler.__init__() when AS_MODE=wgkp and AS_MWC_K_HOT>0.
_mwc_singleton = None  # type: ignore[assignment]

# vLLM import guard
try:
    from vllm.worker.model_runner import ModelRunner
    from vllm.worker.worker import Worker
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

    class ModelRunner:  # type: ignore[no-redef]
        def execute_model(self, *args, **kwargs):
            raise NotImplementedError("vLLM not installed")

    class Worker:  # type: ignore[no-redef]
        pass


class AlignmentAwareModelRunner(ModelRunner):
    """vLLM ModelRunner subclass implementing the WGKP Level-3 kernel path.

    On each execute_model() call:
    1. Extract wgkp_segments from execute_model_req.metadata.
    2. For promoted segments, swap W → W_k via MergedWeightCache.install_merged().
    3. Call super().execute_model() (uses whatever kernel is wired for LoRA).
    4. Restore all swapped weights via MergedWeightCache.uninstall_merged().

    The weight swap is zero-copy (O(1) .data pointer replacement) and has
    negligible overhead relative to the GEMM latency it avoids.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wgkp_enabled = os.environ.get("AS_MODE") == "wgkp"
        self._log_wgkp = os.environ.get("AS_WGKP_LOG", "0") == "1"
        if self._wgkp_enabled:
            print(
                f"[AS] AlignmentAwareModelRunner constructed "
                f"(mwc singleton set later by scheduler init) "
                f"fused_kernel={os.environ.get('AS_FUSED_KERNEL', '0')}",
                file=sys.stderr, flush=True,
            )

    @property
    def _mwc(self):
        """Read the module-level singleton dynamically, not at __init__ time.

        vLLM's engine constructs the executor/worker/model_runner BEFORE
        AlignmentAwareScheduler (see aligned_engine.py's post-super()
        scheduler rebuild) -- AlignmentAwareScheduler.__init__() is what
        calls set_mwc(). Capturing _mwc_singleton once in __init__ would
        permanently freeze it at None, silently disabling promotion for
        the whole process. A property re-reads the current global on every
        access, picking up the real instance once the scheduler runs.
        """
        return _mwc_singleton

    def load_model(self) -> None:
        """Load the model, then (WGKP mode only) wire in the graph-safe
        packed-nslice fused kernel in place of vLLM's stock per-slice
        add_lora_packed_nslice loop.

        Must run after super().load_model() returns: that call is what
        installs the stock LoRA layer instances and PunicaWrapper in the
        first place (via vLLM's own LoRAModelManager, inside load_model()
        itself) -- there is nothing to swap before it returns.

        What must NOT be wired here is install_fused_lora_layers()
        (fused_lora_layers.py's per-layer __class__ swap onto FusedLoRAKernel,
        which fuses the base GEMM too, not just the LoRA branch). Its Python-side
        contiguous-run dispatch has a data-dependent launch count, and vLLM's
        capture_model() cannot capture that, so the graph is skipped for the whole
        model rather than for the LoRA layers alone: running it needs
        --enforce-eager, which measures 3.5x-10x SLOWER end-to-end than vanilla
        vLLM and worsens with K, because losing graph-replay amortization across
        ~40 layers dwarfs anything 2-launch-vs-1 fusion can save. Both files are
        kept in the tree for reference and are not installed.

        install_fused_punica_wrapper() (fused_punica_wrapper.py) is the
        replacement: it only touches the LoRA-only shrink/expand GEMV
        kernels (never the base GEMM, which stays on cuBLAS), using the
        same fixed-grid/on-device-index-load pattern vLLM's own bgmv
        kernels use -- i.e. it is captured into the CUDA graph exactly like
        the stock path, no --enforce-eager required.
        """
        super().load_model()
        if self._wgkp_enabled:
            from adapter_slots.kernel.fused_punica_wrapper import (
                install_fused_punica_wrapper,
            )
            installed = install_fused_punica_wrapper(self.lora_manager)
            print(
                f"[AS] install_fused_punica_wrapper: installed={installed} "
                f"(fused_kernel={os.environ.get('AS_FUSED_KERNEL', '0')})",
                file=sys.stderr, flush=True,
            )

    def execute_model(self, *args, **kwargs):
        """Override: Level-3 (dense weight-merge) promotion is DISABLED.

        The signature must stay *args/**kwargs rather than naming
        execute_model_req: vLLM 0.6.3's real call site
        (vllm/worker/worker_base.py:327) invokes this with keyword args
        (model_input=, kv_caches=, intermediate_tensors=, num_steps=) and never
        with a positional execute_model_req, so declaring one raises TypeError:
        missing 1 required positional argument on every request.

        Why the promotion is disabled: merging is O(K) where every competing
        kernel is ~O(1) in adapter count. Dense weight-merge promotion means one
        separate transformer forward pass per promoted adapter (every layer,
        attention, MLP, sampling, not just one GEMM) plus a per-layer weight
        .clone()+add in MergedWeightCache.install_merged(), measured at 6x-105x
        slower than real Punica/S-LoRA/vLLM BGMV kernels at K=16. Pre-merging
        LoRA's low-rank A/B factors into a dense delta destroys the small-rank
        structure those O(1) kernels depend on, and there is no way to recover
        O(1) while still merging: feeding a merged dense weight into vLLM's own
        bgmv_expand kernel is 100-1000x SLOWER, because that kernel assumes the
        indexed dimension is rank-sized, not hidden-dim-sized.

        Promoted segments therefore fall through to vLLM's own LoRA path (Level
        1, SGMV/BGMV, already O(1) in adapter count) via the unmodified
        super().execute_model() call below. Only the dense-merge *kernel
        dispatch* consequence is disabled: WGKP's promotion classification in
        form_batch_wgkp()/segment_and_promote() still runs and is still measured
        (promotion_fraction, WAR/GWAR).
        """
        return super().execute_model(*args, **kwargs)


class AlignmentAwareWorker(Worker):
    """vLLM Worker subclass that installs AlignmentAwareModelRunner.

    Overrides _init_model_runner() to substitute AlignmentAwareModelRunner for the
    standard ModelRunner, for vLLM builds that both expose a --worker-cls flag and
    call _init_model_runner().

    vllm==0.6.3, which this repo pins, is not such a build: it has no --worker-cls
    flag, vllm/executor/gpu_executor.py hardcodes worker selection in
    _get_worker_module_and_class(), and vllm/worker/worker.py never calls
    _init_model_runner(). On 0.6.3 the model runner is wired through Worker's
    model_runner_cls keyword instead, by _AlignmentAwareGPUExecutorAsync in
    integrations/aligned_engine.py. Either way it is subclassing, not monkeypatching.
    """

    def _init_model_runner(self, *args, **kwargs):
        if not _VLLM_AVAILABLE:
            return super()._init_model_runner(*args, **kwargs)
        runner = AlignmentAwareModelRunner(*args, **kwargs)
        self.model_runner = runner
        return runner


def set_mwc(mwc) -> None:
    """Register the MergedWeightCache singleton for this process.

    Called by AlignmentAwareScheduler.__init__() when AS_MODE=wgkp and
    AS_MWC_K_HOT > 0. The same MWC instance is shared between the scheduler
    (for merge_async / evict) and the ModelRunner (for install/uninstall).
    """
    global _mwc_singleton
    _mwc_singleton = mwc
