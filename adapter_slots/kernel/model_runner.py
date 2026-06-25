"""
model_runner.py -- AlignmentAwareModelRunner: vLLM ModelRunner subclass.

In WGKP mode with AS_SEGGEMM=1, 2 or 3, load_model() replaces vLLM's
PunicaWrapper with the grouped LoRA kernels over the adapter segments
(segmented_punica_wrapper.py), so the LoRA projections run on the
adapter-aligned batches the scheduler produces. Otherwise vLLM's own LoRA
kernels run.

With AS_MWC_K_HOT > 0 (and tensor parallelism 1) it also installs WGKP L3
(l3.py): a step whose tokens all belong to one adapter with merged weights
resident, at least n* of them, runs the attention projections on the merged
weights W_k = W + alpha B_k A_k.

On vLLM 0.6.3 the runner is passed through Worker's model_runner_cls keyword by
the executors in integrations/aligned_engine.py and aligned_mp_engine.py.
"""

import logging
import os

from adapter_slots.kernel import l3

logger = logging.getLogger(__name__)

try:
    from vllm.worker.model_runner import ModelRunner
except ImportError:
    class ModelRunner:  # type: ignore[no-redef]
        def execute_model(self, *args, **kwargs):
            raise NotImplementedError("vLLM not installed")


class AlignmentAwareModelRunner(ModelRunner):
    """vLLM ModelRunner with the AdapterSlots LoRA kernels and WGKP L3."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wgkp_enabled = os.environ.get("AS_MODE") == "wgkp"
        self._n_star = int(os.environ.get("AS_WGKP_THRESHOLD", "8"))
        self._l3_layers: dict = {}
        self.n_promoted_steps = 0

    def load_model(self) -> None:
        """Load the model, then in WGKP mode install the grouped LoRA kernels
        (AS_SEGGEMM set) and L3 (AS_MWC_K_HOT > 0, tensor parallelism 1).

        vLLM's load_model() creates the LoRA layers and the shared PunicaWrapper,
        so the swap happens after it returns. The grouped kernels replace only the
        LoRA shrink and expand kernels (the base GEMM stays on cuBLAS) and launch
        on fixed grids with on-device index loads, so decode steps are still
        captured in CUDA graphs and no --enforce-eager is needed.
        """
        super().load_model()
        if not self._wgkp_enabled:
            return
        if os.environ.get("AS_SEGGEMM", "0") in ("1", "2", "3"):
            from adapter_slots.kernel.segmented_punica_wrapper import (
                install_segmented_punica_wrapper,
            )
            seg = install_segmented_punica_wrapper(self.lora_manager)
            logger.info(f"install_segmented_punica_wrapper: installed={seg} "
                        f"mode={os.environ.get('AS_SEGGEMM')} "
                        f"min_tokens={os.environ.get('AS_SEGGEMM_MIN_TOKENS', '192')}")
        k_hot = int(os.environ.get("AS_MWC_K_HOT", "5"))
        if k_hot > 0 and self.parallel_config.tensor_parallel_size == 1:
            self._l3_layers = l3.install(self.model)
            if self._l3_layers:
                l3.set_mwc(l3.MergedWeightCache(k_hot))
        logger.info(f"L3: {len(self._l3_layers)} promotable attention projections, "
                    f"AS_MWC_K_HOT={k_hot}")

    def execute_model(self, *args, **kwargs):
        """Run one step, on the merged weights when the step is promoted.

        vLLM 0.6.3 calls this with keyword arguments (model_input, kv_caches,
        intermediate_tensors, num_steps), so the signature stays *args/**kwargs.
        """
        mwc = l3.get_mwc()
        if mwc is None or not self._l3_layers:
            return super().execute_model(*args, **kwargs)
        mwc.process(self._l3_layers)
        model_input = kwargs["model_input"] if "model_input" in kwargs else args[0]
        weights = self._promoted_weights(model_input, mwc)
        if weights is None:
            return super().execute_model(*args, **kwargs)
        self.n_promoted_steps += 1
        if self.n_promoted_steps in (1, 10, 100) or self.n_promoted_steps % 1000 == 0:
            logger.info(f"L3: {self.n_promoted_steps} promoted steps")
        with l3.promoted(weights):
            return super().execute_model(*args, **kwargs)

    def _promoted_weights(self, model_input, mwc):
        """Merged weights for a decode step that runs without a CUDA graph and
        whose tokens all belong to one adapter with merged weights resident, at
        least n* of them; None otherwise."""
        attn = model_input.attn_metadata
        if attn is None or attn.prefill_metadata is not None:
            return None
        decode = attn.decode_metadata
        if decode is None or decode.use_cuda_graph or model_input.lora_mapping is None:
            return None
        lora_id = l3.promotable_adapter(list(model_input.lora_mapping.index_mapping), self._n_star)
        return None if lora_id is None else mwc.get(lora_id)
