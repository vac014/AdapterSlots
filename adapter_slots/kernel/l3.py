"""
l3.py -- WGKP L3: merged-weight attention projections (Sec. III).

The MergedWeightCache (MWC) holds, for up to K_hot adapters, the merged weights

    W_k = W + alpha * B_k A_k

of the four attention projections (the packed QKV projection and O) in every
layer. In a promoted step those projections compute x W_k^T in one GEMM and
skip their LoRA kernels; the MLP projections keep the LoRA path.

A step is promoted when the paper's gate holds for the whole step: it is a
decode step whose tokens all belong to one adapter, at least n* of them, and
that adapter's merged weights are resident. The merged weight then replaces the
base weight; in a mixed batch it would be a second full weight read for its
segment, more than the grouped LoRA kernels cost. Promoted steps cannot replay
a CUDA graph (the weights differ from the captured ones), so promotion is also
limited to steps that run without one.

K_hot is sized when the first merge is requested, after vLLM has allocated its
KV cache: as many adapters as fit in the free GPU memory, up to AS_MWC_K_HOT.
The scheduler requests merges (alignment-aware prefetch); the model runner
performs them between steps. Merged weights need the unsharded matrices, so L3
runs with tensor parallelism 1 only.
"""

import json
import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from vllm.lora.layers import MergedQKVParallelLinearWithLora, RowParallelLinearWithLoRA
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

    class MergedQKVParallelLinearWithLora:  # type: ignore[no-redef]
        pass

    class RowParallelLinearWithLoRA:  # type: ignore[no-redef]
        pass


# Merged weights of the step being executed, {layer name: W_k}; None outside promoted steps.
_ACTIVE: Optional[Dict[str, torch.Tensor]] = None


@contextmanager
def promoted(weights: Dict[str, torch.Tensor]):
    """Run the enclosed forward pass with the given merged weights."""
    global _ACTIVE
    _ACTIVE = weights
    try:
        yield
    finally:
        _ACTIVE = None


def _active_weight(layer) -> Optional[torch.Tensor]:
    return None if _ACTIVE is None else _ACTIVE.get(layer._l3_name)


class PromotableQKVWithLoRA(MergedQKVParallelLinearWithLora):
    """Packed QKV LoRA layer that uses the merged weight in a promoted step."""

    def apply(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        w = _active_weight(self)
        if w is not None:
            return F.linear(x, w, bias)
        return super().apply(x, bias)


class PromotableRowWithLoRA(RowParallelLinearWithLoRA):
    """O-projection LoRA layer that uses the merged weight in a promoted step."""

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        w = _active_weight(self)
        if w is not None:
            return F.linear(x, w)
        return super().apply(x)


def install(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """Make the attention projections of a LLaMA-style model promotable.

    Returns {layer name: module} for the packed QKV and O projections, or {} if
    the model does not have that structure.
    """
    if not _VLLM_AVAILABLE:
        return {}
    layers: Dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if type(module) is MergedQKVParallelLinearWithLora and name.endswith("qkv_proj"):
            module.__class__ = PromotableQKVWithLoRA
        elif type(module) is RowParallelLinearWithLoRA and name.endswith("o_proj"):
            module.__class__ = PromotableRowWithLoRA
        else:
            continue
        module._l3_name = name
        layers[name] = module
    return layers


def _layer_index(name: str) -> int:
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


class MergedWeightCache:
    """Merged attention weights for up to K_hot adapters, least recently used evicted.

    Args:
        k_hot_max:  Upper bound on resident adapters (AS_MWC_K_HOT).
        margin_gb:  GPU memory left free when sizing the cache.
    """

    def __init__(self, k_hot_max: int, margin_gb: float = 1.0) -> None:
        self.k_hot_max = k_hot_max
        self.margin_bytes = int(margin_gb * (1 << 30))
        self.k_hot: Optional[int] = None
        self._weights: "OrderedDict[int, Dict[str, torch.Tensor]]" = OrderedDict()
        self._loaded: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._requested: set = set()
        self._lock = threading.Lock()
        self.n_merges = 0

    # Scheduler side

    def request(self, lora_id: int, lora_path: str) -> None:
        """Ask for adapter lora_id to be merged; its LoRA tensors load in the background."""
        with self._lock:
            if self.k_hot == 0 or lora_id in self._weights or lora_id in self._requested:
                return
            self._requested.add(lora_id)
        threading.Thread(target=self._load, args=(lora_id, lora_path), daemon=True).start()

    def is_merged(self, lora_id: int) -> bool:
        with self._lock:
            return lora_id in self._weights

    def _load(self, lora_id: int, lora_path: str) -> None:
        """Read the attention LoRA tensors of one adapter from its PEFT checkpoint (CPU)."""
        from safetensors import safe_open

        path = Path(lora_path)
        cfg = json.loads((path / "adapter_config.json").read_text())
        scaling = cfg.get("lora_alpha", cfg["r"]) / cfg["r"]
        tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        with safe_open(str(path / "adapter_model.safetensors"), framework="pt") as f:
            for key in f.keys():
                if not key.endswith(".lora_A.weight"):
                    continue
                proj = key[: -len(".lora_A.weight")]
                if proj.split(".")[-1] not in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    continue
                a = f.get_tensor(key)
                b = f.get_tensor(proj + ".lora_B.weight") * scaling
                tensors[f"{_layer_index(proj)}.{proj.split('.')[-1]}"] = (a, b)
        with self._lock:
            self._loaded[lora_id] = tensors

    # Model runner side

    def _size(self, layers: Dict[str, torch.nn.Module]) -> None:
        per_adapter = sum(_base_weight(m).numel() * _base_weight(m).element_size()
                          for m in layers.values())
        free, _ = torch.cuda.mem_get_info()
        fit = max(0, (free - self.margin_bytes) // per_adapter)
        self.k_hot = int(min(self.k_hot_max, fit))
        logger.info(f"L3 merged-weight cache: {self.k_hot} adapter(s) resident at most "
                    f"(AS_MWC_K_HOT={self.k_hot_max}, {per_adapter / 2**30:.2f} GiB per adapter, "
                    f"{free / 2**30:.2f} GiB free)")

    def process(self, layers: Dict[str, torch.nn.Module]) -> None:
        """Between steps: size the cache on first use and merge one loaded adapter."""
        with self._lock:
            if not self._loaded:
                return
            lora_id, tensors = self._loaded.popitem()
        if self.k_hot is None:
            self._size(layers)
        if self.k_hot == 0:
            with self._lock:
                self._requested.clear()
                self._loaded.clear()
            return
        merged = {}
        for name, module in layers.items():
            i = _layer_index(name)
            w = _base_weight(module)
            if name.endswith("qkv_proj"):
                deltas = [tensors[f"{i}.{p}"] for p in ("q_proj", "k_proj", "v_proj")]
            else:
                deltas = [tensors[f"{i}.o_proj"]]
            delta = torch.cat([(b.to(w.device, torch.float32) @ a.to(w.device, torch.float32))
                               for a, b in deltas], dim=0)
            merged[name] = (w.float() + delta).to(w.dtype)
        with self._lock:
            while len(self._weights) >= self.k_hot:
                evicted, _ = self._weights.popitem(last=False)
                self._requested.discard(evicted)
            self._weights[lora_id] = merged
            self.n_merges += 1

    def get(self, lora_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Merged weights of lora_id (marked recently used), or None."""
        with self._lock:
            w = self._weights.get(lora_id)
            if w is not None:
                self._weights.move_to_end(lora_id)
            return w


def _base_weight(module) -> torch.Tensor:
    return module.base_layer.weight


_MWC: Optional[MergedWeightCache] = None


def set_mwc(mwc: Optional[MergedWeightCache]) -> None:
    """Register the process's merged-weight cache (shared by runner and scheduler)."""
    global _MWC
    _MWC = mwc


def get_mwc() -> Optional[MergedWeightCache]:
    return _MWC


def promotable_adapter(lora_ids: List[int], n_star: int) -> Optional[int]:
    """The adapter of a step whose tokens all belong to it, with at least n* of them."""
    if len(lora_ids) < n_star:
        return None
    first = lora_ids[0]
    if first <= 0 or any(i != first for i in lora_ids):
        return None
    return first
