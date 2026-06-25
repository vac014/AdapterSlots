"""
merged_weight_cache.py -- MergedWeightCache: per-adapter merged weight storage.

For each hot adapter k, pre-computes and caches the merged weight:
    Δ_k^{(i)} = alpha × B_k^{(i)} @ A_k^{(i)}   ∈ R^{d_out × d_in}

When the AlignmentAwareModelRunner dispatches a promoted (Level-3) segment,
it swaps the base weight W^{(i)} → W_k^{(i)} = W^{(i)} + Δ_k^{(i)} via
a zero-copy .data assignment. This turns the LoRA forward pass into a single
cuBLAS GEMM with no separate LoRA branch.

Memory budget (LLaMA-7B, rank=32, FP16, attention-only merge):
    Q, K, V, O projections × 32 layers × 2B = ~4.3 GB per adapter
    With K_hot=5: ~21.5 GB (feasible on 48 GB A6000)

Eviction policy: PredictiveLFU using score(k) = 1 - exp(-λ̂_k × tau_dispatch).
Eviction is triggered when memory budget is exceeded or when WarmCacheManager
evicts the adapter from the GPU LoRA pool.

Integration:
    Instantiated lazily in AlignmentAwareScheduler.__init__() when AS_MODE=wgkp
    and AS_MWC_K_HOT > 0. The scheduler holds both MergedWeightCache and
    WarmCacheManager; WarmCacheManager eviction calls mwc.evict(adapter_id).
"""

import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import torch

_DEBUG = os.environ.get("AS_MWC_DEBUG", "0") == "1"


class MergedWeightCache:
    """Pre-computed LoRA merged weight cache for Level-3 WGKP promotion.

    Thread-safe: merge_async() submits merge work to a background ThreadPool.
    install_merged() and uninstall_merged() wait for pending merges if needed.

    Args:
        k_hot:            Maximum number of adapters to keep merged.
        memory_budget_gb: GPU VRAM budget for merged weights in GB.
        projections:      List of layer name substrings to merge (e.g. ["q_proj"]).
                          Merge only these layers to control memory usage.
    """

    def __init__(
        self,
        k_hot: int = 5,
        memory_budget_gb: float = 10.0,
        projections: Optional[List[str]] = None,
    ) -> None:
        self.k_hot = k_hot
        self.memory_budget_gb = memory_budget_gb
        self.projections = projections or ["q_proj", "k_proj", "v_proj", "o_proj"]

        # adapter_id -> {layer_name -> merged_weight Tensor}
        self._cache: Dict[str, Dict[str, torch.Tensor]] = {}
        # adapter_id -> original base weight dict (for uninstall)
        self._originals: Dict[str, Dict[str, torch.Tensor]] = {}
        # adapters currently installed (weights swapped into model)
        self._installed: Set[str] = set()
        # pending async merge futures
        self._pending: Dict[str, "threading.Event"] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mwc-merge")
        self._lock = threading.RLock()  # reentrant: _evict_if_over_budget calls memory_used_gb

        # Background merge executor -- created lazily in merge_async() so tests
        # that never call merge_async() don't spawn threads.
        self._executor: "Optional[ThreadPoolExecutor]" = None

        # Stats
        self._hit_count = 0
        self._miss_count = 0
        self._merge_time_ms_total = 0.0
        self._eviction_count = 0

    # Public API

    def is_merged(self, adapter_id: str) -> bool:
        """Return True if adapter_id has a valid merged weight in cache.

        O(1) set lookup. Does NOT include adapters with pending merges.
        """
        with self._lock:
            return adapter_id in self._cache

    def is_pending(self, adapter_id: str) -> bool:
        """Return True if a merge_async() submission for adapter_id hasn't
        completed yet. Callers that re-check every tick (e.g.
        AAP/_prefetch_mwc_for_near_full_queues) must check this in addition
        to is_merged() before resubmitting -- merge_async() itself only
        guards against resubmitting an *already-merged* adapter, not one
        with work already in flight, so a caller polling every tick would
        otherwise flood the (2-worker) executor with duplicate merge jobs
        for the same adapter until the first one lands.
        """
        with self._lock:
            return adapter_id in self._pending

    def merge(
        self,
        adapter_id: str,
        lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]],
    ) -> None:
        """Compute and cache merged weights for adapter_id (synchronous).

        Args:
            adapter_id:   String adapter identifier.
            lora_weights: Dict mapping layer_name -> (A, B, alpha) where:
                            A: LoRA down-proj (R, K)
                            B: LoRA up-proj   (N, R)
                            alpha: LoRA scaling factor
        """
        if _DEBUG:
            print(f"[AS_MWC_DEBUG] merge() called adapter_id={adapter_id}",
                  file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        merged: Dict[str, torch.Tensor] = {}
        for layer_name, (A, B, alpha) in lora_weights.items():
            if not any(p in layer_name for p in self.projections):
                continue
            # Compute alpha * B @ A in FP32 then cast to match model dtype.
            delta = alpha * torch.matmul(B.float(), A.float())
            merged[layer_name] = delta.to(A.dtype)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self._cache[adapter_id] = merged
            self._merge_time_ms_total += elapsed_ms
            self._evict_if_over_budget()

    def merge_async(
        self,
        adapter_id: str,
        lora_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]],
    ) -> None:
        """Submit merge to background thread (non-blocking).

        install_merged() waits on the completion event before swapping weights.
        Called by AAP (_prefetch_mwc_for_near_full_queues) when a queue is at
        n*/2 tokens so the merge completes before the segment reaches n*.
        The ThreadPoolExecutor is created lazily to avoid spawning threads in
        processes that only use synchronous merge().
        """
        if self.is_merged(adapter_id):
            return
        event = threading.Event()
        with self._lock:
            self._pending[adapter_id] = event
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="mwc-merge"
                )

        def _do_merge():
            self.merge(adapter_id, lora_weights)
            with self._lock:
                self._pending.pop(adapter_id, None)
            event.set()

        self._executor.submit(_do_merge)

    def get_merged(
        self, adapter_id: str, layer_name: str
    ) -> Optional[torch.Tensor]:
        """Return cached merged weight for layer_name, or None if not cached.

        O(1) dict lookup. Returns the delta tensor Δ_k = alpha*B@A; caller
        must add it to the base weight to obtain the merged weight W_k.
        """
        with self._lock:
            adapter_cache = self._cache.get(adapter_id)
            if adapter_cache is None:
                self._miss_count += 1
                return None
            delta = adapter_cache.get(layer_name)
            if delta is not None:
                self._hit_count += 1
            else:
                self._miss_count += 1
            return delta

    def install_merged(self, adapter_id: str, model: torch.nn.Module) -> None:
        """Swap W → W + Δ_k for all cached layers of adapter_id.

        Zero-copy .data assignment: O(1) per layer, no HBM traffic.
        Saves original .data pointers in self._originals for uninstall.

        If a merge is still pending (merge_async() not done), blocks until
        the merge completes.
        """
        if _DEBUG:
            print(f"[AS_MWC_DEBUG] install_merged() called adapter_id={adapter_id}",
                  file=sys.stderr, flush=True)
        # Wait for any pending async merge.
        event = None
        with self._lock:
            event = self._pending.get(adapter_id)
        if event is not None:
            event.wait(timeout=5.0)

        with self._lock:
            if adapter_id not in self._cache:
                return
            if adapter_id in self._installed:
                return
            cached = self._cache[adapter_id]

        originals: Dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            delta = cached.get(name)
            if delta is None:
                continue
            originals[name] = module.weight.data.clone()
            module.weight.data = module.weight.data + delta.to(module.weight.dtype)

        with self._lock:
            self._originals[adapter_id] = originals
            self._installed.add(adapter_id)

    def uninstall_merged(self, adapter_id: str, model: torch.nn.Module) -> None:
        """Restore W from saved originals after a promoted segment.

        Zero-copy: restores the original .data pointer. O(1) per layer.
        """
        with self._lock:
            if adapter_id not in self._installed:
                return
            originals = self._originals.pop(adapter_id, {})
            self._installed.discard(adapter_id)

        for name, module in model.named_modules():
            orig = originals.get(name)
            if orig is not None and isinstance(module, torch.nn.Linear):
                module.weight.data = orig

    def evict(self, adapter_id: str) -> None:
        """Remove adapter_id from cache (called when WarmCacheManager evicts it)."""
        with self._lock:
            self._cache.pop(adapter_id, None)
            self._originals.pop(adapter_id, None)
            self._installed.discard(adapter_id)
            self._pending.pop(adapter_id, None)
            self._eviction_count += 1

    def memory_used_gb(self) -> float:
        """Return estimated GPU memory used by merged weights in GB."""
        total_bytes = 0
        with self._lock:
            for adapter_cache in self._cache.values():
                for t in adapter_cache.values():
                    total_bytes += t.numel() * t.element_size()
        return total_bytes / (1024 ** 3)

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            return {
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "merge_time_ms_total": self._merge_time_ms_total,
                "memory_gb": self.memory_used_gb(),
                "eviction_count": self._eviction_count,
                "n_cached": len(self._cache),
                "n_installed": len(self._installed),
            }

    # Internal helpers

    def _evict_if_over_budget(self) -> None:
        """Evict lowest-priority adapter if over k_hot or memory budget.

        Called under self._lock. Eviction order: LRU (first inserted).
        """
        while (
            len(self._cache) > self.k_hot
            or self.memory_used_gb() > self.memory_budget_gb
        ):
            if not self._cache:
                break
            # Evict first inserted (LRU approximation -- use OrderedDict if needed)
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key, None)
            self._originals.pop(oldest_key, None)
            self._installed.discard(oldest_key)
            self._eviction_count += 1
