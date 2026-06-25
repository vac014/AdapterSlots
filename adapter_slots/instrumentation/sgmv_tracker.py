"""
sgmv_tracker.py -- SGMV operational intensity tracker.

Tracks tokens_per_adapter_per_dispatch = Σ(tokens_k) / dispatches_k.
This is the primary hardware-grounded metric for the E1 narrative:
higher intensity → SGMV kernel operates in a more compute-bound regime,
reducing the sensitivity to warp alignment.

Plugs into the serving script at the SGMV call site:
    tracker = SGMVIntensityTracker()
    # inside the forward pass, at the SGMV dispatch:
    tracker.record_dispatch(adapter_id=aid, n_tokens=n)
    # after the experiment:
    intensities = tracker.compute_intensity()
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class SGMVIntensityTracker:
    """
    Tracks per-adapter SGMV dispatch statistics.

    The roofline-relevant quantity is tokens_per_dispatch: how many tokens
    are processed per SGMV kernel launch for each adapter. Low intensity
    (< tile size) means the kernel is latency-bound; high intensity means
    compute-bound.
    """

    def __init__(self):
        self.dispatch_counts: Dict[str, int] = defaultdict(int)
        self.token_counts: Dict[str, int] = defaultdict(int)
        self._history: List[Tuple[str, int]] = []   # (adapter_id, n_tokens)

    def record_dispatch(self, adapter_id: str, n_tokens: int) -> None:
        """
        Record one SGMV kernel launch.

        Args:
            adapter_id: Adapter identifier (string key or str(int)).
            n_tokens:   Number of tokens processed in this dispatch.
        """
        key = str(adapter_id)
        self.dispatch_counts[key] += 1
        self.token_counts[key] += n_tokens
        self._history.append((key, n_tokens))

    def compute_intensity(self) -> Dict[str, float]:
        """
        Compute mean tokens-per-dispatch for each adapter.

        Returns:
            Dict mapping adapter_id -> mean tokens/dispatch.
        """
        return {
            k: self.token_counts[k] / max(self.dispatch_counts[k], 1)
            for k in self.token_counts
        }

    def compute_global_intensity(self) -> float:
        """
        Mean tokens-per-dispatch averaged across all adapters.
        """
        intensities = self.compute_intensity()
        if not intensities:
            return 0.0
        return sum(intensities.values()) / len(intensities)

    def total_tokens(self) -> int:
        return sum(self.token_counts.values())

    def total_dispatches(self) -> int:
        return sum(self.dispatch_counts.values())

    def reset(self) -> None:
        self.dispatch_counts.clear()
        self.token_counts.clear()
        self._history.clear()

    def summary(self) -> Dict:
        intensities = self.compute_intensity()
        return {
            "per_adapter_intensity": intensities,
            "global_intensity": self.compute_global_intensity(),
            "total_tokens": self.total_tokens(),
            "total_dispatches": self.total_dispatches(),
            "n_adapters": len(self.dispatch_counts),
        }

    def to_jsonl_record(self, tick_id: Optional[int] = None) -> Dict:
        """Return a dict suitable for JSONL logging."""
        record = self.summary()
        if tick_id is not None:
            record["tick_id"] = tick_id
        return record
