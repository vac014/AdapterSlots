"""
apt.py -- AdaptivePromoThreshold (APT): runtime n* selection (kernel_promotion Phase 6).

APT dynamically selects the promotion threshold n* based on:
1. The measured crossover curve from E13.6 (hardware profile JSON).
2. The current mean queue depth from AlignmentBuffer.pending_count().

It picks the smallest n where GEMM is empirically faster than SGMV AND the
mean queue depth suggests tokens will accumulate to n* before T_max expires.

Update interval: every 100 scheduling ticks (using a rolling mean of queue depth).
This avoids thrashing n* on every tick while remaining responsive to workload shifts.

Profile JSON format (produced by scripts/experiments/e13_crossover_benchmark.py):
    {
      "hardware": "a6000_tp1",
      "warp_size": 32,
      "gemm_crossover_n": 16,
      "crossover_curve": {
        "8":  {"psi_fuse": 1.15, "psi_gemm": 1.25, "gemm_faster": true},
        "16": {"psi_fuse": 1.28, "psi_gemm": 1.45, "gemm_faster": true},
        ...
      }
    }
"""

import json
import os
import pathlib
from typing import Dict, List, Optional


# Default profile path (relative to this file's package root)
_DEFAULT_PROFILE_A6000_TP1 = str(
    pathlib.Path(__file__).parent / "hw_profiles" / "a6000_tp1.json"
)
_DEFAULT_PROFILE_A6000_TP2 = str(
    pathlib.Path(__file__).parent / "hw_profiles" / "a6000_tp2.json"
)


class AdaptivePromoThreshold:
    """Runtime n* selection from E13.6 hardware profile and queue depth.

    Usage:
        apt = AdaptivePromoThreshold(hw_profile_path="hw_profiles/a6000_tp1.json")
        ...
        apt.update(mean_queue_depth=12.5)   # called every 100 ticks
        n_star = apt.current_threshold()     # use for CASH dispatch
    """

    def __init__(
        self,
        hw_profile_path: str = "",
        min_speedup: float = 1.05,
        update_interval: int = 100,
        fallback_threshold: int = 32,
    ) -> None:
        """
        Args:
            hw_profile_path:   Path to E13.6 JSON profile. Falls back to a6000_tp1.
            min_speedup:       Minimum ψ_gemm to qualify a threshold as promotable.
            update_interval:   Number of scheduling ticks between threshold updates.
            fallback_threshold: n* when no hardware profile is found (= warp size).
        """
        self.min_speedup = min_speedup
        self.update_interval = update_interval
        self.fallback_threshold = fallback_threshold

        # Load hardware profile
        self._profile: dict = {}
        self._crossover_curve: Dict[int, dict] = {}
        self._warp_size: int = 32
        self._load_profile(hw_profile_path)

        # Sorted list of (n, profile_entry) pairs for threshold selection
        self._sorted_thresholds: List[tuple] = sorted(
            (
                (int(n), entry)
                for n, entry in self._crossover_curve.items()
                if entry.get("gemm_faster", False)
                and entry.get("psi_gemm", 0.0) >= self.min_speedup
            ),
            key=lambda x: x[0],
        )

        # Current state
        self._current_threshold: int = self._initial_threshold()
        self._tick_count: int = 0
        self._queue_depth_history: List[float] = []

    def update(self, mean_queue_depth: float) -> int:
        """Update threshold based on mean queue depth; called every tick.

        Returns the new threshold (same as current_threshold()).
        Internal update fires every update_interval ticks.
        """
        self._tick_count += 1
        self._queue_depth_history.append(mean_queue_depth)

        if self._tick_count >= self.update_interval:
            rolling_mean = sum(self._queue_depth_history) / len(self._queue_depth_history)
            self._current_threshold = self._select_threshold(rolling_mean)
            self._tick_count = 0
            self._queue_depth_history = []

        return self._current_threshold

    def current_threshold(self) -> int:
        """Return current promotion threshold n*."""
        return self._current_threshold

    def hardware(self) -> str:
        """Return hardware identifier from loaded profile."""
        return self._profile.get("hardware", "unknown")

    # Internal

    def _load_profile(self, hw_profile_path: str) -> None:
        """Load JSON hardware profile.

        Search order:
          1. hw_profile_path argument (if non-empty)
          2. AS_WGKP_HW_PROFILE env var (if set and non-empty)
          3. Package default: a6000_tp1.json

        If hw_profile_path is provided but does not exist, no fallback is attempted
        (the caller explicitly named a file; missing file = configuration error).
        If hw_profile_path is empty, env var and default profiles are tried in order.
        """
        # Strict mode: if an explicit path was given, only try that path.
        if hw_profile_path:
            try:
                with open(hw_profile_path) as f:
                    self._profile = json.load(f)
                raw_curve = self._profile.get("crossover_curve", {})
                self._crossover_curve = {int(k): v for k, v in raw_curve.items()}
                self._warp_size = self._profile.get("warp_size", 32)
                return
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                pass
            # Explicit path given but failed -- use empty curve (fallback_threshold).
            self._profile = {"hardware": "fallback"}
            self._crossover_curve = {}
            self._warp_size = 32
            return

        # No explicit path: try env var then package default.
        candidates = [
            os.environ.get("AS_WGKP_HW_PROFILE", ""),
            _DEFAULT_PROFILE_A6000_TP1,
        ]
        for path in candidates:
            if not path:
                continue
            try:
                with open(path) as f:
                    self._profile = json.load(f)
                raw_curve = self._profile.get("crossover_curve", {})
                self._crossover_curve = {int(k): v for k, v in raw_curve.items()}
                self._warp_size = self._profile.get("warp_size", 32)
                return
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                continue
        # No profile found -- use empty curve (fallback_threshold applies).
        self._profile = {"hardware": "fallback"}
        self._crossover_curve = {}
        self._warp_size = 32

    def _initial_threshold(self) -> int:
        """Initial threshold: smallest promotable n* or fallback_threshold."""
        if self._sorted_thresholds:
            return self._sorted_thresholds[0][0]
        return self.fallback_threshold

    def _select_threshold(self, mean_queue_depth: float) -> int:
        """Select n* based on mean queue depth.

        Returns the largest promotable n* that is <= mean_queue_depth.
        This ensures segments will likely reach n* before T_max expires.
        Falls back to fallback_threshold if no entry qualifies.
        """
        selected = None
        for n, entry in self._sorted_thresholds:
            if n <= mean_queue_depth:
                selected = n
            else:
                break
        return selected if selected is not None else self.fallback_threshold
