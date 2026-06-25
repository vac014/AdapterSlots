"""
baselines.py -- Simple dispatch baselines for E8-bandit comparison (whittle_scheduler, §6).

Two baselines evaluated on Single A6000 (not required for multi-GPU experiments):
    FIFODispatcher   -- First-In-First-Out across all adapters (ignores fill state)
    GreedyFillDispatcher -- Always dispatch the adapter with the highest fill fraction

These baselines establish the lower end of the WAR spectrum:
    FIFO < Greedy ≤ Threshold ≤ Whittle ≤ Oracle

Multi-GPU note: FIFO and Greedy are only tested on single A6000.  Multi-GPU experiments
(§5.5, §5.6) compare Threshold vs. Whittle only, since the goal there is validating
TP-transparency and cross-hardware near-optimality.

References:
    - whittle_scheduler.md §6
"""

from typing import Dict, List, Optional


class FIFODispatcher:
    """First-In-First-Out dispatch: serve adapters in round-robin order.

    Ignores fill state and arrival rates entirely.  Provides a lower-bound
    WAR reference to show that any intelligent dispatch policy outperforms naive FIFO.

    Args:
        adapters:  Ordered list of adapter IDs.
    """

    def __init__(self, adapters: List[str]) -> None:
        self.adapters = list(adapters)
        self._cursor = 0

    def rank_adapters(
        self,
        fill_fracs: Optional[Dict[str, float]] = None,
        lambda_est: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """Return adapters in round-robin order starting from current cursor.

        Args:
            fill_fracs: Ignored (FIFO does not use fill state).
            lambda_est: Ignored (FIFO does not use arrival rates).

        Returns:
            Adapter list rotated so the next-in-FIFO-order comes first.
        """
        n = len(self.adapters)
        rotated = self.adapters[self._cursor:] + self.adapters[:self._cursor]
        self._cursor = (self._cursor + 1) % n
        return rotated


class GreedyFillDispatcher:
    """Greedy-fill dispatch: always dispatch the adapter with the highest fill fraction.

    Purely reactive to current fill state.  No forward-looking fill-probability
    estimation (unlike Whittle).  Provides a reference to show that Whittle's
    fill-probability look-ahead adds value beyond simple greedy dispatch.

    Args:
        adapters:  List of adapter IDs.
    """

    def __init__(self, adapters: List[str]) -> None:
        self.adapters = list(adapters)

    def rank_adapters(
        self,
        fill_fracs: Dict[str, float],
        lambda_est: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """Return adapters sorted by current fill fraction (highest first).

        Args:
            fill_fracs:  s_k = |Q_k| / W for each adapter.
            lambda_est:  Ignored (Greedy does not use arrival rates).

        Returns:
            Adapter list sorted by fill fraction descending.
        """
        return sorted(
            self.adapters,
            key=lambda k: fill_fracs.get(k, 0.0),
            reverse=True,
        )
