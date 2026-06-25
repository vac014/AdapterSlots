"""
cash.py -- CASH dispatch: Whittle-inspired queue scores (Proposition 1).

Each tick every adapter queue k gets the score

    I_k(s_k) = lambda_k * s_k * P_k(s_k)

and queues are dispatched in decreasing score order:

    lambda_k  the EWMA estimate of the rate at which tokens enter Q_k
    s_k       fill fraction |Q_k| / W
    P_k(s_k)  probability that the queue fills before its deadline: under
              Poisson arrivals the W - |Q_k| missing tokens take an
              Erlang(W - |Q_k|, lambda_k) time, so
              P_k = F_{W - |Q_k|, lambda_k}(T_k - age(Q_k)),
              with T_k = min(T*(k), T_SLO) the queue's deadline

The score is a closed-form stand-in for the Whittle index of the per-arm
restless-bandit problem, not the exact index. Scores cost O(K) and the sort
O(K log K) per tick; the computation runs in the scheduler process.
"""

from typing import Dict, List

import numpy as np
from scipy.special import gammainc


class CASHDispatcher:
    """CASH scores and dispatch order for the adapter queues.

    Args:
        warp_size: Warp width W.
    """

    def __init__(self, warp_size: int = 32) -> None:
        self.W = warp_size

    def compute_indices(
        self,
        queue_lens: Dict[str, int],
        ages: Dict[str, float],
        lambda_est: Dict[str, float],
        deadlines: Dict[str, float],
    ) -> Dict[str, float]:
        """I_k for every adapter in queue_lens. O(K).

        Args:
            queue_lens:  |Q_k| per adapter.
            ages:        age of the oldest token in Q_k, seconds.
            lambda_est:  lambda_k per adapter, tokens/s.
            deadlines:   T_k per adapter, seconds.

        Returns:
            {adapter_id: I_k >= 0}.
        """
        ids = list(queue_lens)
        if not ids:
            return {}
        q = np.array([queue_lens[k] for k in ids], dtype=float)
        lam = np.array([max(lambda_est.get(k, 0.0), 0.0) for k in ids])
        left = np.array([max(deadlines.get(k, 0.0) - ages.get(k, 0.0), 0.0) for k in ids])
        missing = np.maximum(self.W - q, 0.0)
        s = np.minimum(q / self.W, 1.0)
        # Erlang(m, lambda) CDF at t is the regularized lower incomplete gamma P(m, lambda t).
        p_fill = np.where(missing > 0, gammainc(np.maximum(missing, 1.0), lam * left), 1.0)
        return dict(zip(ids, (lam * s * p_fill).tolist()))

    def rank(self, indices: Dict[str, float]) -> List[str]:
        """Adapters sorted by score, highest first. O(K log K)."""
        return sorted(indices, key=lambda k: indices[k], reverse=True)
