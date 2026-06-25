"""
oracle.py -- Offline Oracle Scheduler: brute-force DP upper bound (whittle_scheduler, §4).

The oracle has access to the full future arrival sequence (not deployable in practice)
and solves a finite-horizon DP to compute the maximum achievable WAR.  Used only as
the 100% reference line in E8-bandit experiments.

Complexity: O(K^H × W^K) -- only feasible for K ≤ 4, H ≤ 20.
For K > 4, extrapolate from the K=4 oracle ratio or skip (mark as N/A in tables).

MULTI-GPU NOTE:
    This is a pure CPU-side Python computation.  Hardware-independent in correctness.
    Oracle WAR must be IDENTICAL across hardware setups for the same arrival sequence
    and random seed.  Any difference between hardware setups indicates a bug, not a
    hardware effect.  Only the *throughput* benefit of oracle dispatching differs across
    hardware (via the E11.2 unmasking mechanism -- PCIe all-reduce overhead masks SGMV gains).

References:
    - whittle_scheduler.md §4
    - Theorem 8.7 -- oracle is the 100% reference for near-optimality claims
"""

from functools import lru_cache
from typing import Dict, List, Optional


class OracleScheduler:
    """Offline oracle: brute-force DP over H-tick horizon.

    Not deployable (requires future arrivals as input).
    Used only as an upper bound for E8-bandit experiments.

    Args:
        W:       Warp size (32 for all NVIDIA hardware).
        K:       Number of adapters.  Must be ≤ 4 (DP tractability).
        horizon: Planning horizon in ticks.  Must be ≤ 20 (DP tractability).
    """

    def __init__(self, W: int = 32, K: int = 4, horizon: int = 20) -> None:
        if K > 4:
            raise ValueError(f"Oracle DP only tractable for K ≤ 4; got K={K}")
        if horizon > 20:
            raise ValueError(f"Oracle DP too expensive for H > 20; got H={horizon}")
        self.W = W
        self.K = K
        self.H = horizon

    def solve(self, future_arrivals: List[Dict[str, int]]) -> float:
        """Compute oracle-optimal WAR over the H-tick horizon.

        Args:
            future_arrivals: List of H dicts {adapter_id: n_arrivals_this_tick}.
                             Adapter IDs must be integers 0..K-1 or strings '0'..'K-1'.
                             Missing adapter IDs → 0 arrivals for that adapter that tick.
        Returns:
            Oracle WAR ∈ [0, 1]: fraction of dispatch ticks where a full warp was sent.
        """
        H = min(len(future_arrivals), self.H)
        W = self.W
        K = self.K

        # Normalise arrival dicts to int-keyed 0..K-1
        arrivals_norm: List[Dict[int, int]] = []
        for tick_dict in future_arrivals[:H]:
            d: Dict[int, int] = {}
            for key, val in tick_dict.items():
                try:
                    d[int(key)] = int(val)
                except (ValueError, TypeError):
                    pass
            arrivals_norm.append(d)

        @lru_cache(maxsize=None)
        def dp(t: int, state: tuple) -> float:
            """Returns max aligned warps dispatched from tick t to H given queue state."""
            if t >= H:
                return 0.0

            tick_arrivals = arrivals_norm[t]
            # Apply arrivals to queue state; cap each queue at W
            new_state = tuple(
                min(state[k] + tick_arrivals.get(k, 0), W) for k in range(K)
            )

            best = 0.0
            # Try dispatching each adapter (dispatch_k) or none (-1)
            for dispatch_k in range(-1, K):
                reward = 0.0
                next_state = list(new_state)
                if dispatch_k >= 0 and new_state[dispatch_k] > 0:
                    n_warps = new_state[dispatch_k] // W
                    reward = float(n_warps)        # aligned warps dispatched this tick
                    next_state[dispatch_k] -= n_warps * W
                future_value = dp(t + 1, tuple(next_state))
                total = reward + future_value
                if total > best:
                    best = total

            return best

        init_state = tuple([0] * K)
        total_warps = dp(0, init_state)
        dp.cache_clear()  # free memory after each solve call

        max_possible = float(H)  # one dispatch slot per tick
        return min(total_warps / max(max_possible, 1.0), 1.0)

    def best_action(
        self,
        current_state: List[int],
        future_arrivals: List[Dict[int, int]],
    ) -> int:
        """Return the optimal adapter index to dispatch now (or -1 for no dispatch).

        Used by the oracle policy to make DP-optimal decisions
        given the current queue state and known future arrivals.

        Args:
            current_state:   Current queue depths [q_0, q_1, ..., q_{K-1}].
            future_arrivals: Next H dicts {adapter_id: n_arrivals} -- same format
                             as solve() but starting from the NEXT tick (t+1).
        Returns:
            Adapter index in 0..K-1 that maximises expected aligned warps, or -1
            for no dispatch.
        """
        H = min(len(future_arrivals), self.H)
        W = self.W
        K = self.K

        arrivals_norm: List[Dict[int, int]] = []
        for tick_dict in future_arrivals[:H]:
            d: Dict[int, int] = {}
            for key, val in tick_dict.items():
                try:
                    d[int(key)] = int(val)
                except (ValueError, TypeError):
                    pass
            arrivals_norm.append(d)

        @lru_cache(maxsize=None)
        def dp(t: int, state: tuple) -> float:
            if t >= H:
                return 0.0
            tick_arr = arrivals_norm[t]
            new_state = tuple(min(state[k] + tick_arr.get(k, 0), W) for k in range(K))
            best = 0.0
            for dk in range(-1, K):
                reward = 0.0
                ns = list(new_state)
                if dk >= 0 and new_state[dk] > 0:
                    nw = new_state[dk] // W
                    reward = float(nw)
                    ns[dk] -= nw * W
                total = reward + dp(t + 1, tuple(ns))
                if total > best:
                    best = total
            return best

        state0 = tuple(current_state[:K])
        best_val = -1.0
        best_k = -1
        for dk in range(-1, K):
            reward = 0.0
            ns = list(state0)
            if dk >= 0 and state0[dk] > 0:
                nw = state0[dk] // W
                reward = float(nw)
                ns[dk] -= nw * W
            val = reward + dp(0, tuple(ns))
            if val > best_val:
                best_val = val
                best_k = dk

        dp.cache_clear()
        return best_k

    @staticmethod
    def generate_poisson_arrivals(
        K: int,
        H: int,
        lambda_k: List[float],
        seed: Optional[int] = 42,
    ) -> List[Dict[int, int]]:
        """Generate Poisson arrival sequences for offline oracle testing.

        Args:
            K:        Number of adapters.
            H:        Number of ticks.
            lambda_k: Per-adapter arrival rate (tokens/tick; not tokens/sec).
            seed:     RNG seed for reproducibility.

        Returns:
            List of H dicts {adapter_id: n_arrivals}.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        arrivals = []
        for _ in range(H):
            tick: Dict[int, int] = {}
            for k in range(K):
                n = int(rng.poisson(lambda_k[k]))
                if n > 0:
                    tick[k] = n
            arrivals.append(tick)
        return arrivals
