"""
whittle.py -- Whittle Index Dispatcher for Aligned Dispatch (whittle_scheduler, Theorem 8.7).

Implements the Whittle index policy for the RMAB formulation of adapter dispatch:

    W_k(s_k) = p_k * s_k * [1 - (1 - W*λ_k*Δt)^{W*(1-s_k)}]

where:
    s_k = fill fraction of queue k (|Q_k|/W in [0,1])
    p_k = long-run traffic fraction for adapter k
    λ_k = current arrival rate estimate for adapter k (tokens/sec)
    Δt  = scheduling tick interval (seconds) -- must be set to τ_iter (hardware-dependent)
    W   = warp size (32 for all NVIDIA hardware)

Near-optimality (Theorem 8.7): achieves 85–95% of oracle WAR at O(K) overhead per tick.
Asymptotically optimal as K → ∞ (Weber & Weiss, J. Appl. Prob., 1990).

MULTI-GPU NOTE: delta_t MUST be set to τ_iter for the target hardware:
    PCIe  (τ_iter ≈ 100 ms): use delta_t = 0.10
    NVLink(τ_iter ≈   5 ms): use delta_t = 0.005
    Single A6000 (τ_iter ≈ 30 ms): use delta_t = 0.030
Calibrate τ_iter empirically before each experiment run (measure_tau_iter below).

Using the wrong Δt causes fill-probability estimation error:
    delta_t = 0.001 on PCIe → rate = W*λ_k*0.001 underestimates fill probability by ~100×,
    making Whittle ranking unreliable (dispatches empty queues before partially-filled ones).

TP-transparency (Proposition 8.8):
    rank_adapters() is pure Python over {p_k, s_k, λ_k, W, Δt} -- no GPU interaction.
    Run in the scheduler process (rank-0 only). The ArrivalRateEstimator from erlang_scheduler/6
    must NOT be instantiated in TP worker processes (double-counting bias).
    The ranked batch is passed to vLLM SchedulerOutputs as a single reordered sequence
    group before TP sharding. WAR(Whittle, TP=d) = WAR(Whittle, TP=1) for all d.

References:
    - whittle_scheduler.md §3, §7.4, §7.5, §7.6
    - Theorem 8.7, Propositions 8.8, 8.9, 8.10
    - Weber & Weiss (1990), J. Appl. Prob. 27(3):637-648
"""

import time
from typing import Dict, List, Optional

import numpy as np


def measure_tau_iter(serving_client=None, n_samples: int = 100) -> float:
    """Measure mean decode iteration wall-clock time via sequential decode steps.

    Use before any E8-bandit experiment to calibrate delta_t for the target hardware.
    If serving_client is None, returns a dummy measurement (for offline testing).

    Reference values (re-calibrate empirically each run):
        Single A6000 (TP=1, LLaMA-7B): τ_iter ≈ 0.030 s  → delta_t = 0.030
        Two A6000 PCIe (TP=2):         τ_iter ≈ 0.100 s  → delta_t = 0.100
        Two H100 NVLink (TP=2):        τ_iter ≈ 0.005 s  → delta_t = 0.005

    Args:
        serving_client: Object with a .decode_one_step() method.  Pass None for offline.
        n_samples:      Number of sequential single-token decode steps to average.

    Returns:
        Mean decode iteration time in seconds.
    """
    if serving_client is None:
        # Offline: return a plausible default for unit tests
        return 0.030

    latencies = []
    for _ in range(n_samples):
        t0 = time.monotonic()
        serving_client.decode_one_step()
        latencies.append(time.monotonic() - t0)

    tau_iter = float(np.mean(latencies))
    print(f"Measured τ_iter = {tau_iter * 1000:.1f} ms  →  set delta_t = {tau_iter:.4f} s")
    return tau_iter


class WhittleDispatcher:
    """Whittle index policy for aligned dispatch (Theorem 8.7).

    Computes W_k(s_k) = p_k * s_k * [1 - (1 - W*λ_k*Δt)^{W*(1-s_k)}] for each adapter
    and returns them ranked highest-first.  O(K) per tick.

    Args:
        adapters:  List of adapter IDs served by this dispatcher.
        warp_size: GPU warp width W (32 for all NVIDIA hardware).
        delta_t:   Scheduling tick interval in seconds.  MUST be set to τ_iter for
                   the target hardware (see module docstring).  Default 0.001 is only
                   a placeholder -- calibrate before use.
    """

    def __init__(
        self,
        adapters: List[str],
        warp_size: int = 32,
        delta_t: float = 0.001,
    ) -> None:
        self.W = warp_size
        self.K = len(adapters)
        self.adapters = list(adapters)
        self.delta_t = delta_t  # scheduling tick interval in seconds; SET TO τ_iter

        # Prior probability estimate p_k (uniform initially; updated from traffic)
        self.p_k: Dict[str, float] = {k: 1.0 / self.K for k in self.adapters}

    # Public API

    def compute_indices(
        self,
        fill_fracs: Dict[str, float],
        lambda_est: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute Whittle index for each adapter.  O(K).

        Args:
            fill_fracs:  s_k = |Q_k| / W for each adapter, in [0, 1].
            lambda_est:  Estimated arrival rate λ_k for each adapter (tokens/sec).

        Returns:
            Dict mapping adapter_id → Whittle index value ≥ 0.
        """
        indices: Dict[str, float] = {}
        for k in self.adapters:
            s = min(fill_fracs.get(k, 0.0), 1.0)
            lam = max(lambda_est.get(k, 1e-6), 1e-9)
            p = self.p_k[k]

            remaining_needed = max(int(self.W * (1.0 - s)), 0)
            rate = self.W * lam * self.delta_t

            # P(fill in next Δt | current fill = s): geometric approximation.
            # Special case: remaining_needed=0 means queue is already full → prob=1.
            if remaining_needed == 0:
                fill_prob = 1.0
            else:
                fill_prob = 1.0 - max(1.0 - rate, 0.0) ** remaining_needed

            indices[k] = p * s * fill_prob

        return indices

    def rank_adapters(
        self,
        fill_fracs: Dict[str, float],
        lambda_est: Dict[str, float],
    ) -> List[str]:
        """Return adapters sorted by Whittle index (highest first).  O(K log K)."""
        indices = self.compute_indices(fill_fracs, lambda_est)
        return sorted(indices, key=lambda k: indices[k], reverse=True)

    def update_traffic_fractions(self, request_counts: Dict[str, int]) -> None:
        """Update p_k from observed request counts using EWMA smoothing (α=0.1)."""
        total = max(sum(request_counts.values()), 1)
        for k in self.adapters:
            empirical_p = request_counts.get(k, 0) / total
            self.p_k[k] = 0.9 * self.p_k[k] + 0.1 * empirical_p

    def _index_for_state(self, s: float, lambda_k: float, p_k: float) -> float:
        """Single-state index computation for indexability validation (§7.1).

        Used by check_indexability() to verify monotonicity of indices in s.
        Mathematical property only -- result must be identical across hardware.
        """
        s = min(s, 1.0)
        remaining_needed = max(int(self.W * (1.0 - s)), 0)
        rate = self.W * lambda_k * self.delta_t
        if remaining_needed == 0:
            fill_prob = 1.0
        else:
            fill_prob = 1.0 - max(1.0 - rate, 0.0) ** remaining_needed
        return p_k * s * fill_prob

    def check_indexability(
        self,
        lambda_k: float,
        p_k: float = 1.0,
        n_states: int = 32,
    ) -> bool:
        """Verify Whittle indices are monotonically non-decreasing in fill fraction.

        This is the indexability condition required for RMAB near-optimality (Whittle, 1988).
        Hardware-independent: run on all three setups with hardware-calibrated delta_t.
        A failure on one hardware but not another indicates a delta_t calibration bug.

        IMPORTANT -- heavy-traffic regime:
        Indexability is guaranteed when W*λ_k*Δt ≥ 1 (fill probability saturates to 1.0
        for all partially-filled queues, making index = p_k * s -- trivially monotone).
        For light-traffic adapters (W*λ_k*Δt < 1), the geometric approximation may not be
        monotone over discrete fill states; those adapters are handled by the Erlang fairness
        cap in any case (§7.3, "bad conditions: adversarial ABAB, indexability may fail").

        Args:
            lambda_k:  Arrival rate for the adapter under test.
            p_k:       Traffic fraction (affects magnitude, not monotonicity).
            n_states:  Number of fill-fraction states to check (default = W = 32).

        Returns:
            True if indices are monotonically non-decreasing in s.
        """
        states = [i / n_states for i in range(n_states + 1)]
        indices = [self._index_for_state(s, lambda_k, p_k) for s in states]
        return all(indices[i] <= indices[i + 1] + 1e-12 for i in range(len(indices) - 1))

    def is_heavy_traffic(self, lambda_k: float) -> bool:
        """Return True if W*λ_k*Δt ≥ 1 (heavy-traffic / indexable regime).

        In this regime, fill probability = 1.0 for all partially-filled queues,
        making the Whittle index trivially monotone (index = p_k * s).
        Adapters outside this regime are handled by the Erlang fairness cap.
        """
        return self.W * lambda_k * self.delta_t >= 1.0
