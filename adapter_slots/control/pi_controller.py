"""
pi_controller.py -- Discrete-time PI controller for WAR tracking (pi_controller, Theorem 6.3).

Implements:
    - PIController: core PI update loop with gain schedule and anti-windup
    - IterationBoundaryPIController: wraps PIController for iteration-boundary-driven
      updates (Proposition 6.5) -- n_q = 1 on all hardware, gains hardware-independent

Design notes (pi_controller.md §3):

    T_max(t+1) = T_max(t) + K_p * e(t) + K_i * sum_{s=0}^{t} e(s)
    e(t) = WAR* - WAR(t)

    Stability condition (Theorem 6.3):
        K_p ∈ (0, 2/L)   where L = Lipschitz constant of WAR(T_max)
        K_i < K_p * (2/L - K_p)   [sufficient condition]

    Update mode selection (Proposition 6.5):
        ITERATION_BOUNDARY -- one update per completed iteration (recommended)
            → n_q = 1 on all hardware; K_p, K_i are hardware-independent
        TICK_DRIVEN -- one update per 1ms scheduling tick (comparison only)
            → n_q = ceil(τ_iter / 1ms); K_i must be scaled by 1/n_q per hardware

    Gain portability (Proposition 6.7):
        L depends only on arrival rates λ_k, mixture weights p_k, and warp size W.
        L is hardware-independent -- gains calibrated on Single A6000 are reused
        unchanged on Two A6000 PCIe and Two H100 NVLink.

    Wall-clock convergence (Proposition 6.6):
        t_settle_wc(hw) = N_settle × τ_iter(hw)
        PI converges ~20× faster in wall-clock on NVLink than PCIe.

References:
    - pi_controller.md §3, §7 (Propositions 6.5, 6.6, 6.7; Theorem 6.3)
"""

import numpy as np
from typing import List, Optional


class PIController:
    """
    Discrete-time PI controller for WAR tracking (Theorem 6.3).

    T_max(t+1) = T_max(t) + K_p * e(t) + K_i * sum_{s=0}^{t} e(s)

    Stability condition (Theorem 6.3):
        K_p ∈ (0, 2/L)   where L = Lipschitz constant of WAR(T_max)
        K_i < K_p * (2/L - K_p)   [sufficient small condition]

    Update mode (Proposition 6.5):
        Use ITERATION_BOUNDARY mode (one update per completed iteration).
        This makes n_q = 1 regardless of hardware τ_iter, keeping K_p and K_i
        hardware-independent.
        TICK_DRIVEN is provided for negative-control experiments only.

    Gain schedule (global stability):
        For large |e(t)|, K_p is scaled down to prevent overshoot and ensure
        T_max stays within [tmax_min, tmax_max].

    Anti-windup:
        The integral term is clamped so K_i * integral < (tmax_max - tmax_min) / 2.
        This bound is hardware-independent under iteration-boundary-driven updates.
    """

    # Update modes
    ITERATION_BOUNDARY = "iteration_boundary"  # recommended: one update per τ_iter
    TICK_DRIVEN = "tick_driven"               # comparison only: one update per 1ms tick

    def __init__(
        self,
        kp: float,
        ki: float,
        war_target: float,
        lipschitz: Optional[float] = None,
        tmax_init: float = 0.005,
        tmax_min: float = 0.001,
        tmax_max: float = 5.0,
        update_mode: str = "iteration_boundary",
    ) -> None:
        """
        Args:
            kp:           Proportional gain. Must satisfy K_p ∈ (0, 2/L) for stability.
            ki:           Integral gain. Must satisfy K_i < K_p*(2/L - K_p).
            war_target:   WAR* target ∈ (0, 1).
            lipschitz:    Estimated Lipschitz constant L of WAR(T_max). Set via
                          estimate_lipschitz() before using closed_loop_matrix.
            tmax_init:    Initial T_max value in seconds (default 5ms).
            tmax_min:     Hard lower bound for T_max (default 1ms).
            tmax_max:     Hard upper bound for T_max (default 5s).
            update_mode:  ITERATION_BOUNDARY (recommended) or TICK_DRIVEN.
        """
        self.kp = kp
        self.ki = ki
        self.war_target = war_target
        self.L = lipschitz
        self.tmax = float(tmax_init)
        self.tmax_min = float(tmax_min)
        self.tmax_max = float(tmax_max)
        self.update_mode = update_mode
        self.t = 0                      # update step counter

        self.integral = 0.0            # sum_{s=0}^{t} e(s)
        self._error_history: List[float] = []

        # Anti-windup cap: K_i * |integral| <= (tmax_max - tmax_min) / 2
        # Hardware-independent because K_i is calibrated for n_q=1 (iteration mode).
        self._integral_cap = (self.tmax_max - self.tmax_min) / (2.0 * max(ki, 1e-9))

    # Core update

    def update(self, war_observed: float) -> float:
        """
        One PI update step.

        In ITERATION_BOUNDARY mode: call once per completed iteration (every τ_iter
        wall-clock seconds). n_q = 1 on all hardware; gains are hardware-independent.

        In TICK_DRIVEN mode: call once per 1ms scheduling tick. Requires K_i to be
        scaled by 1/n_q for each hardware platform (see Proposition 6.5). Provided
        for negative-control experiments only.

        Args:
            war_observed: WAR metric measured over the most recent iteration/tick.

        Returns:
            Updated global T_max in seconds.
        """
        e = self.war_target - war_observed
        self._error_history.append(e)

        # Gain schedule: reduce K_p for large errors to prevent overshoot.
        # This extends local Theorem 6.3 stability to a global bounded regime.
        kp_eff = self.kp
        if self.L is not None:
            stability_margin = 2.0 / self.L - self.kp
            if stability_margin > 0 and abs(e) > stability_margin:
                kp_eff = self.kp * stability_margin / abs(e)

        # Anti-windup: clamp integral before accumulation (hardware-independent).
        self.integral = float(np.clip(
            self.integral + e,
            -self._integral_cap,
            self._integral_cap,
        ))

        self.tmax += kp_eff * e + self.ki * self.integral
        self.tmax = float(np.clip(self.tmax, self.tmax_min, self.tmax_max))
        self.t += 1
        return self.tmax

    # Stability analysis

    @property
    def closed_loop_matrix(self) -> np.ndarray:
        """
        Closed-loop state matrix A for stability analysis (Theorem 6.3).

        State vector: x(t) = [e(t), Σ_s e(s)]
        Dynamics: x(t+1) = A × x(t)

        A = [[1 - L*K_p,  -L*K_i],
             [1,           1     ]]

        Hardware-independent: L, K_p, K_i depend only on arrival rates and warp
        size, not on τ_iter, GPU architecture, or interconnect (Proposition 6.7).

        Raises:
            ValueError if L has not been set.
        """
        if self.L is None:
            raise ValueError(
                "Lipschitz constant L not set. Call estimate_lipschitz() first "
                "and assign controller.L = result."
            )
        return np.array([
            [1.0 - self.L * self.kp, -self.L * self.ki],
            [1.0,                     1.0],
        ])

    @property
    def spectral_radius(self) -> float:
        """
        ρ(A) -- spectral radius of the closed-loop matrix.

        Must be < 1 for mean-square stability (Theorem 6.3).
        Hardware-independent for fixed K_p, K_i (Proposition 6.7).
        """
        eigenvalues = np.linalg.eigvals(self.closed_loop_matrix)
        return float(np.max(np.abs(eigenvalues)))

    def stability_report(self) -> dict:
        """
        Full stability check against Theorem 6.3 conditions.

        Returns a dict with all stability checks so callers can assert pass/fail.
        Use validate_theorem_6_3.py for experiment-level validation.
        """
        if self.L is None:
            return {"error": "L not set"}

        kp_upper = 2.0 / self.L
        ki_upper = self.kp * (2.0 / self.L - self.kp)
        rho = self.spectral_radius

        return {
            "L": self.L,
            "kp": self.kp,
            "ki": self.ki,
            "kp_upper_bound": kp_upper,
            "ki_upper_bound": ki_upper,
            "kp_in_range": 0.0 < self.kp < kp_upper,
            "ki_in_range": self.ki < ki_upper,
            "spectral_radius": rho,
            "is_stable": rho < 1.0,
            "kp_stability_margin": kp_upper - self.kp,
            "ki_stability_margin": ki_upper - self.ki,
            "gain_schedule_threshold_delta": (
                2.0 / self.L - self.kp if self.L is not None else None
            ),
        }

    # Diagnostics

    def settling_time_prediction(self, tolerance: float = 0.01) -> int:
        """
        Predicted N_settle = ceil(log(tolerance) / log(ρ)) iterations (Corollary 6.4).

        Hardware-independent: N_settle depends only on ρ (hence K_p, K_i, L).
        Wall-clock: t_settle_wc = N_settle × τ_iter (Proposition 6.6).

        Args:
            tolerance: Error reduction target (default 1% = 0.01).

        Returns:
            Predicted settling time in PI update steps (iterations).
        """
        rho = self.spectral_radius
        if rho >= 1.0:
            return int(1e9)   # unstable -- settling undefined
        import math
        return math.ceil(math.log(tolerance) / math.log(rho))

    def reset(self) -> None:
        """Reset controller state (integral, counter, history) but keep gains."""
        self.integral = 0.0
        self.t = 0
        self._error_history.clear()

    def summary(self) -> dict:
        """Return state dict for logging."""
        return {
            "t": self.t,
            "tmax": self.tmax,
            "integral": self.integral,
            "war_target": self.war_target,
            "kp": self.kp,
            "ki": self.ki,
            "L": self.L,
            "update_mode": self.update_mode,
        }


# Iteration-boundary wrapper


class IterationBoundaryPIController:
    """
    Wraps PIController for iteration-boundary-driven updates (Proposition 6.5).

    The scheduler calls trigger_iteration_end() after each completed iteration.
    This fires exactly one PI update per iteration regardless of how many 1ms
    scheduling ticks occurred during that iteration.  K_p and K_i are therefore
    hardware-independent (n_q = 1 always), eliminating PCIe integral-windup risk.

    Usage:
        pi = PIController(kp=0.01, ki=0.001, war_target=0.8, lipschitz=L)
        ctrl = IterationBoundaryPIController(pi)

        # During dispatch loop for each batch in iteration:
        ctrl.record_batch_war(batch_war)

        # Once at the end of each completed iteration:
        new_tmax = ctrl.trigger_iteration_end()
    """

    def __init__(self, pi: PIController) -> None:
        self._pi = pi
        self._war_accumulator: List[float] = []   # WAR samples in current iteration
        self._current_tmax = pi.tmax
        self._iteration_count = 0
        self._iter_war_history: List[float] = []  # mean WAR per completed iteration

    @property
    def controller(self) -> PIController:
        """Access to the underlying PIController (for stability analysis)."""
        return self._pi

    @property
    def tmax(self) -> float:
        """Current T_max (seconds)."""
        return self._current_tmax

    def record_batch_war(self, batch_war: float) -> None:
        """
        Record WAR for one dispatched batch during the current iteration.

        Call once per batch dispatch in the scheduler process (rank 0 only).
        """
        self._war_accumulator.append(batch_war)

    def trigger_iteration_end(self) -> float:
        """
        Called once when an iteration completes (τ_iter wall-clock seconds have elapsed).

        Averages WAR over all batches in the iteration and issues exactly one PI update.
        Under n_q=1 (this mode), K_p and K_i are hardware-independent -- no scaling needed
        for PCIe or NVLink relative to the Single A6000 calibration.

        Returns:
            Updated T_max in seconds. Also stored in self.tmax.
        """
        if self._war_accumulator:
            war_mean = float(np.mean(self._war_accumulator))
            self._iter_war_history.append(war_mean)
            self._war_accumulator.clear()
            self._current_tmax = self._pi.update(war_mean)

        self._iteration_count += 1
        return self._current_tmax

    def reset(self) -> None:
        """Reset accumulator and iteration counter."""
        self._war_accumulator.clear()
        self._iter_war_history.clear()
        self._iteration_count = 0
        self._pi.reset()
        self._current_tmax = self._pi.tmax

    def summary(self) -> dict:
        """Return state dict for logging."""
        return {
            "iteration_count": self._iteration_count,
            "current_tmax": self._current_tmax,
            "pending_war_samples": len(self._war_accumulator),
            "pi": self._pi.summary(),
        }
