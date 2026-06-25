"""
pi_controller.py -- PI control of the global deadline T_max (Theorem 2).

With the alignment error e(t) = WAR* - WAR(t), each tick applies

    T_max(t+1) = T_max(t) + K_p (e(t) - e(t-1)) + K_i e(t)

and clips T_max to [tmax_min, tmax_max]. Linearised around the operating point,
with L the local slope of E[WAR | T_max], the loop is locally asymptotically
stable if and only if

    K_p > 0,   K_i > 0,   2 K_p + K_i < 2 / L.

L is estimated online from finite differences of the observed WAR against
T_max, and the gains are scaled into the stable region before every update.
"""

from typing import Optional, Tuple


class PIController:
    """Incremental PI controller for WAR tracking.

    Args:
        kp:            Proportional gain K_p (seconds per unit WAR error).
        ki:            Integral gain K_i (seconds per unit WAR error).
        war_target:    WAR* in (0, 1).
        tmax_init:     Initial T_max in seconds.
        tmax_min:      Lower bound on T_max in seconds.
        tmax_max:      Upper bound on T_max in seconds.
        margin:        Fraction of the bound 2 / L the gains may use.
        slope_alpha:   EWMA weight of each new finite-difference slope sample.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        war_target: float,
        tmax_init: float = 0.005,
        tmax_min: float = 0.001,
        tmax_max: float = 5.0,
        margin: float = 0.9,
        slope_alpha: float = 0.2,
    ) -> None:
        if kp <= 0 or ki <= 0:
            raise ValueError(f"gains must be positive, got K_p={kp}, K_i={ki}")
        self.kp = kp
        self.ki = ki
        self.war_target = war_target
        self.tmax = float(tmax_init)
        self.tmax_min = float(tmax_min)
        self.tmax_max = float(tmax_max)
        self.margin = margin
        self.slope_alpha = slope_alpha
        self.L: Optional[float] = None
        self.t = 0
        self._prev_error: Optional[float] = None
        self._prev_war: Optional[float] = None
        self._prev_tmax: Optional[float] = None

    def effective_gains(self) -> Tuple[float, float]:
        """(K_p, K_i) scaled so that 2 K_p + K_i <= margin * 2 / L."""
        if not self.L:
            return self.kp, self.ki
        bound = self.margin * 2.0 / self.L
        total = 2.0 * self.kp + self.ki
        if total <= bound:
            return self.kp, self.ki
        scale = bound / total
        return self.kp * scale, self.ki * scale

    def _update_slope(self, war: float) -> None:
        if self._prev_war is None:
            return
        d_tmax = self.tmax - self._prev_tmax
        if abs(d_tmax) < 1e-6:
            return
        sample = abs((war - self._prev_war) / d_tmax)
        if self.L is None:
            self.L = sample
        else:
            self.L = (1.0 - self.slope_alpha) * self.L + self.slope_alpha * sample

    def update(self, war_observed: float) -> float:
        """One control step from the WAR observed under the current T_max.

        Returns the new T_max in seconds.
        """
        self._update_slope(war_observed)
        kp, ki = self.effective_gains()
        e = self.war_target - war_observed
        de = 0.0 if self._prev_error is None else e - self._prev_error
        self._prev_error = e
        self._prev_war = war_observed
        self._prev_tmax = self.tmax
        self.tmax = min(max(self.tmax + kp * de + ki * e, self.tmax_min), self.tmax_max)
        self.t += 1
        return self.tmax

    def summary(self) -> dict:
        kp, ki = self.effective_gains()
        return {
            "t": self.t,
            "tmax": self.tmax,
            "war_target": self.war_target,
            "kp": kp,
            "ki": ki,
            "L": self.L,
        }
