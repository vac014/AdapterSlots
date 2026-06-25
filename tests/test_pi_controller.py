"""
test_pi_controller.py -- PI control of T_max (Theorem 2): the incremental
update, the online slope estimate and the gain region 2 K_p + K_i < 2 / L.
"""

import pytest

from adapter_slots.control.pi_controller import PIController


def run_linear_plant(ctrl: PIController, slope: float, offset: float, steps: int) -> float:
    """Close the loop on WAR = offset + slope * T_max (clipped to [0, 1])."""
    war = 0.0
    for _ in range(steps):
        war = min(max(offset + slope * ctrl.tmax, 0.0), 1.0)
        ctrl.update(war)
    return war


class TestPIController:

    def test_rejects_non_positive_gains(self):
        with pytest.raises(ValueError):
            PIController(kp=0.0, ki=0.001, war_target=0.8)
        with pytest.raises(ValueError):
            PIController(kp=0.01, ki=-0.001, war_target=0.8)

    def test_first_step_is_integral_only(self):
        """With no previous error the update is K_i * e."""
        ctrl = PIController(kp=0.01, ki=0.002, war_target=0.8, tmax_init=0.05)
        ctrl.update(0.3)
        assert ctrl.tmax == pytest.approx(0.05 + 0.002 * 0.5)

    def test_incremental_update(self):
        """T(t+1) = T(t) + K_p (e(t) - e(t-1)) + K_i e(t) while the slope is unknown."""
        ctrl = PIController(kp=0.01, ki=0.002, war_target=0.8, tmax_init=0.05)
        ctrl.update(0.3)
        t1 = ctrl.tmax
        ctrl.L = None
        ctrl._prev_war = None
        ctrl.update(0.5)
        assert ctrl.tmax == pytest.approx(t1 + 0.01 * (0.3 - 0.5) + 0.002 * 0.3)

    def test_tmax_clipped(self):
        ctrl = PIController(kp=0.5, ki=0.5, war_target=0.8, tmax_init=0.01,
                            tmax_min=0.001, tmax_max=0.2)
        for _ in range(50):
            ctrl.update(0.0)
        assert ctrl.tmax == pytest.approx(0.2)
        for _ in range(50):
            ctrl.update(1.0)
        assert ctrl.tmax == pytest.approx(0.001)

    def test_slope_estimated_from_finite_differences(self):
        ctrl = PIController(kp=0.001, ki=0.001, war_target=0.8, tmax_init=0.01)
        run_linear_plant(ctrl, slope=4.0, offset=0.1, steps=30)
        assert ctrl.L == pytest.approx(4.0, rel=1e-6)

    def test_gains_scaled_into_stable_region(self):
        ctrl = PIController(kp=1.0, ki=1.0, war_target=0.8, margin=0.9)
        ctrl.L = 10.0
        kp, ki = ctrl.effective_gains()
        assert 2 * kp + ki == pytest.approx(0.9 * 2 / 10.0)
        assert kp / ki == pytest.approx(1.0)

    def test_gains_unchanged_inside_region(self):
        ctrl = PIController(kp=0.01, ki=0.001, war_target=0.8)
        ctrl.L = 1.0
        assert ctrl.effective_gains() == (0.01, 0.001)

    def test_converges_on_linear_plant(self):
        """Gains far outside the region still converge once they are scaled."""
        ctrl = PIController(kp=2.0, ki=2.0, war_target=0.8, tmax_init=0.01,
                            tmax_max=5.0)
        war = run_linear_plant(ctrl, slope=5.0, offset=0.1, steps=400)
        assert war == pytest.approx(0.8, abs=1e-3)
        assert ctrl.tmax == pytest.approx(0.14, abs=1e-3)
