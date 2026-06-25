# PI Controller (E7)

Arrival rates λ_k drift (diurnal, popularity, bursts), so a static Erlang timeout
carries a permanent WAR penalty. A discrete-time PI controller adapts T_max online to
track a WAR target:

```
T_max(t+1) = T_max(t) + Kp·e(t) + Ki·Σ e(s),   e(t) = WAR* − WAR(t)   (Theorem 6.3)
```

Mean-square stable for Kp ∈ (0, 2/L) and small Ki.

- Controller: `adapter_slots/control/pi_controller.py`
- Rate/error estimation: `adapter_slots/control/estimator.py`

Stability and drift response validated in `analysis/validate_theorem_6_3.py`.
