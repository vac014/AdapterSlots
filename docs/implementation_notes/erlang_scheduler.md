# Erlang Timeout + Fairness

A single global T_max over- or under-waits depending on per-adapter arrival rate.
This phase sets each adapter's timeout to the Erlang-quantile inverse of its rate:

```
T_max^(k) = Erlang_CDF_inv(W=32, λ_k, quantile=WAR*)     (Theorem 5.3)
```

- Per-adapter timeout policy: `adapter_slots/dispatch/erlang.py`
- Rate estimation λ_k: `adapter_slots/control/estimator.py`
- Buffer integration: `adapter_slots/buffer.py`

A fairness constraint bounds per-adapter waiting so low-rate adapters are not starved.
