# Whittle Index Scheduler (E8)

When several adapters compete for the dispatch slot, threshold dispatch is suboptimal.
We model dispatch as a restless multi-armed bandit and dispatch by Whittle index,
which ranks adapters by the value of dispatching now vs waiting (Theorem 8.7,
indexability proven).

- Whittle dispatcher: `adapter_slots/dispatch/whittle.py`
- Oracle upper bound: `adapter_slots/dispatch/oracle.py`
- Threshold/FIFO baselines: `adapter_slots/dispatch/baselines.py`

Indexability and near-optimality checked in `analysis/validate_indexability.py`.
On saturated throughput the policies tie FIFO; their role is the control plane
(SLO/fairness), not raw throughput (see docs/results.md §1).
