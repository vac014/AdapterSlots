"""adapter_slots.dispatch -- Dispatch policy modules for AdapterSlots."""

from .baselines import FIFODispatcher, GreedyFillDispatcher
from .erlang import (
    compute_tmax_erlang,
    compute_tmax_erlang_batch,
    erlang_cdf,
    erlang_pdf,
    quantization_conservatism_bound,
    fairness_constrained_war,
)
from .oracle import OracleScheduler
from .whittle import WhittleDispatcher, measure_tau_iter

__all__ = [
    "FIFODispatcher",
    "GreedyFillDispatcher",
    "OracleScheduler",
    "WhittleDispatcher",
    "measure_tau_iter",
    "compute_tmax_erlang",
    "compute_tmax_erlang_batch",
    "erlang_cdf",
    "erlang_pdf",
    "quantization_conservatism_bound",
    "fairness_constrained_war",
]
