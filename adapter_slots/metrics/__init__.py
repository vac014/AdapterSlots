"""
adapter_slots.metrics -- WAR, WARτ, H_align, and GWAR metric computation.
"""

from adapter_slots.metrics.war import (
    WARP_SIZE,
    compute_war,
    compute_war_from_ids,
    compute_wartau,
    compute_wartau_per_adapter,
    compute_halign,
    compute_all_metrics,
    warp_alignment_breakdown,
    theoretical_war_random,
)
from adapter_slots.metrics.entropy import halign_upper_bound
from adapter_slots.metrics.gwar import compute_gwar, compute_gwar_curve

__all__ = [
    "WARP_SIZE",
    "compute_war",
    "compute_war_from_ids",
    "compute_wartau",
    "compute_wartau_per_adapter",
    "compute_halign",
    "compute_all_metrics",
    "warp_alignment_breakdown",
    "theoretical_war_random",
    "halign_upper_bound",
    "compute_gwar",
    "compute_gwar_curve",
]
