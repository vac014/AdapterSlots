"""
batch_logger.py -- Per-batch metric event logger.

Records WAR, WARτ, H_align, and SGMV intensity for every dispatched batch.
Supports three backends: stdout, csv, and prometheus (via PrometheusExporter).

Output JSONL format (one line per batch):
    {"tick": 42, "t_ms": 12.345, "war": 0.750, "wartau_ms": 18.2,
     "halign": 0.821, "batch_size": 128, "k_active": 4,
     "sgmv_intensity": {"0": 48.5, "1": 12.0}}
"""

import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from adapter_slots.metrics.war import (
    compute_war,
    compute_wartau,
    compute_halign,
    WARP_SIZE,
)


@dataclass
class BatchEvent:
    tick_id: int
    timestamp_ms: float
    batch: List[Dict]               # list of {"adapter_id": int, "arrival_time_ms": float}
    war: float
    wartau_ms: float
    halign: float
    throughput_estimate_tokps: float
    sgmv_intensity: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("batch")              # batch tokens are too large to log verbatim
        return d


class BatchLogger:
    """
    Intercepts batch formation events, computes metrics, and emits to a backend.

    Args:
        backend:      "stdout" | "csv" | "jsonl" | "prometheus"
        output_path:  File path for "csv" or "jsonl" backends (ignored otherwise).
        warp_size:    GPU warp size (default 32).
        prometheus_exporter: Optional PrometheusExporter instance.
    """

    def __init__(
        self,
        backend: str = "stdout",
        output_path: Optional[str] = None,
        warp_size: int = WARP_SIZE,
        prometheus_exporter=None,
    ):
        self.backend = backend
        self.warp_size = warp_size
        self._prometheus = prometheus_exporter
        self._tick = 0
        self._prev_dispatch_ms: Optional[float] = None
        self._prev_tokens: int = 0

        self._csv_file = None
        self._csv_writer = None
        self._jsonl_file = None

        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            if backend == "csv":
                self._csv_file = open(output_path, "w", newline="")
                fieldnames = [
                    "tick_id", "timestamp_ms", "war", "wartau_ms", "halign",
                    "batch_size", "k_active", "throughput_estimate_tokps",
                ]
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
                self._csv_writer.writeheader()
            elif backend == "jsonl":
                self._jsonl_file = open(output_path, "w")

    def log_batch(
        self,
        batch_tokens: List[Dict],
        dispatch_time_ms: Optional[float] = None,
        sgmv_intensity: Optional[Dict[str, float]] = None,
    ) -> BatchEvent:
        """
        Compute metrics for a dispatched batch and emit to the configured backend.

        Args:
            batch_tokens:     List of token dicts with "adapter_id" (and optionally
                              "arrival_time_ms").
            dispatch_time_ms: Wall-clock time of dispatch in ms. Defaults to now.
            sgmv_intensity:   Pre-computed SGMV intensity dict (optional).

        Returns:
            The BatchEvent that was logged.
        """
        if dispatch_time_ms is None:
            dispatch_time_ms = time.monotonic() * 1000.0

        war = compute_war(batch_tokens, self.warp_size)
        wartau = compute_wartau(batch_tokens, dispatch_time_ms)
        halign = compute_halign(batch_tokens, self.warp_size)

        # Throughput estimate: tokens / elapsed since last dispatch
        n_tokens = len(batch_tokens)
        if self._prev_dispatch_ms is not None and dispatch_time_ms > self._prev_dispatch_ms:
            elapsed_s = (dispatch_time_ms - self._prev_dispatch_ms) / 1000.0
            tput = self._prev_tokens / elapsed_s
        else:
            tput = 0.0

        k_active = len({t["adapter_id"] for t in batch_tokens})

        event = BatchEvent(
            tick_id=self._tick,
            timestamp_ms=dispatch_time_ms,
            batch=batch_tokens,
            war=war,
            wartau_ms=wartau,
            halign=halign,
            throughput_estimate_tokps=tput,
            sgmv_intensity=sgmv_intensity or {},
        )

        self._emit(event, n_tokens, k_active)

        self._tick += 1
        self._prev_dispatch_ms = dispatch_time_ms
        self._prev_tokens = n_tokens
        return event

    def _emit(self, event: BatchEvent, n_tokens: int, k_active: int) -> None:
        if self.backend == "stdout":
            print(
                f"tick={event.tick_id:5d}  t={event.timestamp_ms:10.1f}ms  "
                f"WAR={event.war:.3f}  WARτ={event.wartau_ms:.1f}ms  "
                f"H={event.halign:.3f}  N={n_tokens}  K={k_active}  "
                f"tput={event.throughput_estimate_tokps:.0f}tok/s"
            )

        elif self.backend == "csv" and self._csv_writer:
            row = event.to_dict()
            row["batch_size"] = n_tokens
            row["k_active"] = k_active
            self._csv_writer.writerow({k: row[k] for k in self._csv_writer.fieldnames})
            self._csv_file.flush()

        elif self.backend == "jsonl" and self._jsonl_file:
            record = event.to_dict()
            record["batch_size"] = n_tokens
            record["k_active"] = k_active
            self._jsonl_file.write(json.dumps(record) + "\n")
            self._jsonl_file.flush()

        elif self.backend == "prometheus" and self._prometheus is not None:
            self._prometheus.update(event, n_tokens, k_active)

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
        if self._jsonl_file:
            self._jsonl_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
