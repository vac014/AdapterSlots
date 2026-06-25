"""
prometheus_exporter.py -- Prometheus metrics exporter for AdapterSlots.

Exposes WAR, WARτ, H_align, throughput, and TTFT as Prometheus gauges.
Requires: pip install prometheus-client

Usage:
    from adapter_slots.instrumentation.prometheus_exporter import PrometheusExporter
    exporter = PrometheusExporter(port=9091)
    exporter.start()
    # then pass exporter to BatchLogger(backend="prometheus", prometheus_exporter=exporter)
"""

import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from adapter_slots.instrumentation.batch_logger import BatchEvent


class PrometheusExporter:
    """
    Prometheus Pushgateway / HTTP server exporter.

    Args:
        port:        Port to expose the /metrics HTTP endpoint on.
        pushgateway: If set, push to this Pushgateway URL instead of serving.
        job_name:    Job label for Pushgateway.
    """

    def __init__(
        self,
        port: int = 9091,
        pushgateway: Optional[str] = None,
        job_name: str = "adapter_slots",
    ):
        self.port = port
        self.pushgateway = pushgateway
        self.job_name = job_name
        self._started = False
        self._gauges = {}

        try:
            import prometheus_client as prom
            self._prom = prom
            self._gauges = {
                "war": prom.Gauge("adapter_slots_war", "Warp Alignment Ratio"),
                "wartau_ms": prom.Gauge("adapter_slots_wartau_ms", "WARτ mean misalignment age (ms)"),
                "halign": prom.Gauge("adapter_slots_halign", "H_align alignment entropy (bits)"),
                "throughput_tokps": prom.Gauge(
                    "adapter_slots_throughput_tokps", "Estimated throughput (tokens/s)"
                ),
                "batch_size": prom.Gauge("adapter_slots_batch_size", "Batch size (tokens)"),
                "k_active": prom.Gauge("adapter_slots_k_active", "Active adapters in batch"),
            }
            self._available = True
        except ImportError:
            self._available = False
            self._prom = None

    def start(self) -> None:
        """Start the HTTP metrics server (non-blocking)."""
        if not self._available:
            raise RuntimeError(
                "prometheus_client is not installed. "
                "Run: pip install prometheus-client"
            )
        if self._started:
            return
        self._prom.start_http_server(self.port)
        self._started = True

    def update(self, event: "BatchEvent", n_tokens: int, k_active: int) -> None:
        """Push latest metric values. Called by BatchLogger on each batch."""
        if not self._available:
            return

        self._gauges["war"].set(event.war)
        self._gauges["wartau_ms"].set(event.wartau_ms)
        self._gauges["halign"].set(event.halign)
        self._gauges["throughput_tokps"].set(event.throughput_estimate_tokps)
        self._gauges["batch_size"].set(n_tokens)
        self._gauges["k_active"].set(k_active)

        if self.pushgateway:
            self._prom.push_to_gateway(
                self.pushgateway,
                job=self.job_name,
                registry=self._prom.REGISTRY,
            )

    def is_available(self) -> bool:
        return self._available
