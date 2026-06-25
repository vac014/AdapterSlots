"""
apis_router.py -- APISRouter: Adapter-Partitioned Independent Serving router (kernel_promotion).

APIS replaces TP=2 tensor parallelism with two independent LLaMA-7B instances,
each serving K/2 adapters. This eliminates the PCIe allreduce bottleneck that
limits τ_iter to ~100ms, dropping it to ~30ms per GPU and enabling fine-grained
WAR control at K=10–25 (§2.6, kernel_promotion).

Routing strategy: Zipf-balanced round-robin.
    sorted_adapters = sorted by λ̂_k descending (most popular first)
    gpu_assignment[adapter_i] = i % n_gpus
This ensures each GPU handles approximately equal total traffic under Zipf α=0.9.

HTTP integration: APISRouter is plugged into the existing HTTP proxy infrastructure
(proxy.py, infrastructure). When AS_APIS_ENABLED=1, the proxy's upstream selection calls
APISRouter.route() instead of the default round-robin.

Rebalancing: every AS_APIS_REBALANCE_S seconds (default 30s), rebalance() is called
with the current EWMA rate estimates from ArrivalRateEstimator. For static Zipf
workloads, the initial assignment is stable and rebalancing is rarely needed.

Standalone usage (as HTTP proxy, AS_APIS_ENABLED=1):
    python -m adapter_slots.integrations.apis_router --port 8000

Integration usage (via proxy.py):
    export AS_APIS_ENABLED=1
    export AS_APIS_UPSTREAM_URLS="http://localhost:8001,http://localhost:8002"
    python scripts/vllm_serve_adapter_slots.py [vllm args]
"""

import json
import os
import time
import threading
from typing import Dict, List, Optional

# Optional aiohttp for HTTP proxy mode
try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False


class APISRouter:
    """Adapter-Partitioned Independent Serving router.

    Assigns adapters to GPU shards using Zipf-balanced round-robin and serves
    as the HTTP routing layer for multi-GPU APIS deployments.

    Args:
        n_gpus:            Number of independent GPU shards (vLLM instances).
        upstream_urls:     List of upstream vLLM server URLs, one per GPU.
        rebalance_interval_s: Seconds between APIS load rebalancing.
    """

    def __init__(
        self,
        n_gpus: int = 2,
        upstream_urls: Optional[List[str]] = None,
        rebalance_interval_s: float = 30.0,
    ) -> None:
        if n_gpus < 1:
            raise ValueError(f"n_gpus must be >= 1, got {n_gpus}")
        self._n_gpus = n_gpus
        self._upstream_urls: List[str] = upstream_urls or [
            f"http://localhost:{8001 + i}" for i in range(n_gpus)
        ]
        if len(self._upstream_urls) != n_gpus:
            raise ValueError(
                f"upstream_urls length ({len(self._upstream_urls)}) must equal "
                f"n_gpus ({n_gpus})"
            )
        self.rebalance_interval_s = rebalance_interval_s

        # adapter_id (str) → GPU index (int)
        self._assignment: Dict[str, int] = {}
        self._lock = threading.RLock()

        # Stats
        self._route_counts: Dict[int, int] = {i: 0 for i in range(n_gpus)}
        self._last_rebalance: float = time.time()
        self._total_routes: int = 0

    # Public API

    def route(self, adapter_id: str) -> str:
        """Return the upstream URL for the given adapter_id.

        O(1) dict lookup. Falls back to consistent hashing if adapter not
        yet assigned (happens before first rebalance call).

        Args:
            adapter_id: String identifier of the requested adapter.

        Returns:
            Upstream URL string (e.g., "http://localhost:8001").
        """
        with self._lock:
            gpu_idx = self._assignment.get(
                adapter_id,
                hash(adapter_id) % self._n_gpus,
            )
            self._route_counts[gpu_idx] = self._route_counts.get(gpu_idx, 0) + 1
            self._total_routes += 1
            return self._upstream_urls[gpu_idx]

    def rebalance(self, lambda_k_dict: Dict[str, float]) -> None:
        """Update adapter-to-GPU assignment based on current EWMA rate estimates.

        Sorted by λ̂_k descending; assigned round-robin across GPUs. This ensures
        balanced traffic under Zipf: the highest-rate adapter goes to GPU 0,
        second-highest to GPU 1, third-highest to GPU 0, etc.

        Args:
            lambda_k_dict: {adapter_id: estimated_arrival_rate}
        """
        sorted_adapters = sorted(
            lambda_k_dict.keys(),
            key=lambda k: lambda_k_dict.get(k, 0.0),
            reverse=True,
        )
        with self._lock:
            self._assignment = {
                adapter_id: i % self._n_gpus
                for i, adapter_id in enumerate(sorted_adapters)
            }
            self._last_rebalance = time.time()

    def assignment_table(self) -> Dict[str, int]:
        """Return current adapter → GPU index mapping (copy)."""
        with self._lock:
            return dict(self._assignment)

    def load_balance_ratio(self) -> float:
        """Return max/min route count ratio across GPUs (1.0 = perfectly balanced).

        Values > 1.30 indicate load imbalance that may warrant rebalancing.
        """
        with self._lock:
            counts = list(self._route_counts.values())
        if min(counts) == 0:
            return float("inf")
        return max(counts) / min(counts)

    def maybe_rebalance(self, lambda_k_dict: Dict[str, float]) -> bool:
        """Rebalance if the interval has elapsed. Returns True if rebalanced."""
        if time.time() - self._last_rebalance >= self.rebalance_interval_s:
            self.rebalance(lambda_k_dict)
            return True
        return False

    def stats(self) -> dict:
        """Return router statistics."""
        with self._lock:
            return {
                "n_gpus": self._n_gpus,
                "n_assigned_adapters": len(self._assignment),
                "route_counts": dict(self._route_counts),
                "total_routes": self._total_routes,
                "load_balance_ratio": self.load_balance_ratio(),
                "seconds_since_rebalance": time.time() - self._last_rebalance,
            }


# Standalone HTTP proxy entry point

def _run_proxy_server(port: int, router: "APISRouter") -> None:
    """Run APISRouter as a standalone async HTTP proxy (requires aiohttp)."""
    if not _AIOHTTP_AVAILABLE:
        raise RuntimeError(
            "aiohttp is required to run the APIS proxy server. "
            "Install with: pip install aiohttp"
        )

    import asyncio
    from aiohttp import web, ClientSession

    async def handle_request(request: web.Request) -> web.Response:
        adapter_id = request.headers.get("X-Adapter-ID", "")
        if not adapter_id:
            body = await request.json()
            adapter_id = body.get("model", "")

        upstream = router.route(adapter_id)
        url = upstream + str(request.rel_url)

        async with ClientSession() as session:
            async with session.request(
                method=request.method,
                url=url,
                headers=request.headers,
                data=await request.read(),
            ) as resp:
                body = await resp.read()
                return web.Response(
                    status=resp.status,
                    headers=dict(resp.headers),
                    body=body,
                )

    async def run():
        app = web.Application()
        app.router.add_route("*", "/{path_info:.*}", handle_request)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        print(f"[APIS] Proxy server listening on port {port}")
        await site.start()
        while True:
            await asyncio.sleep(3600)

    asyncio.run(run())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="APIS HTTP Router proxy")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n-gpus", type=int,
                        default=int(os.environ.get("AS_APIS_N_GPUS", "2")))
    parser.add_argument("--upstream-urls", type=str,
                        default=os.environ.get("AS_APIS_UPSTREAM_URLS", ""))
    args = parser.parse_args()

    urls = [u.strip() for u in args.upstream_urls.split(",") if u.strip()]
    if not urls:
        urls = [f"http://localhost:{8001 + i}" for i in range(args.n_gpus)]

    router = APISRouter(n_gpus=args.n_gpus, upstream_urls=urls)
    _run_proxy_server(args.port, router)
