"""
aligned_api_server.py -- vLLM OpenAI-compatible API server using
AlignmentAware* engines.

Reuses vLLM's own FastAPI app/routes (build_app, init_app_state) and HTTP
server (serve_http) unmodified. Two engine_client backends are supported:

1. Multiprocessing (default, matches vanilla vLLM's default architecture):
   spawns AlignmentAwareMQLLMEngine (aligned_mp_engine.py) in a subprocess,
   talks to it over the same ZMQ protocol vLLM's own MQLLMEngineClient
   uses -- mirrors the multiprocessing branch of vllm.entrypoints.openai.
   api_server.build_async_engine_client_from_engine_args almost verbatim,
   with only the spawned function swapped.
2. In-process (--disable-frontend-multiprocessing): constructs
   AlignmentAwareAsyncEngine (aligned_engine.py) directly in this process.

Why both exist: an in-process-only engine forces AS to run HTTP + tokenization +
detokenization + scheduling in one process, while vanilla vLLM's frontend process
is isolated from its engine subprocess. Under saturation (real workload, K=8
adapters, 13B model, overload conditions) that single difference, and not scheduler
CPU cost, buffer-wait latency, CASH batch-holdback, or the Level-2/3 kernel paths,
is what costs AS a ~10% decode-throughput tax against vanilla vLLM. The
multiprocessing path (aligned_mp_engine.py) restores the isolation, and it does so
the same way aligned_engine.py supports a custom scheduler in-process: subclass,
don't monkeypatch. Keeping the in-process path is what makes the two comparable.

No vLLM module attribute is ever reassigned in either path.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import socket
import tempfile

import uvloop
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    TIMEOUT_KEEP_ALIVE,
    build_app,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_open_zmq_ipc_path

from adapter_slots.integrations.aligned_engine import AlignmentAwareAsyncEngine
from adapter_slots.integrations.aligned_mp_engine import run_mp_engine_aligned


async def _build_engine_client_mp(engine_args, engine_config):
    """Multiprocessing path: spawn AlignmentAwareMQLLMEngine, connect over ZMQ.

    Mirrors vllm.entrypoints.openai.api_server.
    build_async_engine_client_from_engine_args's multiprocessing branch
    (vllm==0.6.3), with run_mp_engine -> run_mp_engine_aligned as the only
    change. Returns (engine_client, engine_process) -- caller is responsible
    for terminate/join/kill + client.close() on shutdown.
    """
    import asyncio

    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
        # Keep the TemporaryDirectory object alive for the server's lifetime
        # by stashing it on the function object -- same lifecycle vLLM's own
        # module-level `prometheus_multiproc_dir` global gives it.
        _build_engine_client_mp._tmpdir = prometheus_multiproc_dir

    ipc_path = get_open_zmq_ipc_path()
    context = multiprocessing.get_context("spawn")
    engine_process = context.Process(
        target=run_mp_engine_aligned,
        args=(engine_args, UsageContext.OPENAI_API_SERVER, ipc_path),
    )
    engine_process.start()

    mp_engine_client = MQLLMEngineClient(ipc_path, engine_config)
    while True:
        try:
            await mp_engine_client.setup()
            break
        except TimeoutError:
            if not engine_process.is_alive():
                raise RuntimeError("Engine process failed to start") from None

    return mp_engine_client, engine_process


async def run_server(args) -> None:
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_config = engine_args.create_engine_config()
    if getattr(AsyncLLMEngine._get_executor_cls(engine_config), "uses_ray", False):
        raise RuntimeError("aligned_api_server does not support Ray executors.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", args.port))

    def signal_handler(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    use_mp = (
        not args.disable_frontend_multiprocessing
        and not MQLLMEngineClient.is_unsupported_config(engine_args)
    )

    engine_process = None
    if use_mp:
        engine_client, engine_process = await _build_engine_client_mp(
            engine_args, engine_config
        )
    else:
        engine_client = AlignmentAwareAsyncEngine.from_engine_args(
            engine_args=engine_args,
            engine_config=engine_config,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )
        # Give each scheduler instance the lora_name -> on-disk path mapping
        # from --lora-modules, so AlignmentAwareScheduler._get_lora_weights()
        # can read the real adapter checkpoint for MergedWeightCache
        # promotion. Only reachable in-process -- the mp path's engine lives
        # in a subprocess with no direct attribute access. Harmless to skip
        # there: _get_lora_weights() falling back to {} only disables Level-3
        # promotion, which AlignmentAwareModelRunner.execute_model() already
        # never dispatches to a kernel (see model_runner.py) -- no behavior
        # difference between the two paths results from this.
        lora_paths = {lm.name: lm.path for lm in (args.lora_modules or [])}
        for sched in engine_client.engine.scheduler:
            sched._lora_paths = lora_paths

    try:
        app = build_app(args)
        model_config = await engine_client.get_model_config()
        init_app_state(engine_client, model_config, app.state, args)

        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            fd=sock.fileno(),
        )
        await shutdown_task
    finally:
        if engine_process is not None:
            engine_process.terminate()
            engine_client.close()
            engine_process.join(4)
            if engine_process.exitcode is None:
                engine_process.kill()
            from prometheus_client import multiprocess
            multiprocess.mark_process_dead(engine_process.pid)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-compatible API server with AlignmentAwareScheduler.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))
