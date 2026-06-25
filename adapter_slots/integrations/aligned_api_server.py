"""
aligned_api_server.py -- vLLM's OpenAI-compatible API server on the
AlignmentAware engines.

vLLM's FastAPI app, routes and HTTP server are used unchanged. Two engine
clients are supported:

1. Multiprocessing (default, vLLM's own default layout): AlignmentAwareMQLLMEngine
   (aligned_mp_engine.py) runs in a subprocess and talks ZMQ to vLLM's
   MQLLMEngineClient. This follows the multiprocessing branch of
   vllm.entrypoints.openai.api_server.build_async_engine_client_from_engine_args
   with only the spawned function swapped.
2. In-process (--disable-frontend-multiprocessing): AlignmentAwareAsyncEngine
   (aligned_engine.py) in this process.

Both paths subclass vLLM classes; no vLLM module attribute is reassigned.
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
    """Spawn AlignmentAwareMQLLMEngine and connect to it over ZMQ.

    Returns (engine_client, engine_process); the caller shuts both down.
    """

    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
        # Kept alive for the server's lifetime, as vLLM's module global is.
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
