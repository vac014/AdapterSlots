"""
vllm_serve_adapter_slots.py -- vLLM API server with AlignmentAwareScheduler.

Thin CLI wrapper around adapter_slots.integrations.aligned_api_server: parses
AdapterSlots-specific flags (--tmax, --wgkp-threshold) into the matching
AS_* env vars and forwards everything else to vLLM's own argument parser.

By default it runs the multiprocessing frontend (AlignmentAwareMQLLMEngine, see
aligned_mp_engine.py), the same process layout vanilla vLLM uses by default.
--disable-frontend-multiprocessing selects the in-process AlignmentAwareAsyncEngine
(aligned_engine.py) instead.

The if __name__ == "__main__" guard is required: multiprocessing.spawn
re-executes this file in the child process (with __name__ == "__mp_main__"),
and without the guard the child would try to bind the same port and crash.

Usage:
    python scripts/vllm_serve_adapter_slots.py \
        --model ./models/llama-7b \
        --enable-lora \
        --max-loras 16 \
        --max-lora-rank 16 \
        --lora-modules adapter_0=./adapters/adapter_r16_k0_s42 ... \
        --tensor-parallel-size 2 \
        --port 8000
"""

import logging
import os
import sys
from pathlib import Path

# Launched as a subprocess (python scripts/vllm_serve_adapter_slots.py), so
# sys.path[0] is this script's own dir, not the repo root where adapter_slots/
# lives; add it before the adapter_slots import below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# vLLM configures only its own loggers. This runs at import, so it also covers
# the engine subprocess, which re-imports this file.
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s", "%m-%d %H:%M:%S"))
logging.getLogger("adapter_slots").addHandler(_handler)
logging.getLogger("adapter_slots").setLevel(logging.INFO)

if __name__ == "__main__":
    # Translate AdapterSlots-only flags into env vars before vLLM's parser
    # sees argv (it would reject --tmax/--wgkp-threshold as unrecognized).
    clean: list = []
    raw = sys.argv[1:]
    i = 0
    while i < len(raw):
        arg = raw[i]
        if arg == "--tmax" and i + 1 < len(raw):
            os.environ["AS_TMAX_MS"] = raw[i + 1]
            i += 2
            continue
        if arg == "--wgkp-threshold" and i + 1 < len(raw):
            os.environ["AS_WGKP_THRESHOLD"] = raw[i + 1]
            i += 2
            continue
        if arg.startswith("--tmax="):
            os.environ["AS_TMAX_MS"] = arg.split("=", 1)[1]
            i += 1
            continue
        if arg.startswith("--wgkp-threshold="):
            os.environ["AS_WGKP_THRESHOLD"] = arg.split("=", 1)[1]
            i += 1
            continue
        clean.append(arg)
        i += 1
    sys.argv = [sys.argv[0]] + clean

    import uvloop
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils import FlexibleArgumentParser

    from adapter_slots.integrations.aligned_api_server import run_server

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))
