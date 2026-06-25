"""Streaming completion request shared by the benchmark clients.

A non-streaming request returns only when the whole completion is done, so the
time it returns is end-to-end latency, not time to first token. Streaming
gives one chunk per generated token: the first chunk with text marks the first
token, each later one a token time, and the final usage chunk carries the
completion token count.
"""
from __future__ import annotations

import json
import time


async def stream_completion(session, url: str, payload: dict, timeout_s: float,
                            collector, req_id: str) -> int:
    """POST a streaming completion; record token times on ``collector``.

    Returns completion_tokens from the usage chunk, or the number of text
    chunks if the server sent no usage.
    """
    import aiohttp

    body = dict(payload, stream=True, stream_options={"include_usage": True})
    n_usage, n_chunks = None, 0
    async with session.post(url, json=body,
                            timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage")
            if usage:
                n_usage = usage.get("completion_tokens", n_usage)
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.perf_counter()
                if n_chunks == 0:
                    collector.record_first_token(req_id, now)
                else:
                    collector.record_token(req_id, now)
                n_chunks += 1
    return n_usage if n_usage is not None else n_chunks
