"""
backends/ -- Server backend wrappers for AdapterSlots benchmarks.

Each backend starts a serving server, runs requests against it and tears it
down cleanly.

Usage:
    from backends import get_backend
    bkd = get_backend("adapterslots", model=..., adapter_dirs=..., port=8100, ...)
    bkd.start()
    try:
        url, payload = bkd.build_request_payload(prompt, adapter_id, max_tokens)
        ...
    finally:
        bkd.stop()
"""

from backends.base import BaseBackend

# S-LoRA depends on the vendored deps/slora install; a missing install must not
# break vLLM or AdapterSlots runs, so its import failure is recorded, not raised.
_REGISTRY = {}
_UNAVAILABLE = {}

from backends.backend_adapterslots import AdapterSlotsBackend
_REGISTRY["adapterslots"] = AdapterSlotsBackend

from backends.backend_vllm import VLLMBackend
_REGISTRY["vllm"] = VLLMBackend

try:
    from backends.backend_slora import SLoRABackend
    _REGISTRY["slora"] = SLoRABackend
except ImportError as e:
    _UNAVAILABLE["slora"] = str(e)

def get_backend(name: str, **kwargs) -> BaseBackend:
    """Instantiate a backend by name.

    Args:
        name:    One of "adapterslots", "vllm", "slora".
        **kwargs: Passed verbatim to the backend constructor.

    Returns:
        A BaseBackend instance (not yet started).
    """
    name = name.lower()
    if name in _UNAVAILABLE:
        raise ImportError(
            f"Backend {name!r} is registered but its dependency failed to import: "
            f"{_UNAVAILABLE[name]}. Install deps/{name} (see "
            f"backends/backend_{name}.py) before using this backend."
        )
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend: {name!r}. Valid choices: "
            f"{sorted(set(_REGISTRY) | set(_UNAVAILABLE))}"
        )
    return _REGISTRY[name](**kwargs)
