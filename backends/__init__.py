"""
backends/ -- Server backend wrappers for AdapterSlots benchmarks.

Each backend starts a real serving server, runs requests against it,
and tears it down cleanly. No simulation or calibrated fallbacks.

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

# Per-backend import guards
# punica/slora/dlora each depend on a vendored deps/ install (CUDA extensions
# for punica/slora, a separate clone for dlora). A missing install must not
# break pure-vLLM or pure-AdapterSlots runs, so each import failure is isolated here
# rather than failing the whole `backends` package -- mirrors the optional-
# import pattern in adapter_slots/integrations/vllm_scheduler.py.
_REGISTRY = {}
_UNAVAILABLE = {}

from backends.backend_adapterslots import AdapterSlotsBackend
_REGISTRY["adapterslots"] = AdapterSlotsBackend

from backends.backend_vllm import VLLMBackend
_REGISTRY["vllm"] = VLLMBackend

try:
    from backends.backend_punica import PunicaBackend
    _REGISTRY["punica"] = PunicaBackend
except ImportError as e:
    _UNAVAILABLE["punica"] = str(e)

try:
    from backends.backend_slora import SLoRABackend
    _REGISTRY["slora"] = SLoRABackend
except ImportError as e:
    _UNAVAILABLE["slora"] = str(e)

try:
    from backends.backend_dlora import DLoRABackend
    _REGISTRY["dlora"] = DLoRABackend
except ImportError as e:
    _UNAVAILABLE["dlora"] = str(e)


def get_backend(name: str, **kwargs) -> BaseBackend:
    """Instantiate a backend by name.

    Args:
        name:    One of "adapterslots", "vllm", "punica", "slora", "dlora".
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
