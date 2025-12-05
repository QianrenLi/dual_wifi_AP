from . import register_filter
from typing import Any, Dict, Union

Number = Union[int, float]

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float))

@register_filter
def bitrate_delta(value: Any, alpha: float = 1.0, beta: float = 0.0, offset = 0.0):
    """
    Compute alpha * bitrate + beta * Δbitrate with memory.
    Works for a single number or a dict of numbers, keeping per-key state.
    """
    # per-callable memory
    if not hasattr(bitrate_delta, "_prev"):
        bitrate_delta._prev = {}  # type: ignore[attr-defined]

    prev_store: Dict[str, Number] = bitrate_delta._prev  # type: ignore[attr-defined]

    # scalar
    if _is_num(value):
        prev = prev_store.get("__scalar__")
        delta = 0.0 if prev is None else abs(float(value) - float(prev))
        prev_store["__scalar__"] = float(value)
        return alpha * float(value) + beta * delta + offset

    # mapping (e.g., per-flow)
    if isinstance(value, dict):
        out: Dict[str, Number] = {}
        for k, v in value.items():
            if not _is_num(v):
                # ignore non-numeric leaves
                continue
            key = str(k)
            prev = prev_store.get(key)
            delta = 0.0 if prev is None else abs(float(v) - float(prev))
            prev_store[key] = float(v)
            out[key] = alpha * float(v) + beta * delta + offset
        return out

    # unsupported shape
    return None


@register_filter
def stat_bitrate(value: Any, alpha: float = 1e-6):
    """
    Compute alpha * bitrate . Works for scalar or dict-of-scalars.
    """
    if _is_num(value):
        return alpha * float(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if _is_num(v):
                out[str(k)] = alpha * float(v)
        return out
    return None
