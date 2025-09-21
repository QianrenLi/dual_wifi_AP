from . import register_filter
from typing import Any, Union

Number = Union[int, float]

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float))

@register_filter
def scale_outage(value: Any, zeta: float = 1.0):
    """
    Compute zeta * outage_rate. Works for scalar or dict-of-scalars.
    """
    if _is_num(value):
        return zeta * float(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if _is_num(v):
                out[str(k)] = zeta * float(v)
        return out
    return None