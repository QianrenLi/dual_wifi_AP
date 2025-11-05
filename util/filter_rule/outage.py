from . import register_filter
from typing import Any, Union

Number = Union[int, float]

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float))

@register_filter
def scale_outage(value: Any, zeta: float = 1.0, minimum = None):
    """
    Compute zeta * outage_rate. Works for scalar or dict-of-scalars.
    """
    def clip_minimum(val, minimum):
        if minimum == None:
            return val
        else:
            return max(val, minimum)
        
    if _is_num(value):
        return clip_minimum(zeta * float(value), minimum=minimum)
    
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if _is_num(v):
                out[str(k)] = clip_minimum(zeta * float(v), minimum=minimum)
        return out
    return None