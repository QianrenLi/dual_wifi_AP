#!/usr/bin/env python3
import dataclasses
from dataclasses import dataclass
from typing import List, Any, Dict, Callable
import numpy as np

_JO_REGISTRY: Dict[str, Callable[[Any], Any]] = {}

def register_jo(name: str | None = None):
    def _decorator(obj):
        key = name or getattr(obj, "__jo_name__", None) or obj.__name__

        if hasattr(obj, "from_jo") and callable(getattr(obj, "from_jo")):
            _JO_REGISTRY[key] = lambda data, _cls=obj: _cls.from_jo(data)
        else:
            _JO_REGISTRY[key] = lambda data, _cls=obj: _cls(data)
        return obj
    return _decorator

def revive_jsonlike(obj: Any) -> Any:
    if isinstance(obj, dict):
        # First, recursively revive children
        revived = {k: revive_jsonlike(v) for k, v in obj.items()}
        # Then, give the dict to your hook (so it can detect __class__/__data__)
        return _json_object_hook(revived)
    elif isinstance(obj, list):
        return [revive_jsonlike(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(revive_jsonlike(v) for v in obj)
    else:
        return obj

def _json_default(o):
    if hasattr(o, "to_jsonable"):
        return {"__class__": o.__class__.__name__, "__data__": o.to_jsonable()}
    if dataclasses.is_dataclass(o):
        return {"__class__": o.__class__.__name__,
                "__data__": {f.name: getattr(o, f.name) for f in dataclasses.fields(o)}}
    raise TypeError

def _json_object_hook(dct):
    if "__class__" in dct and "__data__" in dct:
        clsname, data = dct["__class__"], dct["__data__"]
        # Note: `data` is already decoded (object_hook is recursive).
        maker = _JO_REGISTRY.get(clsname)
        if maker:
            return maker(data)
        # fallback: return raw
        return {"__class__": clsname, "__data__": data}
    return dct

@register_jo()
class CListFloat:
    """Base class: a list of floats with constraints."""
    dim: int = None
    value_range: tuple = (None, None)  # (low, high)

    def __init__(self, values: List[float]):
        if not isinstance(values, list):
            raise TypeError("Must be a list of floats")
        if len(values) != self.dim:
            raise ValueError(f"Must have exactly {self.dim} elements")
        lo, hi = self.value_range
        clipped = []
        for value in values:
            value = value * (hi - lo) / 2 + (hi + lo) / 2
            if lo is not None and value < lo:
                value = lo
            if hi is not None and value > hi:
                value = hi
            clipped.append(value)
            
        self.values = clipped

    def to_jsonable(self):
        return self.values

    def __repr__(self):
        return f"{self.__class__.__name__}({self.values})"
    
@register_jo()
class CInt:
    """Constrained integer with optional range."""
    dim: int = 1
    value_range: tuple = (None, None)  # (low, high)

    def __init__(self, value: int):
        if isinstance(value, list):
            value = value[0]
            
        lo, hi = self.value_range
        value = value * (hi - lo) / 2 + (hi + lo) / 2
        
        if lo is not None and value < lo:
            value = lo
        if hi is not None and value > hi:
            value = hi

        self.value = int(value)

    def to_jsonable(self):
        return self.value

    def __int__(self):
        return self.value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"
    
    
@register_jo()
class CIntTypeII:
    """Constrained integer with optional range."""
    dim: int = 1
    value_range: tuple = (None, None)  # (low, high)

    def __init__(self, value: int):
        if isinstance(value, list):
            value = value[0]
            
        lo, hi = self.value_range
        if lo is not None and hi is not None:
            if value < 0:
                value = hi + value * (hi - lo) / 2
            else:
                value = lo + value * (hi - lo) / 2
                
        self.value = int(value)

    def to_jsonable(self):
        return self.value

    def __int__(self):
        return self.value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"
    
    
@register_jo()
class C_LIST_FLOAT_DIM4_0_500(CListFloat):
    dim = 4
    value_range = (0.0, 500)
    
@register_jo()
class C_LIST_FLOAT_DIM3_0_500(CListFloat):
    dim = 3
    value_range = (0.0, 500)
    
@register_jo()
class C_LIST_FLOAT_DIM3_0_10(CListFloat):
    dim = 3
    value_range = (0.0, 10)

@register_jo()
class C_INT_RANGE_0_13(CInt):
    value_range = (0, 13)
    
@register_jo()
class C_INT_RANGE_0_13_TypeII(CIntTypeII):
    value_range = (0, 13)

@register_jo()
@dataclass
class ControlCmd:
    policy_parameters: C_LIST_FLOAT_DIM3_0_10
    version: C_INT_RANGE_0_13
    
    @staticmethod
    def __dim__():
        return sum(field.type.dim for field in ControlCmd.__dataclass_fields__.values())
    
    def __iter__(self):
        for field in self.__dataclass_fields__.values():
            yield field.name, getattr(self, field.name).to_jsonable()
            
    @classmethod
    def from_jo(cls, data: dict) -> "ControlCmd":
        return cls(
            policy_parameters=data["policy_parameters"],
            version=data["version"],
        )

    @property
    def value(self):
        value = cmd_to_list(self)
        return value
    
    @staticmethod
    def sum_log_scales() -> float:
        """
        Sum of log scale factors for tanh->linear mapping per continuous action dim:
            a = b + s * u,  where s = (hi - lo) / 2
        Only counts continuous (CListFloat) fields. Integers are excluded.
        """
        C = 0.0
        for name, field in ControlCmd.__dataclass_fields__.items():
            ftype = field.type
            # Only include continuous list-of-floats fields
            if not isinstance(ftype, type) or not issubclass(ftype, CListFloat):
                continue
            lo, hi = ftype.value_range
            if lo is None or hi is None:
                raise ValueError(f"Field '{name}' has open range {ftype.value_range}; cannot compute scale.")
            s = (hi - lo) / 2.0
            if s <= 0:
                raise ValueError(f"Field '{name}' has non-positive scale (lo={lo}, hi={hi}).")
            C += ftype.dim * np.log(s)
        return C
        
            
            
def cmd_to_list(cmd: ControlCmd) -> List[float]:
    """Convert ControlCmd to a flat list of floats."""
    result = []
    for field in cmd.__dataclass_fields__.values():
        value = getattr(cmd, field.name)
        if isinstance(value, CListFloat):
            result.extend(value.values)
        elif isinstance(value, CInt):
            result.append(float(value.value))
        else:
            raise TypeError(f"Unsupported field type: {type(value)}")
    return result


def list_to_cmd(cls, lst: List[float]):
    """Reconstruct from flat list."""
    out_kwargs: dict[str, Any] = {}
    idx = 0
    for field in cls.__dataclass_fields__.values():
        dim = field.type.dim
        seg = lst[idx: idx + dim]
        idx += dim
        out_kwargs[field.name] = field.type(seg)
    return cls(**out_kwargs)

if __name__ == "__main__":
    print(C_INT_RANGE_0_13_TypeII(-0.1))
    print(C_INT_RANGE_0_13_TypeII(0.1))
    print(C_INT_RANGE_0_13(0.5))