import json
import numpy as np
from pathlib import Path
from typing import List

EPS = 1e-8

class _StateTransform:
    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std  = np.asarray(std,  dtype=np.float32)
        self.std  = np.where(self.std < EPS, 1.0, self.std).astype(np.float32)
        self.dim  = int(self.mean.shape[0])

    @classmethod
    def from_json(cls, path: str | Path) -> "_StateTransform":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        st = obj.get("state", {})
        mean = st.get("mean", [])
        std  = st.get("std", [])
        return cls(mean, std)

    def apply_to_list(self, x_list: List[float]) -> List[float]:
        if not x_list:
            return x_list
        x = np.asarray(x_list, dtype=np.float32)
        # handle length mismatch gracefully
        D = min(x.shape[0], self.dim)
        x[:D] = (x[:D] - self.mean[:D]) / (self.std[:D] + EPS)
        return x.astype(np.float32).tolist()