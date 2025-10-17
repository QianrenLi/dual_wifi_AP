# normalizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import time
import json
from pathlib import Path
from util.trace_watcher import TraceWatcher

EPS = 1e-8

def _to_2d(arr_list: List[List[float]]) -> np.ndarray:
    if not arr_list:
        return np.empty((0, 0), dtype=np.float32)
    x = np.asarray(arr_list, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    mask = np.isfinite(x).all(axis=1)
    return x[mask]

@dataclass
class _Affine:
    mean: np.ndarray  # (D,)
    std:  np.ndarray  # (D,)

    def transform_np(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + EPS)

class DatasetStateNormalizer:
    """
    Compute ONLY the state normalization from TraceWatcher.
    """
    def __init__(self, watcher: "TraceWatcher") -> None:
        self.watcher = watcher
        self.state: _Affine | None = None

    def _compute_affine(self, arr_list: List[List[float]]) -> _Affine:
        X = _to_2d(arr_list)
        if X.size == 0:
            return _Affine(mean=np.zeros((1,), dtype=np.float32),
                           std =np.ones((1,),  dtype=np.float32))
        mu = X.mean(axis=0).astype(np.float32)
        sd = X.std(axis=0, ddof=0).astype(np.float32)
        sd = np.where(sd < EPS, 1.0, sd).astype(np.float32)
        return _Affine(mean=mu, std=sd)

    def _merge_state(self, traces: List[Tuple[list, list, list, list]]) -> List[List[float]]:
        S_all: List[List[float]] = []
        for (s, _a, _r, _net) in traces:
            if s: S_all.extend(s)
        return S_all

    def fit_from_initial(self, poll_interval_sec: float = 1.0) -> None:
        traces = self.watcher.load_initial_traces()
        while traces == []:
            time.sleep(poll_interval_sec)
            traces = self.watcher.load_initial_traces()
        S_all = self._merge_state(traces)
        self.state = self._compute_affine(S_all)

    def save_state_json(self, outfile: str | Path, meta: Dict | None = None) -> str:
        """
        Save state normalization to JSON:
        {
          "state": {"mean": [...], "std": [...]},
          "dim": D,
          "created_at": "...",
          "meta": {...}
        }
        """
        assert self.state is not None, "Call fit_from_initial() first."
        p = Path(outfile)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "state": {
                "mean": self.state.mean.tolist(),
                "std":  self.state.std.tolist(),
            },
            "dim": int(self.state.mean.shape[0]),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meta": meta or {},
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p.resolve())

# --------------------
# Minimal CLI-style usage (example)
# --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_config", required=True)
    parser.add_argument("--trace_path", required=True)
    parser.add_argument("--out", default="state_transform.json")
    args = parser.parse_args()

    control_cfg = json.loads(Path(args.control_config).read_text(encoding="utf-8"))
    watcher = TraceWatcher(args.trace_path, control_cfg)

    norm = DatasetStateNormalizer(watcher)
    norm.fit_from_initial()

    out_path = norm.save_state_json(
        args.out,
        meta={"trace_path": str(Path(args.trace_path).resolve()),
              "control_config": str(Path(args.control_config).resolve())}
    )
    print(f"Saved state transform to: {out_path}")
