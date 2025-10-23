# util/trace_watcher.py
import re

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Union, Optional

from util.trace_collec import trace_collec, flatten_leaves
from net_util.state_transfom import _StateTransform

class TraceWatcher:
    """
    Watch a single root directory and treat every '*.jsonl' file under it
    (recursively) as a trace unit.

    max_step:
      Maximum number of trace-units to process per call. If None, process all.

    API:
      - load_initial_traces() -> List[(s, a, r, net)]
        Loads up to `max_step` existing jsonl traces (or unlimited if None) and
        marks ONLY those as seen.

      - poll_new_paths() -> List[Path]
        Returns up to `max_step` newly appeared jsonl files since last check
        (and marks ONLY those as seen).

      - poll_new_traces() -> List[(s, a, r, net)]
        Convenience: load newly appeared jsonl files (up to `max_step`) and return merged traces.
    """

    def __init__(self, root: Union[str, Path], control_config: dict, max_step: Optional[int] = None) -> None:
        self.root: Path = Path(root).resolve()
        self.control_config = control_config
        self.max_step: Optional[int] = max_step

        self._state_tf: Optional[_StateTransform] = None
        if 'state_transform_dict' in control_config and control_config['state_transform_dict']:
            try:
                self._state_tf = _StateTransform.from_obj(control_config['state_transform_dict'])
            except Exception as e:
                # Fall back silently (or log if you prefer)
                print(f"[PolicyBase] Failed to load state transform: {e}")
                self._state_tf = None

        self.id_regex = r'(?<=^IL_)\d+(?=_trial_)'
        
        # Files we've already processed (absolute paths as strings)
        self._seen: set[str] = set()
        self._init_seen()

    # -----------------------------
    # Public API
    # -----------------------------
    def load_initial_traces(self, max_step: Optional[int] = None) -> List[Tuple[list, list, list, list]]:
        """
        Load up to `max_step` existing jsonl traces and mark them as seen.
        If both method arg and instance property are provided, the method arg wins.
        """
        limit = self._resolve_limit(max_step)
        units = self._list_units()
        # Only consider those not yet seen
        candidates = [p for p in units if str(p.resolve()) not in self._seen]
        to_load = candidates[:limit] if limit is not None else candidates

        # Mark only loaded ones as seen
        for p in to_load:
            self._seen.add(str(p.resolve()))

        return self._load_units(to_load)

    def poll_new_paths(self, max_step: Optional[int] = None) -> List[Path]:
        """
        Detect up to `max_step` newly created '*.jsonl' files under root (recursively)
        since last call. Mark ONLY those as seen and return them.
        """
        limit = self._resolve_limit(max_step)
        current = self._list_units()

        # Find unseen
        new_candidates: List[Path] = []
        for p in current:
            p_str = str(p.resolve())
            if p_str not in self._seen:
                new_candidates.append(p)

        to_take = new_candidates[:limit] if limit is not None else new_candidates

        # Mark only taken ones as seen
        for p in to_take:
            self._seen.add(str(p.resolve()))
        return to_take

    def poll_new_traces(self, max_step: Optional[int] = None) -> List[Tuple[list, list, list, list]]:
        """
        Detect and load up to `max_step` newly appeared jsonl traces.
        """
        new_units = self.poll_new_paths(max_step=max_step)
        if not new_units:
            return []
        return self._load_units(new_units)

    # -----------------------------
    # Introspection helpers (optional)
    # -----------------------------
    def remaining_unseen_count(self) -> int:
        """How many '*.jsonl' files under root remain unseen."""
        return sum(1 for p in self._list_units() if str(p.resolve()) not in self._seen)

    # -----------------------------
    # Internals
    # -----------------------------
    def _init_seen(self) -> None:
        """
        Do NOT pre-mark existing files as seen; we honor max_step by only marking
        files when we actually process them.
        """
        self._seen = set()

    def _list_units(self) -> List[Path]:
        """
        Return all '*.jsonl' files under root, recursively, with stable order.
        """
        if not self.root.exists():
            return []
        # rglob covers root and all subdirectories; ordering ensures deterministic batches
        units = sorted(self.root.rglob("*.jsonl"), key=lambda p: (p.parent.as_posix(), p.name))
        return units

    def _load_units(self, paths: Iterable[Path]) -> List[Tuple[list, list, list, list]]:
        merged: List[Tuple[list, list, list, list]] = []
        interference_vals = [ re.search(path, self.id_regex).group(1) for path in paths ]
        for tp in paths:
            s, a, r, net = trace_collec(
                str(tp),
                state_descriptor=self.control_config.get("state_cfg", None),
                reward_descriptor=self.control_config.get("reward_cfg", None),
            )
            if self._state_tf is not None:
                s = [self._state_tf.apply_to_list(flatten_leaves(x)) for x in s]
            else:
                s = [flatten_leaves(x) for x in s]
            a = [flatten_leaves(x) for x in a]
            r = [flatten_leaves(x) for x in r]
            merged.append((s, a, r, net))
        return merged, interference_vals

    def _resolve_limit(self, local_limit: Optional[int]) -> Optional[int]:
        """Choose per-call limit if provided, else instance default."""
        limit = local_limit if local_limit is not None else self.max_step
        if limit is None:
            return None
        # Normalize to a non-negative int (0 means process nothing)
        return max(0, int(limit))
