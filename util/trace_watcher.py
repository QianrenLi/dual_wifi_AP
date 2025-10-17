# util/trace_watcher.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Union, Optional

from util.trace_collec import trace_collec, flatten_leaves
from net_util.state_transfom import _StateTransform

class TraceWatcher:
    """
    Watch a single root directory and treat every '*.jsonl' file under it
    (recursively) as a trace unit.

    API:
      - load_initial_traces() -> List[(s, a, r, net)]
        Loads all existing jsonl traces at construction time and marks them as seen.

      - poll_new_paths() -> List[Path]
        Returns newly appeared jsonl files since last check (and marks them seen).

      - poll_new_traces() -> List[(s, a, r, net)]
        Convenience: load newly appeared jsonl files and return merged traces.
    """

    def __init__(self, root: Union[str, Path], control_config: dict) -> None:
        self.root: Path = Path(root).resolve()
        self.control_config = control_config
        self._state_tf: Optional[_StateTransform] = None
        if control_config['state_transform_dict']:
            try:
                self._state_tf = _StateTransform.from_obj(control_config['state_transform_dict'])
            except Exception as e:
                # Fall back silently (or log if you prefer)
                print(f"[PolicyBase] Failed to load state transform: {e}")
                self._state_tf = None
                
        self._seen: set[str] = set()
        self._init_seen()

    # -----------------------------
    # Public API
    # -----------------------------
    def load_initial_traces(self) -> List[Tuple[list, list, list, list]]:
        """
        Load all existing jsonl traces under root and mark them as seen.
        """
        units = self._list_units()
        self._seen = {str(p.resolve()) for p in units}
        return self._load_units(units)

    def poll_new_paths(self) -> List[Path]:
        """
        Detect newly created '*.jsonl' files under root (recursively) since last call.
        """
        current = self._list_units()
        new_units: List[Path] = []
        for p in current:
            p_str = str(p.resolve())
            if p_str not in self._seen:
                new_units.append(p)
                self._seen.add(p_str)
        return new_units

    def poll_new_traces(self) -> List[Tuple[list, list, list, list]]:
        """
        Detect and load newly appeared jsonl traces.
        """
        new_units = self.poll_new_paths()
        if not new_units:
            return []
        return self._load_units(new_units)

    # -----------------------------
    # Internals
    # -----------------------------
    def _init_seen(self) -> None:
        if not self.root.exists():
            self._seen = set()
            return
        self._seen = {str(p.resolve()) for p in self._list_units()}

    def _list_units(self) -> List[Path]:
        """
        Return all '*.jsonl' files under root, recursively.
        """
        if not self.root.exists():
            return []
        # rglob covers root and all subdirectories
        units = sorted(self.root.rglob("*.jsonl"), key=lambda p: (p.parent.as_posix(), p.name))
        return units

    def _load_units(self, paths: Iterable[Path]) -> List[Tuple[list, list, list, list]]:
        merged: List[Tuple[list, list, list, list]] = []
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
        return merged
