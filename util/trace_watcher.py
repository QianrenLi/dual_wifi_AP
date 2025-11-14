# util/trace_watcher.py
from __future__ import annotations

import re
import random
from pathlib import Path
from typing import Iterable, List, Tuple, Union, Optional, Sequence

from util.trace_collec import trace_collec, flatten_leaves
from net_util.state_transfom import _StateTransform


PathLike = Union[str, Path]
MultiPathLike = Union[PathLike, Iterable[PathLike]]


class TraceWatcher:
    """
    Watch one or multiple root directories and treat every '*.jsonl' file under them
    (recursively) as a trace unit.

    max_step:
      Maximum number of trace-units to process per call. If None, process all.

    API:
      - load_initial_traces() -> Tuple[List[(s, a, r, net)], List[int]]
        Loads up to `max_step` existing jsonl traces (or unlimited if None) and
        marks ONLY those as seen. Returns (merged_traces, interference_vals).

      - poll_new_paths() -> List[Path]
        Returns up to `max_step` newly appeared jsonl files since last check
        (and marks ONLY those as seen).

      - poll_new_traces() -> Tuple[List[(s, a, r, net)], List[int]]
        Convenience: load newly appeared jsonl files (up to `max_step`) and return merged traces
        plus their inferred interference values.
    """

    def __init__(self, root: MultiPathLike, control_config: dict, max_step: Optional[int] = None) -> None:
        # --- allow single or multiple roots ---
        self.roots: List[Path] = self._normalize_roots(root)

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

        # Example: IL_1_trial_20251023-190512 -> capture "1"
        self.id_regex = r'(?<=^IL_)\d+(?=_trial_)'

        # Files we've already processed (absolute paths as strings)
        self._seen: set[str] = set()
        self._init_seen()

    # -----------------------------
    # Public API
    # -----------------------------
    def load_initial_traces(self, max_step: Optional[int] = None) -> Tuple[List[Tuple[list, list, list, list]], List[int]]:
        """
        Load up to `max_step` existing jsonl traces across all roots and mark them as seen.
        If both method arg and instance property are provided, the method arg wins.

        Returns:
          (merged_traces, interference_vals)
        """
        limit = self._resolve_limit(max_step)
        units = self._list_units()  # across all roots
        candidates = [p for p in units if str(p.resolve()) not in self._seen]
        random.shuffle(candidates)
        to_load = candidates[:limit] if limit is not None else candidates

        for p in to_load:
            self._seen.add(str(p.resolve()))
        return self._load_units(to_load)

    def poll_new_paths(self, max_step: Optional[int] = None) -> List[Path]:
        """
        Detect up to `max_step` newly created '*.jsonl' files under all roots (recursively)
        since last call. Mark ONLY those as seen and return them.
        """
        limit = self._resolve_limit(max_step)
        current = self._list_units()

        new_candidates: List[Path] = []
        for p in current:
            p_str = str(p.resolve())
            if p_str not in self._seen:
                new_candidates.append(p)

        random.shuffle(new_candidates)
        to_take = new_candidates[:limit] if limit is not None else new_candidates

        for p in to_take:
            self._seen.add(str(p.resolve()))
        return to_take

    def poll_new_traces(self, max_step: Optional[int] = None) -> Tuple[List[Tuple[list, list, list, list]], List[int]]:
        """
        Detect and load up to `max_step` newly appeared jsonl traces across all roots.

        Returns:
          (merged_traces, interference_vals)

        If no new units: returns ([], [])
        """
        new_units = self.poll_new_paths(max_step=max_step)
        if not new_units:
            return [], []
        return self._load_units(new_units)

    # -----------------------------
    # Introspection helpers (optional)
    # -----------------------------
    def remaining_unseen_count(self) -> int:
        """How many '*.jsonl' files under all roots remain unseen."""
        return sum(1 for p in self._list_units() if str(p.resolve()) not in self._seen)

    # -----------------------------
    # Internals
    # -----------------------------
    def _normalize_roots(self, root: MultiPathLike) -> List[Path]:
        """Normalize single-or-multiple root input to a unique, resolved list."""
        if isinstance(root, (str, Path)):
            roots_seq: Sequence[PathLike] = [root]
        else:
            roots_seq = list(root)

        roots: List[Path] = []
        seen = set()
        for r in roots_seq:
            p = Path(r).resolve()
            # skip duplicates; keep deterministic order
            if p.as_posix() in seen:
                continue
            seen.add(p.as_posix())
            roots.append(p)

        # it's fine if some roots don't exist yet; we simply list none from them
        return roots

    def _init_seen(self) -> None:
        """
        Do NOT pre-mark existing files as seen; we honor max_step by only marking
        files when we actually process them.
        """
        self._seen = set()

    def _list_units(self) -> List[Path]:
        """
        Return all '*.jsonl' files under all roots, recursively, with stable order
        (first by parent path, then by file name). Non-existing roots are ignored.
        """
        units: List[Path] = []
        for root in self.roots:
            if not root.exists():
                continue
            units.extend(root.rglob("*.jsonl"))
        # stable, deterministic ordering across possibly many roots
        units = sorted(units, key=lambda p: (p.parent.as_posix(), p.name))
        return units

    def _load_units(self, paths: Iterable[Path]) -> Tuple[List[Tuple[list, list, list, list]], List[int]]:
        merged: List[Tuple[list, list, list, list]] = []
        interference_vals: List[int] = []

        # infer interference id from each path's parent dir name (same as before)
        for path in paths:
            res = re.search(self.id_regex, path.parent.stem)
            if res is None:
                interference_vals.append(0)
            else:
                interference_vals.append(int(res.group(0)))

        for tp in paths:
            s, a, r, net = trace_collec(
                str(tp),
                state_descriptor=self.control_config.get("state_cfg", None),
                reward_descriptor=self.control_config.get("reward_cfg", None),
            )
            s_nor = []
            if self._state_tf:
                for _s in s:
                    s_nor.append(self._state_tf.apply_to_list(_s))
                s = s_nor
            merged.append((s, a, r, net))
        return merged, interference_vals

    def _resolve_limit(self, local_limit: Optional[int]) -> Optional[int]:
        """Choose per-call limit if provided, else instance default."""
        limit = local_limit if local_limit is not None else self.max_step
        if limit is None:
            return None
        return max(0, int(limit))
