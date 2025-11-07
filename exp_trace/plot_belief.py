#!/usr/bin/env python3
from typing import Any, Dict, Iterable, Optional, Callable, Union, List, Tuple
from pathlib import Path
import json
import re
import numpy as np
import matplotlib.pyplot as plt

# If you use these elsewhere, keep them. Not needed for the box figure itself.
Agg = Union[str, Callable[[Iterable[float]], float]]  # "sum" | "mean" | custom reducer

# ------------------------ Folder / naming utils ------------------------ #
_IL_PATTERN = re.compile(r"^IL_(\d+)_trial_(.+)$")

def extract_il_and_trial(name: str) -> Tuple[Optional[int], Optional[str]]:
    m = _IL_PATTERN.match(name)
    if not m:
        return None, None
    try:
        return int(m.group(1)), m.group(2)
    except Exception:
        return None, m.group(2)

def pick_latest_per_il(meta_folder: Path,
                       pattern: str = "IL_*_trial_*",
                       by_mtime: bool = True) -> Dict[int, Path]:
    """
    Scan `meta_folder` for IL_*_trial_* subfolders, group by IL id,
    and pick the latest (by mtime or lexicographic name) per IL.
    Returns { il_id: latest_run_folder_path }.
    """
    if not meta_folder.is_dir():
        return {}
    runs = [p for p in meta_folder.glob(pattern) if p.is_dir()]
    groups: Dict[int, List[Path]] = {}
    for p in runs:
        il_id, _ = extract_il_and_trial(p.name)
        if il_id is None:
            continue
        groups.setdefault(il_id, []).append(p)

    latest: Dict[int, Path] = {}
    for il_id, folders in groups.items():
        if not folders:
            continue
        if by_mtime:
            folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        else:
            folders.sort(key=lambda x: x.name, reverse=True)
        latest[il_id] = folders[0]
    return latest

def find_rollout_file(run_folder: Path) -> Optional[Path]:
    for name in ("rollout.jsonl", "roolout.jsonl"):
        p = run_folder / name
        if p.exists():
            return p
    return None

# ----------------------------- Belief loading ----------------------------- #
def load_beliefs_from_rollout(rollout_path: Path) -> List[float]:
    """
    Parse rollout.jsonl and collect a scalar belief per line.
    Accepts scalar, [scalar], or vector-like (take first component).
    """
    beliefs: List[float] = []
    with rollout_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            bel = rec.get("res", {}).get("belief", None)
            if bel is None:
                continue

            if isinstance(bel, (int, float)):
                beliefs.append(float(bel))
            elif isinstance(bel, list) and len(bel) > 0:
                # try first element; fallback to flatten
                try:
                    beliefs.append(float(bel[0]))
                except Exception:
                    try:
                        flat = np.array(bel, dtype=float).ravel()
                        if flat.size > 0:
                            beliefs.append(float(flat[0]))
                    except Exception:
                        continue
            else:
                try:
                    arr = np.array(bel, dtype=float).ravel()
                    if arr.size > 0:
                        beliefs.append(float(arr[0]))
                except Exception:
                    continue
    return beliefs

# ------------------------------- Plotting ------------------------------- #
def plot_belief_box_per_il(il_to_beliefs: Dict[int, List[float]],
                           title: str,
                           out_path: Path,
                           showfliers: bool = False):
    """
    Make ONE box figure: each box is the belief distribution for one IL (latest run).
    X tick labels are IL indices.
    """
    if not il_to_beliefs:
        print("[info] Nothing to plot.")
        return

    ils = sorted(il_to_beliefs.keys())
    data = [il_to_beliefs[i] if il_to_beliefs[i] else [0.0] for i in ils]

    fig, ax = plt.subplots()
    ax.boxplot(data, tick_labels=[str(i) for i in ils], showfliers=showfliers)
    ax.set_xlabel("IL index")
    ax.set_ylabel("Belief")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] Saved: {out_path}")

# ----------------------------------- CLI ----------------------------------- #
def main():
    """
    For each --meta-folder:
      - pick the latest run per IL
      - load belief sequences
    Merge across all provided meta-folders by IL id, then
    plot ONE box figure where each box corresponds to one IL index.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Plot belief box of latest run per IL (x-label = IL index).")
    ap.add_argument("--meta-folder", nargs="+", required=True,
                    help="One or more meta-folders to scan.")
    ap.add_argument("--pattern", default="IL_*_trial_*",
                    help="Glob pattern for run folders (default: IL_*_trial_*)")
    ap.add_argument("--out-dir", default="plots",
                    help="Directory to save the figure (default: plots)")
    ap.add_argument("--filename", default="belief_box_latest_per_IL.png",
                    help="Output filename for the box plot")
    ap.add_argument("--by-name", action="store_true",
                    help="Pick latest by lexicographic name instead of modification time")
    ap.add_argument("--showfliers", action="store_true",
                    help="Show outliers in the box plot")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    by_mtime = not args.by_name

    # Accumulate beliefs by IL across all meta-folders
    il_to_beliefs: Dict[int, List[float]] = {}

    for meta in args.meta_folder:
        meta_path = Path(meta)
        if not meta_path.is_dir():
            print(f"[warn] meta-folder not found or not a directory: {meta_path}")
            continue

        latest_map = pick_latest_per_il(meta_path, pattern=args.pattern, by_mtime=by_mtime)
        if not latest_map:
            print(f"[warn] no IL runs matched under: {meta_path}")
            continue

        for il_id, run_folder in latest_map.items():
            rollout = find_rollout_file(run_folder)
            if rollout is None:
                print(f"[warn] no rollout.jsonl in {run_folder}")
                continue
            beliefs = load_beliefs_from_rollout(rollout)
            if not beliefs:
                print(f"[warn] parsed 0 belief entries from {rollout}")
                continue
            il_to_beliefs.setdefault(il_id, []).extend(beliefs)

    # Plot one box figure: x = IL index
    if il_to_beliefs:
        out_path = out_dir / args.filename
        title = "Belief distribution — latest run per IL"
        plot_belief_box_per_il(il_to_beliefs, title=title, out_path=out_path, showfliers=args.showfliers)
    else:
        print("[info] No beliefs found; nothing to plot.")

if __name__ == "__main__":
    main()
