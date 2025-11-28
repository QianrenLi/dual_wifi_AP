#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot res.action vectors with optional smoothing.

Usage:
  python plot_actions.py --root /path/to/root --recent 3 --stride 5 --smooth 20
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

_TS_RE = re.compile(r"^trial_(\d{8}-\d{6})$")

def _trial_sort_key(p: Path) -> Tuple[int, str]:
    m = _TS_RE.match(p.name)
    if m:
        try:
            ts = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
            return (int(ts.timestamp()), p.name)
        except:
            pass
    return (int(p.stat().st_mtime), p.name)

def _list_trials(root: Path, recent: int) -> List[Path]:
    trials = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("IL_")]
    trials.sort(key=_trial_sort_key, reverse=True)
    if recent and recent > 0:
        trials = trials[:recent]
    trials.reverse()
    return trials

def _read_actions(trials: List[Path]) -> List[List[float]]:
    acts: List[List[float]] = []
    for t in trials:
        f = t / "rollout.jsonl"
        if not f.exists():
            continue
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except:
                    continue
                action = (((rec.get("res") or {}).get("action")) if "res" in rec else None)
                if isinstance(action, list) and all(isinstance(x, (int, float)) for x in action):
                    acts.append([float(x) for x in action])
    return acts

def _smooth(y: List[float], window: int) -> List[float]:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same").tolist()

def plot_actions(root: str | Path, recent: int = 0, stride: Optional[int] = None,
                 smooth: int = 1, action_selection: Optional[int] = None,
                 ax=None, labels: Optional[List[str]] = None):
    root = Path(root)
    trials = _list_trials(root, recent)
    acts = _read_actions(trials)
    if not acts:
        if ax is None:
            _, ax = plt.subplots()
        ax.set_title("No actions found")
        ax.set_xlabel("step")
        ax.set_ylabel("action value")
        return ax

    D = len(acts[0])
    raw_series: List[List[float]] = [[] for _ in range(D)]
    for vec in acts:
        if len(vec) == D:
            for i in range(D):
                raw_series[i].append(vec[i])

    xs = list(range(1, len(raw_series[0]) + 1))
    if stride and stride > 1:
        xs = xs[::stride]
        for i in range(D):
            raw_series[i] = raw_series[i][::stride]

    # Prepare smoothed copy (keeps raw for background plotting)
    smoothed_series: Optional[List[List[float]]] = None
    if smooth > 1:
        smoothed_series = [ _smooth(raw_series[i], smooth) for i in range(D) ]

    if ax is None:
        _, ax = plt.subplots()

    def _label_for(i: int) -> str:
        return labels[i] if labels and i < len(labels) else f"a{i}"

    # Styling
    lw_raw = 1.0
    lw_smooth = 2.0
    alpha_raw = 0.10
    z_raw = 1
    z_smooth = 2

    # Plot helper for one dimension
    def plot_dim(i: int):
        base_label = _label_for(i)
        if smoothed_series is None:
            # No smoothing: plot only raw
            ax.plot(xs, raw_series[i], label=base_label)
            return

        # With smoothing: plot raw in background, smoothed on top using same color
        # Grab a color from the cycler once per dimension
        color = ax._get_lines.get_next_color()
        # Raw background
        ax.plot(xs, raw_series[i], label=f"{base_label} (raw)", linewidth=lw_raw,
                alpha=alpha_raw, zorder=z_raw, color=color)
        # Smoothed foreground
        ax.plot(xs, smoothed_series[i], label=f"{base_label} (smoothed w={smooth})",
                linewidth=lw_smooth, zorder=z_smooth, color=color)

    if action_selection is not None:
        if 0 <= action_selection < D:
            plot_dim(action_selection)
        else:
            ax.set_title(f"Selected action index {action_selection} out of range (D={D})")
    else:
        for i in range(D):
            plot_dim(i)

    ax.set_xlabel("global step")
    ax.set_ylabel("action value")
    ax.set_title(f"res.action over time • root={root.name} • recent={recent or 'all'}")
    ax.legend(ncol=2 if smoothed_series is not None else 1, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    return ax

# -------- CLI --------

def _parse_args():
    p = argparse.ArgumentParser(description="Plot res.action from rollout.jsonl with smoothing")
    p.add_argument("--root", required=True, type=str, help="Directory containing trial_*")
    p.add_argument("--recent", type=int, default=0, help="Only include N most recent trials (0 = all)")
    p.add_argument("--stride", type=int, default=None, help="Downsample stride (e.g., 5)")
    p.add_argument("--smooth", type=int, default=1, help="Moving average window size (e.g., 10). 1 = no smoothing")
    p.add_argument("--selected", type=int, default=None, help="Action selection")
    return p.parse_args()

def main():
    args = _parse_args()
    plot_actions(
        root=args.root,
        recent=args.recent,
        stride=args.stride,
        smooth=args.smooth,
        action_selection=args.selected
    )
    plt.tight_layout()
    # plt.show()
    plt.savefig("action_plot.png")
    print("Saved to exp_trace/action_plot.png")

if __name__ == "__main__":
    main()
