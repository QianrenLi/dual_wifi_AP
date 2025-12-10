#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot res.action vectors with optional smoothing.

Usage:
  python plot_actions.py --root /path/to/root --recent 3 --stride 5 --smooth 20
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np

# Import shared plotting utilities
from exp_trace.plot_config import PlotTheme
from exp_trace.plot_utils import (
    list_trials, moving_average, create_figure, save_figure,
    apply_scientific_style
)

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


def plot_actions(root: str | Path, recent: int = 0, stride: Optional[int] = None,
                 smooth: int = 1, action_selection: Optional[int] = None,
                 ax=None, labels: Optional[List[str]] = None,
                 colors: Optional[List[str]] = None):
    """
    Plot action vectors with optional smoothing.

    Parameters
    ----------
    root : str | Path
        Directory containing trial_* folders
    recent : int, optional
        Number of most recent trials to include (default: 0 = all)
    stride : int, optional
        Downsampling stride (default: None)
    smooth : int, optional
        Moving average window size (default: 1 = no smoothing)
    action_selection : int, optional
        Specific action dimension to plot (default: None = all)
    ax : matplotlib.axes.Axes, optional
        Axis to plot on (default: None = create new)
    labels : list, optional
        Labels for each action dimension (default: auto-generated)
    colors : list, optional
        Colors for each action dimension (default: from theme)

    Returns
    -------
    matplotlib.axes.Axes
    """
    root = Path(root)
    trials = list_trials(root, recent, pattern="IL_")
    acts = _read_actions(trials)

    if not acts:
        if ax is None:
            _, ax = create_figure(size="medium")
        ax.set_title("No actions found", fontsize=PlotTheme.FONT_SIZE_XLARGE)
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
        smoothed_series = [moving_average(raw_series[i], smooth) for i in range(D)]

    if ax is None:
        _, ax = create_figure(size="wide")

    def _label_for(i: int) -> str:
        return labels[i] if labels and i < len(labels) else f"a{i}"

    # Use theme colors if not provided
    if colors is None:
        colors = [PlotTheme.get_color(i) for i in range(D)]

    # Plot helper for one dimension
    def plot_dim(i: int):
        base_label = _label_for(i)
        color = colors[i % len(colors)]

        if smoothed_series is None:
            # No smoothing: plot only raw
            ax.plot(xs, raw_series[i], label=base_label,
                   color=color, linewidth=2)
            return

        # With smoothing: plot raw in background, smoothed on top
        ax.plot(xs, raw_series[i], label=f"{base_label} (raw)",
               color=color, linewidth=1, alpha=0.3, zorder=1)
        ax.plot(xs, smoothed_series[i],
               label=f"{base_label} (smoothed w={smooth})",
               color=color, linewidth=2.5, zorder=2)

    if action_selection is not None:
        if 0 <= action_selection < D:
            plot_dim(action_selection)
        else:
            ax.set_title(f"Selected action index {action_selection} out of range (D={D})")
    else:
        for i in range(D):
            plot_dim(i)

    # Apply scientific styling
    apply_scientific_style(
        ax,
        xlabel="global step",
        ylabel="action value",
        title=f"Actions over time • {root.name} • recent={recent or 'all'}"
    )

    ax.legend(ncol=2 if smoothed_series is not None else 1, fontsize=10)
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
    fig, ax = create_figure(size="wide")
    plot_actions(
        root=args.root,
        recent=args.recent,
        stride=args.stride,
        smooth=args.smooth,
        action_selection=args.selected,
        ax=ax
    )
    save_figure(fig, "action_plot.png")
    print("Saved to action_plot.png")
    # plt.show()

if __name__ == "__main__":
    main()
