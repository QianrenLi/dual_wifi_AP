#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward Plot (Matplotlib)
- Reads reward from trial_*/rollout.jsonl under --root
- Uses control_config.json's "reward_cfg" to compute reward per record
- Two modes:
    * element: raw per-step reward
    * acc: discounted running sum with gamma
- Simple, readable, and easy to extend

Usage:
  python reward_plot.py --root /path/to/root --control-config control.json \
      --mode element --gamma 0.99 --recent 3

As a library:
  from reward_plot import plot_rewards
  ax = plot_rewards(root="...", control_config="...", mode="acc", gamma=0.97)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# --- Optional project utilities (fallbacks provided) ---
from util.trace_collec import trace_filter, flatten_leaves  # type: ignore

# Import shared plotting utilities
from exp_trace.plot_config import PlotTheme
from exp_trace.plot_utils import (
    list_trials, discounted_running_sum, downsample_series,
    create_figure, save_figure, apply_scientific_style
)

@dataclass
class Config:
    root: Path
    control_path: Path
    mode: str = "element"     # "element" or "acc"
    gamma: float = 0.99
    recent: int = 0           # 0 = all
    stride: Optional[int] = None  # optional downsample stride (e.g., 5, 10)

# ---------- helpers ----------

def read_reward_series(trials: List[Path], reward_cfg: Optional[Dict[str, Any]], agg: str) -> List[float]:
    """Aggregate reward per record from rollout.jsonl across trials (oldest->newest)."""
    series: List[float] = []
    for t in trials:
        f = t / "rollout.jsonl"
        if not f.exists():
            continue
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        continue
                except Exception:
                    continue
                if reward_cfg is None:
                    r = 0.0
                else:
                    filtered = trace_filter(rec, reward_cfg)
                    leaves = flatten_leaves(filtered)
                    if not leaves:
                        r = 0.0
                    elif agg == "mean":
                        r = float(sum(leaves) / max(1, len(leaves)))
                    else:
                        r = float(sum(leaves))
                series.append(r)
    return series

def load_reward_cfg(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("reward_cfg")
    except Exception:
        return None

# ---------- main plotting API ----------

def plot_rewards(
    root: str | Path,
    control_config: str | Path,
    mode: str = "element",
    gamma: float = 0.99,
    recent: int = 0,
    stride: Optional[int] = None,
    ax=None,
    label: str = "reward",
    color: Optional[str] = None,
):
    """
    Plot reward time series from trial_* rollout.jsonl using Matplotlib.

    Parameters
    ----------
    root : str | Path
        Directory containing trial_* folders.
    control_config : str | Path
        JSON file containing {"reward_cfg": ...}.
    mode : {"element","acc"}
        "element" = raw per-step reward; "acc" = discounted running sum.
    gamma : float
        Discount factor for "acc" mode.
    recent : int
        Only include N most recent trials (0 = all).
    stride : Optional[int]
        Optional stride-based downsampling for large series (e.g., 10).
    ax : matplotlib.axes.Axes or None
        Axis to plot into; creates a new figure/axis if None.
    label : str
        Series label for the legend.
    color : str, optional
        Color for the plot line (default: from theme)

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    root = Path(root)
    control_config = Path(control_config)
    reward_cfg = load_reward_cfg(control_config)

    trials = list_trials(root, recent, pattern="IL_")
    ys = read_reward_series(trials, reward_cfg, agg="sum")  # default agg=sum (same as your app)
    if mode == "acc":
        ys = discounted_running_sum(ys, gamma)

    xs = list(range(1, len(ys) + 1))
    if stride and stride > 1 and len(xs) > stride:
        xs, ys = downsample_series(xs, ys, stride)

    # Create figure if no axis provided
    if ax is None:
        fig, ax = create_figure(size="wide")

    # Use theme color if none provided
    if color is None:
        color = PlotTheme.get_color(0)

    # Plot with styling
    ax.plot(xs, ys, color=color, linewidth=2,
            label=f"{label} ({mode}{'' if mode=='element' else f', γ={gamma:.3f}'})")

    # Apply scientific styling
    apply_scientific_style(
        ax,
        xlabel="global step",
        ylabel="reward",
        title=f"Reward over time • {root.name} • recent={recent or 'all'}"
    )

    ax.legend()
    return ax

# ---------- CLI ----------

def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Plot reward from trial_* with Matplotlib")
    p.add_argument("--root", required=True, type=str, help="Directory containing trial_*")
    p.add_argument("--control-config", required=True, type=str, help="JSON with reward_cfg")
    p.add_argument("--mode", choices=["element", "acc"], default="element")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--recent", type=int, default=0)
    p.add_argument("--stride", type=int, default=None, help="Optional downsample stride (e.g., 10)")
    args = p.parse_args()
    return Config(
        root=Path(args.root),
        control_path=Path(args.control_config),
        mode=args.mode,
        gamma=args.gamma,
        recent=args.recent,
        stride=args.stride,
    )

def main():
    cfg = _parse_args()
    fig, ax = create_figure(size="wide")
    plot_rewards(
        root=cfg.root,
        control_config=cfg.control_path,
        mode=cfg.mode,
        gamma=cfg.gamma,
        recent=cfg.recent,
        stride=cfg.stride,
        ax=ax
    )
    save_figure(fig, "reward.png")
    # plt.show()

if __name__ == "__main__":
    main()
