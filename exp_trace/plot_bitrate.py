from util.trace_collec import trace_filter, flatten_leaves  # your project utilities
from exp_trace.util import ExpTraceReader, Rollout, FolderRollout, FolderChannel, ChannelLog
from typing import Any, Dict, Iterable, Callable, Union, List, Sequence, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from exp_trace.plot_config import PlotTheme
from exp_trace.plot_utils import (
    flatten_dict_of_lists, create_figure, save_figure,
    apply_scientific_style
)

Agg = Union[str, Callable[[Iterable[float]], float]]  # "sum" | "mean" | custom reducer

def summarize_rollouts_for_bars(rollouts: List["Rollout"]):
    thr_vals = []
    for r in rollouts:
        thr_arr = flatten_dict_of_lists(r.throughput)  # dict[link]->list[float]
        thr_vals.append(thr_arr)
    return thr_vals

# ------------------------------- Plotting ------------------------------- #
def thru_plot(il_ids, thr_vals, out_dir, figsize: str = "medium"):
    """
    Plot throughput values for each interference level as subplots.

    Parameters
    ----------
    il_ids : list
        Interference level IDs
    thr_vals : list
        Throughput value arrays for each IL
    out_dir : Path
        Output directory
    figsize : str, optional
        Figure size name (default: "medium")
    """
    # Filter out empty throughput arrays first
    plot_items = [
        (il, thr)
        for il, thr in zip(il_ids, thr_vals)
        if isinstance(thr, np.ndarray) and thr.size > 0
    ]

    if not plot_items:
        print("[warn] No throughput data to plot.")
        return

    n_plots = len(plot_items)

    # Create figure with subplots
    fig, axes = create_figure(
        size=figsize,
        nrows=n_plots,
        ncols=1,
        sharex=True
    )

    # If only one subplot, axes is not a list
    if n_plots == 1:
        axes = [axes]

    colors = [PlotTheme.get_color(i) for i in range(n_plots)]

    for i, (ax, (il, thr)) in enumerate(zip(axes, plot_items)):
        xs = np.arange(len(thr))
        ax.plot(xs, thr, color=colors[i], linewidth=1.5)
        ax.set_ylabel(f"IL {il}", fontsize=PlotTheme.FONT_SIZE_MEDIUM)
        apply_scientific_style(ax, minor_ticks=False)

    # Bottom subplot gets the x label
    axes[-1].set_xlabel("Sample Index", fontsize=PlotTheme.FONT_SIZE_MEDIUM)

    # Overall title; keep subplots "flat" by minimal spacing
    fig.suptitle("Throughput per interference level", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save figure
    out_path = out_dir / "thr_vals_by_interference_subplots.png"
    save_figure(fig, out_path, dpi=PlotTheme.DPI_PUBLICATION)
    print(f"[ok] Saved: {out_path}")



#!/usr/bin/env python3
def main():
    """
    Use argparse to:
      - read multiple experiment folders
      - compute channel stats from output.log via compute_channel_stats(...)
      - compute total reward from rollout.jsonl via compute_reward(...)
      - draw:
          * box plots for channel 0 and 1 (one box per folder)
          * bar plot for total rewards (one bar per folder)

    Example:
      python script.py runA runB --control-config control.json --agg sum --out-dir plots --showfliers
    """
    import argparse
    import json
    from pathlib import Path

    # ---- CLI ----
    p = argparse.ArgumentParser(description="Channel box plots and reward bar plot across folders.")
    p.add_argument("--meta-folder", required=True, help="One or more meta-folders to scan.")
    p.add_argument("--control-config", required=True, help="Path to control_config.json containing {'reward_cfg': ...}")
    p.add_argument("--agg", choices=["sum", "mean"], default="mean", help="Aggregation used in compute_reward (default: sum)")
    p.add_argument("--out-dir", default="plots", help="Directory to save generated plots (default: plots)")
    p.add_argument("--showfliers", action="store_true", help="Show outliers in box plots")
    args = p.parse_args()

    # ---- Load reward_cfg ----
    cfg_path = Path(args.control_config)
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            reward_cfg = json.load(f).get("reward_cfg")
    except Exception as e:
        print(f"[error] Failed to read control-config '{cfg_path}': {e}")
        return
    

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_reader = ExpTraceReader(args.meta_folder)

    rollouts = []
    il_ids = []
    seq_interface = []
    for il_id, run_folder in sorted(data_reader.latest_folders.items()):
        il_ids.append(il_id)
        
        rollout = Rollout(FolderRollout(run_folder), {})
        rollouts.append(rollout)
        
        try:
            channel = ChannelLog(FolderChannel(run_folder))
            seq_interface.append(channel)
        except:
            continue            

        
    thr_vals  = summarize_rollouts_for_bars(rollouts)
    thru_plot(il_ids, thr_vals, out_dir)
    


if __name__ == "__main__":
    main()