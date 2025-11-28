from util.trace_collec import trace_filter, flatten_leaves  # your project utilities
from exp_trace.util import ExpTraceReader, Rollout, FolderRollout, FolderChannel, ChannelLog
from typing import Any, Dict, Iterable, Callable, Union, List, Sequence, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


Agg = Union[str, Callable[[Iterable[float]], float]]  # "sum" | "mean" | custom reducer

import matplotlib
plt.rcParams.update({
    # "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def _flatten_dict_of_lists(d: Dict[str, List[float]]) -> np.ndarray:
    if not d:
        return np.array([], dtype=float)
    parts = []
    for v in d.values():
        a = np.asarray(v, dtype=float)
        parts.append(a)
    return np.concatenate(parts) if parts else np.array([], dtype=float)

def summarize_rollouts_for_bars(rollouts: List["Rollout"]):
    thr_vals = []
    for r in rollouts:
        thr_arr = _flatten_dict_of_lists(r.throughput)  # dict[link]->list[float]
        thr_vals.append(thr_arr)
    return thr_vals

# ------------------------------- Plotting ------------------------------- #
def thru_plot(il_ids, thr_vals, out_dir):
    # ------------------- Plot thr_vals for each interference -------------------
    # ------------------- Subplots: one line per interference -------------------
    # Filter out empty throughput arrays first
    plot_items = [
        (il, thr)
        for il, thr in zip(il_ids, thr_vals)
        if isinstance(thr, np.ndarray) and thr.size > 0
    ]

    if not plot_items:
        print("[warn] No throughput data to plot.")
    else:
        n_plots = len(plot_items)

        # Make each subplot the same size and stack them vertically
        base_height = 2.5  # height per subplot (inches)
        fig_width = 10
        fig_height = base_height * n_plots

        fig, axes = plt.subplots(
            nrows=n_plots,
            ncols=1,
            sharex=True,
            figsize=(fig_width, fig_height),
            dpi=120,
        )

        # If only one subplot, axes is not a list
        if n_plots == 1:
            axes = [axes]

        for ax, (il, thr) in zip(axes, plot_items):
            xs = np.arange(len(thr))
            ax.plot(xs, thr, linewidth=1.5)
            ax.set_ylabel(f"IL {il}")
            ax.grid(True, linestyle="--", alpha=0.4)

        # Bottom subplot gets the x label
        axes[-1].set_xlabel("Sample Index")

        # Overall title; keep subplots "flat" by minimal spacing
        fig.suptitle("Throughput per interference", y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        out_path = out_dir / "thr_vals_by_interference_subplots.png"
        fig.savefig(out_path)
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