"""
Plot reward lines from different meta_folders for comparison.

X-axis: IL (Interference Level) index
Y-axis: Average reward
Each folder becomes one line on the plot.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from exp_trace.util import ExpTraceReader, Rollout, FolderRollout
from exp_trace.plot_config import PlotTheme
from exp_trace.plot_utils import (
    create_figure,
    save_figure,
    apply_scientific_style,
    aggregate_values,
)
from exp_trace.plot_eval_res import aggregate_stats_per_il, summarize_single_rollout
from util.trace_collec import trace_filter, flatten_leaves


def compute_reward(record: Dict[str, Any], reward_cfg: Dict[str, Any], agg: str = "sum") -> float:
    """Compute reward from a record using the reward configuration."""
    if not isinstance(record, dict) or not reward_cfg:
        return 0.0
    try:
        filtered = trace_filter(record, reward_cfg)
        leaves = flatten_leaves(filtered)
    except Exception:
        return 0.0

    if not leaves:
        return 0.0

    if agg == "mean":
        return float(sum(leaves) / len(leaves))
    return float(sum(leaves))


def aggregate_rewards_per_il(il_to_rollouts: Dict[str, List[Rollout]], agg: str = "mean") -> Tuple[List[str], np.ndarray]:
    """
    Aggregate rewards across rollouts for each IL.
    Uses aggregate_stats_per_il from plot_eval_res.py.
    """
    il_ids, _, _, _, _, rew_mean, _ = aggregate_stats_per_il(il_to_rollouts, agg=agg)
    return il_ids, np.asarray(rew_mean)


def collect_rewards_from_folder(
    meta_folder: Path,
    reward_cfg: Dict[str, Any],
    agg: str = "mean",
    last_k: List[int] = [0],
) -> Tuple[List[str], np.ndarray]:
    """Collect aggregated rewards from a meta_folder, grouped by IL."""
    def reward_handler(rec: Dict[str, Any]) -> float:
        return compute_reward(rec, reward_cfg, agg='sum')

    data_reader = ExpTraceReader(meta_folder)
    il_to_rollouts: Dict[str, List[Rollout]] = {}
    seen_keys = set()

    for k in last_k:
        per_il = data_reader.pick_last_k_per_il(k)
        for il_id, run_folder in sorted(per_il.items()):
            key = (il_id, str(run_folder))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            rollout = Rollout(FolderRollout(run_folder), {'reward': reward_handler})
            il_to_rollouts.setdefault(il_id, []).append(rollout)

    if not il_to_rollouts:
        return [], np.array([])

    return aggregate_rewards_per_il(il_to_rollouts, agg=agg)


def plot_reward_lines(
    folder_il_data: Dict[str, Tuple[List[str], np.ndarray]],
    labels: List[str] = None,
    xlabel: str = "IL Index",
    ylabel: str = "Reward",
    title: str = None,
    save_path: str = None,
    figsize: str = "small",
    line_width: float = 2.5,
    marker_size: float = 8,
):
    """Plot reward lines from different folders, with IL index on x-axis."""
    fig, ax = create_figure(size=figsize)

    if labels is None:
        labels = list(folder_il_data.keys())

    # Get all unique IL IDs
    all_il_ids = set()
    for il_ids, _ in folder_il_data.values():
        all_il_ids.update(il_ids)
    common_il_ids = sorted(all_il_ids, key=lambda x: int(x))

    for idx, (folder_name, (il_ids, rew_mean)) in enumerate(folder_il_data.items()):
        il_to_idx = {il_id: i for i, il_id in enumerate(common_il_ids)}

        x_values = []
        y_values = []

        for il_id, mean_val in zip(il_ids, rew_mean):
            if il_id in il_to_idx and np.isfinite(mean_val):
                x_values.append(il_to_idx[il_id])
                y_values.append(mean_val)

        if not x_values:
            continue

        x_values = np.array(x_values)
        y_values = np.array(y_values)

        # Sort by x values
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = y_values[sort_idx]

        # Plot line with markers
        label = labels[idx] if idx < len(labels) else folder_name
        markers = ['o', 'v', 's', 'p', '^', 'D', '*', 'h', 'X', '+']
        ax.plot(
            x_values,
            y_values,
            color=PlotTheme.get_color(idx),
            linewidth=line_width,
            marker=markers[idx % len(markers)],
            markersize=marker_size,
            label=label,
        )

    # Set x-ticks to IL IDs
    ax.set_xticks(range(len(common_il_ids)))
    ax.set_xticklabels(common_il_ids)

    # Apply styling
    apply_scientific_style(ax, xlabel=xlabel, ylabel=ylabel, title=title, minor_ticks=True)

    # Add legend
    ax.legend(loc="best", framealpha=PlotTheme.LEGEND_FRAME_ALPHA, fontsize=PlotTheme.LEGEND_FONT_SIZE)

    if save_path:
        save_figure(fig, save_path, dpi=PlotTheme.DPI_PUBLICATION)
        print(f"Saved to {save_path}")


def main():
    p = argparse.ArgumentParser(description="Plot reward lines from different meta_folders.")
    p.add_argument("--meta-folder", nargs="+", required=True, help="Meta-folders to compare.")
    p.add_argument("--control-config", required=True, help="Path to control_config.json with reward_cfg")
    p.add_argument("--agg", choices=["sum", "mean"], default="mean", help="Aggregation method (default: mean)")
    p.add_argument("--labels", nargs="+", default=None, help="Labels for each folder")
    p.add_argument("--out-dir", default="plots", help="Output directory (default: plots)")
    p.add_argument("--save-name", default="reward_lines.pdf", help="Output file name (default: reward_lines.pdf)")
    p.add_argument("--last", nargs="+", type=int, default=[0], help="Indices of last runs per IL (default: 0)")
    p.add_argument("--xlabel", default="IL Index", help="X-axis label")
    p.add_argument("--ylabel", default="Reward", help="Y-axis label")
    p.add_argument("--title", default=None, help="Plot title")
    p.add_argument("--figsize", default="medium", help="Figure size")
    args = p.parse_args()

    # Load reward configuration
    cfg_path = Path(args.control_config)
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            reward_cfg = json.load(f).get("reward_cfg")
    except Exception as e:
        print(f"[error] Failed to read control-config '{cfg_path}': {e}")
        return

    if not reward_cfg:
        print("[error] No reward_cfg found in control config")
        return

    # Collect data from all folders
    folder_il_data: Dict[str, Tuple[List[str], np.ndarray]] = {}

    for folder_path in args.meta_folder:
        meta_path = Path(folder_path)
        if not meta_path.is_dir():
            print(f"[warn] Skipping non-existent folder: {folder_path}")
            continue

        print(f"Collecting from {folder_path}...")
        il_ids, rew_mean = collect_rewards_from_folder(meta_path, reward_cfg, agg=args.agg, last_k=args.last)
        print(f"  ILs: {il_ids}")
        print(f"  Rewards: {rew_mean}")

        folder_il_data[meta_path.name] = (il_ids, rew_mean)

    if not folder_il_data:
        print("[error] No data collected")
        return

    # Create output directory and plot
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    plot_reward_lines(
        folder_il_data,
        labels=args.labels,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title,
        save_path=str(Path(args.out_dir) / args.save_name),
        figsize=args.figsize,
    )


if __name__ == "__main__":
    main()
