"""
Plot reward lines from different meta_folders for comparison.

This script reads experiment traces from multiple meta_folders and plots
reward curves for comparison across different experiments or configurations.

X-axis: IL (Interference Level) index
Y-axis: Average reward
Each folder becomes one line on the plot.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from exp_trace.util import ExpTraceReader, Rollout, FolderRollout
from exp_trace.plot_config import PlotTheme
from exp_trace.plot_utils import (
    create_figure,
    save_figure,
    apply_scientific_style,
    flatten_dict_of_lists,
    aggregate_values,
)

from util.trace_collec import trace_filter, flatten_leaves


def compute_reward(
    record: Dict[str, Any],
    reward_cfg: Dict[str, Any],
    agg: str = "mean",
) -> float:
    """
    Compute reward from a record using the reward configuration.

    Parameters
    ----------
    record : dict
        Single record from rollout
    reward_cfg : dict
        Reward configuration from control_config.json
    agg : str, optional
        Aggregation method ("sum" or "mean", default: "mean")

    Returns
    -------
    float
        Computed reward value
    """
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


def aggregate_rewards_per_il(
    il_to_rollouts: Dict[str, List[Rollout]],
    agg: str = "mean",
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Aggregate rewards across rollouts for each IL.

    Parameters
    ----------
    il_to_rollouts : dict
        Dictionary mapping IL IDs to lists of rollouts
    agg : str, optional
        Aggregation method (default: "mean")

    Returns
    -------
    tuple
        (il_ids, reward_mean, reward_std)
    """
    il_ids = sorted(il_to_rollouts.keys(), key=lambda x: int(x))
    rew_mean, rew_std = [], []

    for il in il_ids:
        rew_list = []
        for r in il_to_rollouts[il]:
            rewards = np.array(r.optional_data.get('reward', []))
            if rewards.size > 0:
                rew_list.extend(rewards)

        rew_arr = np.asarray(rew_list, dtype=float)

        # Use aggregate_values utility
        rew_mean.append(aggregate_values(rew_arr, agg) if rew_arr.size > 0 else float('nan'))

        # Calculate standard deviation
        if rew_arr.size > 1:
            rew_std.append(aggregate_values(rew_arr, "std"))
        else:
            rew_std.append(0.0)

    return il_ids, np.asarray(rew_mean), np.asarray(rew_std)


def collect_rewards_from_folder(
    meta_folder: Path,
    reward_cfg: Dict[str, Any],
    agg: str = "mean",
    last_k: List[int] = [0],
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Collect aggregated rewards from a meta_folder, grouped by IL.

    Parameters
    ----------
    meta_folder : Path
        Path to the meta_folder
    reward_cfg : dict
        Reward configuration
    agg : str, optional
        Aggregation method (default: "mean")
    last_k : list of int, optional
        Which last runs to include (default: [0] for latest)

    Returns
    -------
    tuple
        (il_ids, reward_mean, reward_std)
    """
    def reward_handler(rec: Dict[str, Any]) -> float:
        return compute_reward(rec, reward_cfg, agg=agg)

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

            try:
                rollout = Rollout(FolderRollout(run_folder), {'reward': reward_handler})
                il_to_rollouts.setdefault(il_id, []).append(rollout)
            except Exception as e:
                print(f"[warn] Failed to load rollout for {run_folder}: {e}")
                continue

    if not il_to_rollouts:
        return [], np.array([]), np.array([])

    return aggregate_rewards_per_il(il_to_rollouts, agg=agg)


def get_common_il_ids(folder_il_data: Dict[str, Tuple[List[int], np.ndarray, np.ndarray]]) -> List[int]:
    """
    Get the common IL IDs across all folders.

    Parameters
    ----------
    folder_il_data : dict
        Dictionary mapping folder names to (il_ids, reward_mean, reward_std) tuples

    Returns
    -------
    list
        Sorted list of common IL IDs
    """
    all_il_sets = [set(il_ids) for il_ids, _, _ in folder_il_data.values()]
    common_il_ids = set.intersection(*all_il_sets) if all_il_sets else set()
    return sorted(common_il_ids, key=lambda x: int(x))


def plot_reward_lines(
    folder_il_data: Dict[str, Tuple[List[int], np.ndarray, np.ndarray]],
    labels: Optional[List[str]] = None,
    xlabel: str = "IL Index",
    ylabel: str = "Reward",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: str = "medium",
    line_width: float = 2.5,
    marker_size: float = 8,
    show_error_bars: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Plot reward lines from different folders, with IL index on x-axis.

    Parameters
    ----------
    folder_il_data : dict
        Dictionary mapping folder names to (il_ids, reward_mean, reward_std) tuples
    labels : list, optional
        Labels for each folder (default: uses folder names)
    xlabel : str, optional
        X-axis label (default: "IL Index")
    ylabel : str, optional
        Y-axis label (default: "Reward")
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the figure
    figsize : str, optional
        Figure size name (default: "medium")
    line_width : float, optional
        Line width (default: 2.5)
    marker_size : float, optional
        Marker size (default: 8)
    show_error_bars : bool, optional
        Whether to show error bars (default: True)
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    """
    fig, ax = create_figure(size=figsize)

    if labels is None:
        labels = list(folder_il_data.keys())

    # Get common IL IDs for consistent x-axis
    common_il_ids = get_common_il_ids(folder_il_data)

    if not common_il_ids:
        print("[warn] No common IL IDs found across folders, using all IL IDs")
        # Use union of all IL IDs
        all_il_ids = set()
        for il_ids, _, _ in folder_il_data.values():
            all_il_ids.update(il_ids)
        common_il_ids = sorted(all_il_ids, key=lambda x: int(x) if x.isdigit() else 0)

    for idx, (folder_name, (il_ids, rew_mean, rew_std)) in enumerate(folder_il_data.items()):
        # Create mapping from IL ID to index
        il_to_idx = {il_id: i for i, il_id in enumerate(common_il_ids)}

        # Filter to common IL IDs and convert to numeric indices
        x_values = []
        y_values = []
        y_errors = []

        for il_id, mean_val, std_val in zip(il_ids, rew_mean, rew_std):
            if il_id in il_to_idx and np.isfinite(mean_val):
                x_values.append(il_to_idx[il_id])
                y_values.append(mean_val)
                y_errors.append(std_val)

        if not x_values:
            print(f"[warn] No valid data points for {folder_name}")
            continue

        x_values = np.array(x_values)
        y_values = np.array(y_values)
        y_errors = np.array(y_errors)

        # Sort by x values for proper line plotting
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = y_values[sort_idx]
        y_errors = y_errors[sort_idx]

        # Plot line with markers
        label = labels[idx] if idx < len(labels) else folder_name
        ax.plot(
            x_values,
            y_values,
            color=PlotTheme.get_color(idx),
            linewidth=line_width,
            marker='o',
            markersize=marker_size,
            label=label,
        )

        # Add error bars if requested and available
        if show_error_bars and np.any(y_errors > 0):
            ax.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                color=PlotTheme.get_color(idx),
                fmt='none',
                capsize=4,
                elinewidth=1.5,
                alpha=0.7,
            )

    # Set x-ticks to IL IDs
    ax.set_xticks(range(len(common_il_ids)))
    ax.set_xticklabels(common_il_ids)

    # Apply styling
    apply_scientific_style(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xlim=xlim,
        ylim=ylim,
        minor_ticks=True
    )

    # Add legend
    ax.legend(
        loc="best",
        framealpha=PlotTheme.LEGEND_FRAME_ALPHA,
        fontsize=PlotTheme.LEGEND_FONT_SIZE,
    )

    if save_path:
        save_figure(fig, save_path, dpi=PlotTheme.DPI_PUBLICATION)
        print(f"Saved to {save_path}")
        return None

    return fig, ax


def main():
    p = argparse.ArgumentParser(
        description="Plot reward lines from different meta_folders, with IL index on x-axis."
    )
    p.add_argument(
        "--meta-folder",
        nargs="+",
        required=True,
        help="One or more meta-folders to compare."
    )
    p.add_argument(
        "--control-config",
        required=True,
        help="Path to control_config.json containing {'reward_cfg': ...}"
    )
    p.add_argument(
        "--agg",
        choices=["sum", "mean"],
        default="mean",
        help="Aggregation used in compute_reward (default: mean)"
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each meta-folder (default: use folder names)"
    )
    p.add_argument(
        "--out-dir",
        default="plots",
        help="Directory to save generated plots (default: plots)"
    )
    p.add_argument(
        "--save-name",
        default="reward_lines.pdf",
        help="Output file name (default: reward_lines.pdf)"
    )
    p.add_argument(
        "--last",
        nargs="+",
        type=int,
        default=[0],
        help="Indices of last runs per IL (e.g., 0 1 for last and second last). Default: 0"
    )
    p.add_argument(
        "--xlabel",
        default="IL Index",
        help="X-axis label (default: IL Index)"
    )
    p.add_argument(
        "--ylabel",
        default="Reward",
        help="Y-axis label (default: Reward)"
    )
    p.add_argument(
        "--title",
        default=None,
        help="Plot title (default: None)"
    )
    p.add_argument(
        "--figsize",
        default="medium",
        choices=["tiny", "tiny2", "small", "medium", "large", "wide", "square", "portrait"],
        help="Figure size (default: medium)"
    )
    p.add_argument(
        "--no-error-bars",
        action="store_true",
        help="Disable error bars on the plot"
    )
    p.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        help="X-axis limits (e.g., --xlim 0 10)"
    )
    p.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        help="Y-axis limits (e.g., --ylim 0 1.5)"
    )
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
    folder_il_data: Dict[str, Tuple[List[int], np.ndarray, np.ndarray]] = {}

    for folder_path in args.meta_folder:
        meta_path = Path(folder_path)
        if not meta_path.is_dir():
            print(f"[warn] Skipping non-existent folder: {folder_path}")
            continue

        folder_name = meta_path.name

        print(f"Collecting data from {folder_path}...")
        il_ids, rew_mean, rew_std = collect_rewards_from_folder(
            meta_path,
            reward_cfg,
            agg=args.agg,
            last_k=args.last,
        )
        print(f"  Found {len(il_ids)} IL levels: {il_ids}")
        print(f"  Rewards: {rew_mean}")

        folder_il_data[folder_name] = (il_ids, rew_mean, rew_std)

    if not folder_il_data:
        print("[error] No data collected from any folder")
        return

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    save_path = str(out_dir / args.save_name)
    plot_reward_lines(
        folder_il_data,
        labels=args.labels,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title,
        save_path=save_path,
        figsize=args.figsize,
        show_error_bars=not args.no_error_bars,
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
    )


if __name__ == "__main__":
    main()
