from util.trace_collec import trace_filter, flatten_leaves
from exp_trace.util import ExpTraceReader, Rollout, FolderRollout, FolderChannel, ChannelLog
from typing import Any, Dict, Iterable, Callable, Union, List, Sequence, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from exp_trace.plot_config import PlotTheme
from exp_trace.plot_utils import (
    flatten_dict_of_lists, aggregate_values, create_figure,
    save_figure, apply_scientific_style, error_bar_format
)

Agg = Union[str, Callable[[Iterable[float]], float]]


def _collect_pct_from_logs(logs: List["ChannelLog"]) -> tuple[np.ndarray, np.ndarray]:
    pct0_all, pct1_all = [], []
    for lg in logs:
        per_seq = lg.get_interface_percentages(group_factor=6)
        pct0_all.append([])
        for _, pair in per_seq.items():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a, b = float(pair[0]), float(pair[1])
            if np.isfinite(a):
                pct0_all[-1].append(a)
        pct0_all[-1] = np.array(pct0_all[-1])
    return pct0_all


def summarize_single_rollout(r: "Rollout", agg: str = "mean") -> Tuple[float, float, float]:
    """
    Summarize a single rollout's throughput, outage, and reward.

    Parameters
    ----------
    r : Rollout
        Rollout data
    agg : str, optional
        Aggregation method (default: "mean")

    Returns
    -------
    tuple
        (throughput_array, outage_array, reward_array)
    """
    thr_arr = flatten_dict_of_lists(r.throughput)
    out_arr = flatten_dict_of_lists(r.outage)
    reward_arr = np.array(r.optional_data.get('reward', []))
    return thr_arr, out_arr, reward_arr


def aggregate_stats_per_il(il_to_rollouts: Dict[str, List["Rollout"]], agg: str = "mean"):
    """
    Aggregate statistics across rollouts for each IL.

    Parameters
    ----------
    il_to_rollouts : dict
        Dictionary mapping IL IDs to lists of rollouts
    agg : str, optional
        Aggregation method (default: "mean")

    Returns
    -------
    tuple
        (il_ids, thr_mean, thr_std, out_mean, out_std, rew_mean, rew_std)
    """
    il_ids = sorted(il_to_rollouts.keys())
    thr_mean, thr_std = [], []
    out_mean, out_std = [], []
    rew_mean, rew_std = [], []

    for il in il_ids:
        thr_list, out_list, rew_list = [], [], []
        for r in il_to_rollouts[il]:
            thr, out, rew = summarize_single_rollout(r, agg=agg)
            thr_list.extend(thr)
            out_list.extend(out)
            rew_list.extend(rew)

        thr_arr = np.asarray(thr_list, dtype=float)
        out_arr = np.asarray(out_list, dtype=float)
        rew_arr = np.asarray(rew_list, dtype=float)

        # Use aggregate_values utility
        thr_mean.append(aggregate_values(thr_arr, agg))
        out_mean.append(aggregate_values(out_arr, agg))
        rew_mean.append(aggregate_values(rew_arr, agg))

        # Calculate standard deviation
        thr_std.append(aggregate_values(thr_arr, "std") if thr_arr.size > 1 else 0.0)
        out_std.append(aggregate_values(out_arr, "std") if out_arr.size > 1 else 0.0)
        rew_std.append(aggregate_values(rew_arr, "std") if rew_arr.size > 1 else 0.0)

    return il_ids, thr_mean, thr_std, out_mean, out_std, rew_mean, rew_std


def plot_interface_percentages_boxplot(
    logs: Any,
    labels: Sequence[str],
    show_points: bool = False,
    jitter: float = 0.10,
    point_alpha: float = 0.25,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    y_range: Optional[Tuple[float, float]] = (0.0, 100.0),
    annotate_n: bool = True,
    bw_method: Optional[float] = None,
    figsize = "medium",
):
    if isinstance(logs, dict):
        data = []
        for lbl in labels:
            lgs = logs.get(lbl, [])
            vals = []
            for lg in lgs:
                per_seq = lg.get_interface_percentages(group_factor=6)
                for _, pair in per_seq.items():
                    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                        continue
                    a = float(pair[0])
                    if np.isfinite(a):
                        vals.append(a)
            data.append(np.asarray(vals, dtype=float))
    else:
        pct0 = _collect_pct_from_logs(logs)
        data = [np.asarray(a, dtype=float).ravel() for a in pct0]

    positions = np.arange(1, len(data) + 1, dtype=float)

    # fig, ax = plt.subplots()
    fig, ax = create_figure(size=figsize)

    vp = ax.violinplot(
        dataset=data,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8,
        bw_method=bw_method,
    )

    for b_i, b in enumerate(vp["bodies"]):
        b.set_edgecolor('black')
        b.set_linewidth(1.2)
        b.set_facecolor(PlotTheme.get_color(b_i))  # optional fill color
        b.set_alpha(0.6)

    for i, arr in enumerate(data, start=1):
        if arr.size == 0 or not np.isfinite(arr).any():
            continue
        q1, q2, q3 = np.percentile(arr, [25, 50, 75])
        ax.hlines(q2, i - 0.15, i + 0.15, lw=1.0, zorder=5, color="black")

    if show_points:
        rng = np.random.default_rng(0)
        for i, arr in enumerate(data, start=1):
            if arr.size == 0:
                continue
            xs = i + rng.uniform(-jitter, jitter, size=arr.size)
            ax.scatter(xs, arr, alpha=point_alpha, s=14, zorder=3, edgecolors="none")

    ax.set_xticks(positions)
    ax.set_xticklabels(list(labels))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=15)

    if y_range is not None:
        ax.set_ylim(*y_range)

    # Apply scientific styling
    apply_scientific_style(
        ax,
        ylabel=ylabel,
        title=title,
        minor_ticks=True
    )

    # Adjust tick sizes
    ax.tick_params(axis="x", labelsize=PlotTheme.FONT_SIZE_MEDIUM)
    ax.tick_params(axis="y", labelsize=PlotTheme.FONT_SIZE_MEDIUM)

    # plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, dpi=PlotTheme.DPI_PUBLICATION)
        print(f"Saved to {save_path}")
        return None
    return fig, ax


def plot_bar_simple(
    values: Sequence[float],
    labels: Sequence[str],
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    annotate: bool = True,
    rotation: int = 0,
    save_path: Optional[str] = None,
    ylimits: Tuple[Optional[float]] = (None, None),
    errors: Optional[Sequence[float]] = None,
    figsize: str = "medium",
):
    """
    Create a simple bar plot with scientific styling.

    Parameters
    ----------
    values : sequence
        Bar heights
    labels : sequence
        Bar labels
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    annotate : bool, optional
        Whether to annotate bars with values (default: True)
    rotation : int, optional
        Label rotation angle (default: 0)
    save_path : str, optional
        Output file path
    y_limits_low : float, optional
        Lower y-axis limit
    errors : sequence, optional
        Error bar values
    figsize : str, optional
        Figure size name (default: "medium")

    Returns
    -------
    tuple or None
        (fig, ax) if save_path is None, otherwise None
    """
    fig, ax = create_figure(size=figsize)

    x = np.arange(len(labels))
    width = 0.8

    bars = ax.bar(
        x,
        values,
        width=width,
        edgecolor="black",
        alpha=0.6,
        linewidth=1.2,
        yerr=errors,
        capsize=4,
        color=[PlotTheme.get_color(i) for i in range(len(values))]
    )

    # Apply scientific styling
    apply_scientific_style(
        ax,
        ylabel=ylabel,
        title=title,
        ylim=ylimits if ylimits is not None else None,
        minor_ticks=False
    )

    # Set x-ticks and rotation
    ax.set_xticks(x, labels)
    plt.setp(ax.get_xticklabels(), rotation=rotation)

    if annotate and len(values) > 0:
        vmax = np.nanmax(values)
        diff = vmax - 0 if ylimits[0] is None else vmax - ylimits[0]
        offset = 0.01 * (diff if np.isfinite(diff) else 1.0)
        for rect, v in zip(bars, values):
            label = "NaN" if not np.isfinite(v) else f"{v:.2f}"
            y = rect.get_height() if not np.isfinite(v) else v
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                y + offset,
                label,
                ha="center",
                va="bottom",
                fontsize=PlotTheme.FONT_SIZE_SMALL,
                fontweight="bold",
            )

    if save_path:
        save_figure(fig, save_path, dpi=PlotTheme.DPI_PUBLICATION)
        print(f"Saved to {save_path}")
        return None
    return fig, ax


#!/usr/bin/env python3
def main():
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description="Channel box plots and reward bar plot across folders.")
    p.add_argument("--meta-folder", required=True, help="One or more meta-folders to scan.")
    p.add_argument("--control-config", required=True, help="Path to control_config.json containing {'reward_cfg': ...}")
    p.add_argument("--agg", choices=["sum", "mean"], default="mean", help="Aggregation used in compute_reward")
    p.add_argument("--out-dir", default="plots", help="Directory to save generated plots")
    p.add_argument(
        "--showfliers",
        action="store_true",
        help="Show outliers in box plots",
    )
    p.add_argument(
        "--last",
        nargs="+",
        type=int,
        default=[0],
        help="Indices of last runs per IL (e.g., 0 1 for last and second last).",
    )
    args = p.parse_args()

    cfg_path = Path(args.control_config)
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            reward_cfg = json.load(f).get("reward_cfg")
    except Exception as e:
        print(f"[error] Failed to read control-config '{cfg_path}': {e}")
        return

    def compute_reward(
        record: Dict[str, Any],
        agg: Agg = "sum",
    ) -> float:
        if not isinstance(record, dict) or not reward_cfg:
            return 0.0
        try:
            filtered = trace_filter(record, reward_cfg)
            leaves = flatten_leaves(filtered)
        except Exception:
            return 0.0

        if not leaves:
            return 0.0

        if callable(agg):
            try:
                return float(agg(leaves))
            except Exception:
                return 0.0

        if agg == "mean":
            return float(sum(leaves) / len(leaves))
        return float(sum(leaves))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_reader = ExpTraceReader(args.meta_folder)

    il_to_rollouts: Dict[str, List[Rollout]] = {}
    il_to_logs: Dict[str, List[ChannelLog]] = {}
    seen_keys = set()

    for k in args.last:
        per_il = data_reader.pick_last_k_per_il(k)
        for il_id, run_folder in sorted(per_il.items()):
            key = (il_id, str(run_folder))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            print(f"k={k}, il={il_id}, folder={run_folder}")

            rollout = Rollout(FolderRollout(run_folder), {'reward': compute_reward})
            il_to_rollouts.setdefault(il_id, []).append(rollout)

            try:
                channel = ChannelLog(FolderChannel(run_folder))
                il_to_logs.setdefault(il_id, []).append(channel)
            except Exception as e:
                print(f"[warn] Failed to load ChannelLog for {run_folder}: {e}")
                continue

    if not il_to_rollouts:
        print("[warn] No rollouts collected.")
        return

    il_ids, thr_mean, thr_std, out_mean, out_std, rew_mean, rew_std = aggregate_stats_per_il(
        il_to_rollouts, agg=args.agg
    )

    thr_mean = np.asarray(thr_mean, dtype=float)
    thr_std = np.asarray(thr_std, dtype=float)
    out_mean = np.asarray(out_mean, dtype=float)
    out_std = np.asarray(out_std, dtype=float)
    rew_mean = np.asarray(rew_mean, dtype=float)
    rew_std = np.asarray(rew_std, dtype=float)

    plot_bar_simple(
        thr_mean / 1e6,
        il_ids,
        title=None,
        ylabel="Throughput (Mb/s)",
        save_path=str(out_dir / "tput.pdf"),
        figsize="tiny",
        ylimits=(5, 27),
    )
    plot_bar_simple(
        out_mean * 100,
        il_ids,
        title=None,
        ylabel="Outage (%)",
        save_path=str(out_dir / "outage.pdf"),
        figsize="tiny",
        ylimits=(0, 22),
    )
    plot_bar_simple(
        rew_mean,
        il_ids,
        title=None,
        ylabel="Reward",
        save_path=str(out_dir / "reward.pdf"),
        figsize="tiny",
        ylimits=(None, 1.6),
    )

    try:
        plot_interface_percentages_boxplot(
            il_to_logs,
            il_ids,
            # title="Channel Utilization per rollout",
            ylabel="Channel Utilization (%)",
            save_path=str(out_dir / "percentage.pdf"),
            show_points=False,
            y_range=(80, 100),
            figsize="tiny",
        )
    except Exception as e:
        print(f"[error] Failed to plot interface percentages: {e}")
        pass


if __name__ == "__main__":
    main()
