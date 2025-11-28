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

def _agg_vals(arr: np.ndarray, mode: str = "mean") -> float:
    if arr.size == 0:
        return float("nan")
    if mode == "mean":
        return float(np.nanmean(arr))
    if mode == "median":
        return float(np.nanmedian(arr))
    if mode == "sum":
        return float(np.nansum(arr))
    # default
    return float(np.nanmean(arr))

def _collect_pct_from_logs(logs: List["ChannelLog"]) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-seq percentages (% units) across multiple ChannelLog objects."""
    pct0_all, pct1_all = [], []
    for lg in logs:
        per_seq = lg.get_interface_percentages(group_factor = 6)  # {seq: [pct0, pct1]}
        pct0_all.append([])
        for _, pair in per_seq.items():
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a, b = float(pair[0]), float(pair[1])
            if np.isfinite(a):
                pct0_all[-1].append(a)
        pct0_all[-1] = np.array(pct0_all[-1])
    return pct0_all


def summarize_rollouts_for_bars(rollouts: List["Rollout"], agg: str = "mean"):
    thr_vals, out_vals, reward_vals = [], [], []
    for r in rollouts:
        thr_arr = _flatten_dict_of_lists(r.throughput)  # dict[link]->list[float]
        out_arr = _flatten_dict_of_lists(r.outage)      # dict[link]->list[float]
        reward_arr = np.array(r.optional_data['reward'])
        thr_vals.append(_agg_vals(thr_arr, agg))
        out_vals.append(_agg_vals(out_arr, agg))
        reward_vals.append(_agg_vals(reward_arr))
    return thr_vals, out_vals, reward_vals

# ------------------------------- Plotting ------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from typing import Sequence, Dict, List, Optional, Tuple, Any
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
    bw_method: Optional[float] = None,   # e.g., 0.3 for smoother, None = default
):
    """
    Scientific-style violin plot (Matplotlib only).

    Enhancements:
    - Quartiles (Q1, median, Q3) and Tukey whiskers (1.5*IQR) overlaid on each violin
    - Optional raw points with gentle jitter
    - Minor gridlines, de-cluttered spines, consistent y-range default to [0,100]
    - Optional sample-size annotation under each x tick
    - Optional KDE bandwidth control via `bw_method`
    """
    # Collect data (each entry a 1D array-like of percentages)
    pct0 = _collect_pct_from_logs(logs)
    data = [np.asarray(a, dtype=float).ravel() for a in pct0]

    # Positions
    positions = np.arange(1, len(data) + 1, dtype=float)

    fig, ax = plt.subplots()

    # ---- Violin distributions ----
    vp = ax.violinplot(
        dataset=data,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8,
        bw_method=bw_method,
    )

    # Make violins slightly translucent (don’t set specific colors)
    for b in vp["bodies"]:
        b.set_alpha(0.6)

    # ---- Overlays: quartiles, median, whiskers ----
    for i, arr in enumerate(data, start=1):
        if arr.size == 0 or not np.isfinite(arr).any():
            continue

        q1, q2, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        # Tukey whiskers
        lo = np.min(arr[arr >= q1 - 1.5 * iqr]) if arr.size else q1
        hi = np.max(arr[arr <= q3 + 1.5 * iqr]) if arr.size else q3

        # IQR bar
        ax.vlines(i, q1, q3, lw=3, zorder=4)
        # Median mark
        ax.scatter([i], [q2], s=22, zorder=5)
        # Whiskers + caps
        ax.vlines(i, lo, hi, lw=1.2, zorder=4)
        ax.hlines([lo, hi], i - 0.08, i + 0.08, lw=1.2, zorder=4)

    # ---- Optional raw samples ----
    if show_points:
        rng = np.random.default_rng(0)
        for i, arr in enumerate(data, start=1):
            if arr.size == 0:
                continue
            xs = i + rng.uniform(-jitter, jitter, size=arr.size)
            ax.scatter(xs, arr, alpha=point_alpha, s=14, zorder=3, edgecolors="none")

    # ---- Axes cosmetics ----
    ax.set_xticks(positions)
    ax.set_xticklabels(list(labels))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13)
    if title:
        ax.set_title(title, fontsize=15)

    if y_range is not None:
        ax.set_ylim(*y_range)

    # Grid: major + minor on Y only
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.4, zorder=0)
    ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.25, zorder=0)

    # De-clutter spines & ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
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
):
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    
    total_bar_span = 3
    width = total_bar_span / max(len(labels), 1)
    bars = ax.bar(
        x, values,
        width=width * 0.9,          # a touch slimmer than the slot share
        color="cyan",
        edgecolor="black",
        hatch="///",
        alpha=0.9,
        linewidth=0.8,
    )
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=18)
    if title:
        ax.set_title(title)
    ax.set_xticks(x, labels)
    plt.setp(ax.get_xticklabels(), rotation=rotation)

    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    if annotate and len(values) > 0:
        vmax = np.nanmax(values)
        offset = 0.01 * (vmax if np.isfinite(vmax) else 1.0)
        for rect, v in zip(bars, values):
            label = "NaN" if not np.isfinite(v) else f"{v:.2f}"
            y = rect.get_height() if not np.isfinite(v) else v
            ax.text(
                rect.get_x() + rect.get_width()/2.0,
                y + offset,
                label,
                ha="center", va="bottom",
                fontsize=14, fontweight="bold"
            )

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Save to {save_path}")
        return None
    return fig, ax

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
            # Be defensive: if filtering/flattening fails, treat as zero reward
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
        # default: "sum"
        return float(sum(leaves))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_reader = ExpTraceReader(args.meta_folder)

    rollouts = []
    il_ids = []
    seq_interface = []
    for il_id, run_folder in sorted(data_reader.latest_folders.items()):
        il_ids.append(il_id)
        
        rollout = Rollout(FolderRollout(run_folder), {'reward': compute_reward})
        rollouts.append(rollout)
        
        try:
            channel = ChannelLog(FolderChannel(run_folder))
            seq_interface.append(channel)
        except:
            continue            

        
    thr_vals, out_vals, rewards  = summarize_rollouts_for_bars(rollouts, agg="mean")
    plot_bar_simple(np.array(thr_vals) / 1e6, il_ids, title=None, ylabel="Throughput (Mb/s)", save_path=str(out_dir / "tput.png"))
    plot_bar_simple(np.array(out_vals) * 100, il_ids, title=None, ylabel="Outage (%)", save_path=str(out_dir / "outage.png"))
    plot_bar_simple(rewards, il_ids, title=None, ylabel="Reward", save_path=str(out_dir / "reward.png"))
    
    try:
        plot_interface_percentages_boxplot(seq_interface, il_ids, title="Channel Utilization per rollout", ylabel="Factor", save_path=str(out_dir / "percentage.png"), show_points=False)
    except Exception as e:
        print(f"[error] Failed to plot interface percentages: {e}")
        pass
    


if __name__ == "__main__":
    main()