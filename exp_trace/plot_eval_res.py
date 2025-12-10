from util.trace_collec import trace_filter, flatten_leaves
from exp_trace.util import ExpTraceReader, Rollout, FolderRollout, FolderChannel, ChannelLog
from typing import Any, Dict, Iterable, Callable, Union, List, Sequence, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

Agg = Union[str, Callable[[Iterable[float]], float]]

import matplotlib
plt.rcParams.update({
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
    return float(np.nanmean(arr))


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
    thr_arr = _flatten_dict_of_lists(r.throughput)
    out_arr = _flatten_dict_of_lists(r.outage)
    reward_arr = np.array(r.optional_data['reward'])
    # thr = _agg_vals(thr_arr, agg)
    # out = _agg_vals(out_arr, agg)
    # rew = _agg_vals(reward_arr)
    return thr_arr, out_arr, reward_arr


# def summarize_rollouts_for_bars(rollouts: List["Rollout"], agg: str = "mean"):
#     thr_vals, out_vals, reward_vals = [], [], []
#     for r in rollouts:
#         thr, out, rew = summarize_single_rollout(r, agg=agg)
#         thr_vals.append(thr)
#         out_vals.append(out)
#         reward_vals.append(rew)
#     return thr_vals, out_vals, reward_vals


def aggregate_stats_per_il(il_to_rollouts: Dict[str, List["Rollout"]], agg: str = "mean"):
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

        thr_mean.append(float(np.nanmean(thr_arr)) if thr_arr.size else float("nan"))
        out_mean.append(float(np.nanmean(out_arr)) if out_arr.size else float("nan"))
        rew_mean.append(float(np.nanmean(rew_arr)) if rew_arr.size else float("nan"))

        thr_std.append(float(np.nanstd(thr_arr, ddof=1)) if thr_arr.size > 1 else 0.0)
        out_std.append(float(np.nanstd(out_arr, ddof=1)) if out_arr.size > 1 else 0.0)
        rew_std.append(float(np.nanstd(rew_arr, ddof=1)) if rew_arr.size > 1 else 0.0)

    return il_ids, thr_mean, thr_std, out_mean, out_std, rew_mean, rew_std


from matplotlib.ticker import AutoMinorLocator


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

    fig, ax = plt.subplots()

    vp = ax.violinplot(
        dataset=data,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8,
        bw_method=bw_method,
    )

    for b in vp["bodies"]:
        b.set_alpha(0.6)

    for i, arr in enumerate(data, start=1):
        if arr.size == 0 or not np.isfinite(arr).any():
            continue
        q1, q2, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        lo = np.min(arr[arr >= q1 - 1.5 * iqr]) if arr.size else q1
        hi = np.max(arr[arr <= q3 + 1.5 * iqr]) if arr.size else q3
        ax.vlines(i, q1, q3, lw=3, zorder=4)
        ax.scatter([i], [q2], s=22, zorder=5)
        ax.vlines(i, lo, hi, lw=1.2, zorder=4)
        ax.hlines([lo, hi], i - 0.08, i + 0.08, lw=1.2, zorder=4)

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

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.4, zorder=0)
    ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.25, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    if annotate_n:
        for i, arr in enumerate(data, start=1):
            n = np.sum(np.isfinite(arr))
            ax.text(i, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom", fontsize=9)

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
    y_limits_low: Optional[float] = None,
    errors: Optional[Sequence[float]] = None,
):
    x = np.arange(len(labels))
    fig, ax = plt.subplots()

    total_bar_span = 3
    width = total_bar_span / max(len(labels), 1)
    bars = ax.bar(
        x,
        values,
        width=width * 0.9,
        edgecolor="black",
        hatch="///",
        alpha=0.9,
        linewidth=0.8,
        yerr=errors,
        capsize=4,
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

    if y_limits_low is not None:
        ax.set_ylim(bottom=y_limits_low)

    if annotate and len(values) > 0:
        vmax = np.nanmax(values)
        diff = vmax - 0 if y_limits_low is None else vmax - y_limits_low
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
                fontsize=14,
                fontweight="bold",
            )

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Save to {save_path}")
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
    )
    plot_bar_simple(
        out_mean * 100,
        il_ids,
        title=None,
        ylabel="Outage (%)",
        save_path=str(out_dir / "outage.pdf"),
        y_limits_low=None,
    )
    plot_bar_simple(
        rew_mean,
        il_ids,
        title=None,
        ylabel="Reward",
        save_path=str(out_dir / "reward.pdf"),
    )

    try:
        plot_interface_percentages_boxplot(
            il_to_logs,
            il_ids,
            title="Channel Utilization per rollout",
            ylabel="Factor",
            save_path=str(out_dir / "percentage.pdf"),
            show_points=False,
            y_range=None,
        )
    except Exception as e:
        print(f"[error] Failed to plot interface percentages: {e}")
        pass


if __name__ == "__main__":
    main()
