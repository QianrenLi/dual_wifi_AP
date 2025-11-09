from typing import Any, Dict, Iterable, Optional, Callable, Union
from util.trace_collec import trace_filter, flatten_leaves  # your project utilities

import numpy as np

Agg = Union[str, Callable[[Iterable[float]], float]]  # "sum" | "mean" | custom reducer

def compute_reward(
    record: Dict[str, Any],
    reward_cfg: Optional[Dict[str, Any]],
    agg: Agg = "sum",
) -> float:
    """
    Compute reward for a single rollout record.

    Parameters
    ----------
    record : dict
        One parsed JSON line (a rollout record).
    reward_cfg : dict | None
        The `reward_cfg` object (e.g., from control_config.json). If None, returns 0.0.
    agg : {"sum","mean"} or callable
        How to aggregate the leaf values returned by `flatten_leaves`.
        - "sum"  -> sum of leaves (default)
        - "mean" -> average of leaves
        - callable(iterable[float]) -> custom reducer

    Returns
    -------
    float
        The computed reward.
    """
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


def compute_channel_stats(log_path: str):
    """
    Read a .log file with lines like: "INFO - <channel>, <seq>".

    Returns a dict with:
      - distribution: {channel: {"count": int, "percent": float}, ...}   # overall counts
      - seq_hist:     {channel: {seq: count, ...}, ...}                  # per-seq counts per channel
      - total:        int                                                # total parsed pairs
      - seq_totals:   {seq: total_count_for_that_seq}
      - seq_ch0:      {seq: count_of_channel0_for_that_seq}
      - seq_ch0_frac: {seq: ch0_count/total_count_for_that_seq}          # fraction of channel 0 per seq
      - ch0_fraction_list: [ch0_fraction_per_seq, ...]                   # convenient list for box plots
    """
    import re
    from collections import Counter, defaultdict

    pair_re = re.compile(r"INFO\s*-\s*(\d+)\s*,\s*(\d+)")
    ch_counts = Counter()                 # overall per-channel counts
    seq_hist = defaultdict(Counter)       # seq_hist[channel][seq] -> count
    seq_totals = Counter()                # total observations per seq
    seq_ch0 = Counter()                   # channel 0 observations per seq

    total = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pair_re.search(line)
            if not m:
                continue
            ch = int(m.group(1))
            seq = int(m.group(2))

            ch_counts[ch] += 1
            seq_hist[ch][seq] += 1

            seq_totals[seq] += 1
            if ch == 0:
                seq_ch0[seq] += 1

            total += 1

    # overall distribution
    denom = max(total, 1)
    distribution = {
        ch: {"count": cnt, "percent": 100.0 * cnt / denom}
        for ch, cnt in sorted(ch_counts.items())
    }

    # convert nested Counters to plain dicts (sorted by seq)
    seq_hist_plain = {ch: dict(sorted(c.items())) for ch, c in seq_hist.items()}

    # per-seq channel-0 fraction
    seq_ch0_frac = {}
    for s, tot in seq_totals.items():
        if tot > 0:
            seq_ch0_frac[s] = float(seq_ch0.get(s, 0)) / float(tot)
        else:
            seq_ch0_frac[s] = 0.0

    ch0_fraction_list = [seq_ch0_frac[s] for s in sorted(seq_ch0_frac.keys())]

    return {
        "distribution": distribution,
        "seq_hist": seq_hist_plain,
        "total": total if total != 1 or sum(ch_counts.values()) == 1 else 0,
        "seq_totals": dict(sorted(seq_totals.items())),
        "seq_ch0": dict(sorted(seq_ch0.items())),
        "seq_ch0_frac": dict(sorted(seq_ch0_frac.items())),
        "ch0_fraction_list": ch0_fraction_list,  # <- feed this to your box plot
    }


import matplotlib.pyplot as plt

def plot_channel_box(ch_distributions, folder_names, title=None, out_path=None, showfliers=False):
    """
    Box plot for per-folder channel distributions.

    Parameters
    ----------
    ch_distributions : List[List[int]]
        For each folder, a list of per-seq counts for the chosen channel.
        Example: [[3,2,1], [5,1], ...] aligned with folder_names.
    folder_names : List[str]
        Labels for x-axis, one per folder.
    title : str | None
        Optional plot title.
    out_path : str | None
        If provided, save the figure to this path. Otherwise just return ax.
    showfliers : bool
        Whether to show outliers (points outside whiskers).

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots()
    ax.boxplot(ch_distributions, labels=folder_names, showfliers=showfliers)
    ax.set_xlabel("Folder")
    ax.set_ylabel("Per-seq count")
    if title:
        ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    return ax


def plot_reward_bar(reward_totals, folder_names, title=None, out_path=None, ylabel = None):
    """
    Bar plot for per-folder total rewards.

    Parameters
    ----------
    reward_totals : List[float]
        Total reward per folder, aligned with folder_names.
    folder_names : List[str]
        Labels for x-axis, one per folder.
    title : str | None
        Optional plot title.
    out_path : str | None
        If provided, save the figure to this path. Otherwise just return ax.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(folder_names)), reward_totals, tick_label=folder_names,             color="cyan",
            edgecolor="black",
            hatch= "//",
            alpha= 0.9,
            # width=0.08,
            linewidth=1.0,
            zorder=3,)
    # ax.set_xlabel("Folder")
    ax.set_ylabel(ylabel, fontsize=18) if ylabel else ax.set_ylabel("Total reward")
    if title:
        ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    # Add value labels with slight offset
    for rect in bars:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height + 0.01 * max(reward_totals),  # small offset above bar
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=14,
            weight="bold",
        )

    # Make it pretty: subtle grid + clean axes
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    # ax.set_xlim([-0.25, 0.5])
    # ax.set_ylim(bottom = 9)
        
    if out_path:
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
    return ax


# -------- Optional helper to build input for plot_channel_box --------
def build_channel_distributions(stats_list, channel):
    """
    Convert a list of compute_channel_stats(...) results into the
    ch_distributions expected by plot_channel_box for a given channel.

    Parameters
    ----------
    stats_list : List[dict]
        Each item is the dict returned by compute_channel_stats(log_path).
    channel : int
        0 or 1 (or any channel key present).

    Returns
    -------
    List[List[int]]
        Each element is a list of per-seq counts for that folder.
    """
    out = []
    for stats in stats_list:
        arr = stats.get("ch0_fraction_list", [])
        out.append(arr if arr else [0])
    return out



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
    p.add_argument("--folders", nargs="+", help="Folders to process (each with output.log and rollout.jsonl/roolout.jsonl)")
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

    def _find_rollout(folder: Path):
        # Accept common typo too
        for name in ("rollout.jsonl", "roolout.jsonl"):
            pth = folder / name
            if pth.exists():
                return pth
        return None

    # ---- Collect per-folder stats ----
    folder_names = []
    stats_list = []       # for box plots (channel 0/1)
    reward_totals = []    # for bar plot
    tputs_totals = []
    outage_totals = []
    belief_totals = []

    for folder in args.folders:
        root = Path(folder)
        if not root.is_dir():
            print(f"[warn] Skip non-directory: {root}")
            continue

        folder_names.append(root.name)

        # # Channel stats
        # log_path = root / "output.log"
        # if log_path.exists():
        #     try:
        #         stats = compute_channel_stats(str(log_path))
        #     except Exception as e:
        #         print(f"[warn] compute_channel_stats failed for {root}: {e}")
        #         stats = {"seq_hist": {}}
        # else:
        #     print(f"[warn] No output.log in {root}")
        #     stats = {"seq_hist": {}}
        # stats_list.append(stats)

        # Reward total
        ro_path = _find_rollout(root)
        total_reward = []
        tput = []
        outage = []
        beliefs = []
        if ro_path:
            try:
                with ro_path.open("r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        total_reward.append(compute_reward(rec, reward_cfg, agg=args.agg))
                        tput.append(rec['stats']['flow_stat']['6203@128']['throughput'])
                        outage.append(rec['stats']['flow_stat']['6203@128']['outage_rate'])
                        beliefs.append(rec['res']['belief'][0])
            except Exception as e:
                print(f"[warn] Failed reading rewards in {root}: {e}")
        else:
            print(f"[warn] No rollout.jsonl/roolout.jsonl in {root}")
        reward_totals.append( np.mean(total_reward) )
        tputs_totals.append(np.mean(tput))
        outage_totals.append(np.mean(outage))
        belief_totals.append(np.mean(beliefs))

    # ---- Build channel distributions and plot ----
    ch0_distributions = build_channel_distributions(stats_list, channel=0)

    # plot_channel_box(
    #     ch_distributions=ch0_distributions,
    #     folder_names=folder_names,
    #     title="Channel 0 per-seq counts (box)",
    #     out_path=str(out_dir / "channel0_box.png"),
    #     showfliers=args.showfliers,
    # )
    # plot_channel_box(
    #     ch_distributions=ch1_distributions,
    #     folder_names=folder_names,
    #     title="Channel 1 per-seq counts (box)",
    #     out_path=str(out_dir / "channel1_box.png"),
    #     showfliers=args.showfliers,
    # )
    
    plot_reward_bar(
        reward_totals=reward_totals,
        folder_names=[0,1,2,3],
        title=f"",
        out_path=str(out_dir / "reward.png"),
        ylabel="Average Reward"
    )
    
    plot_reward_bar(
        reward_totals=tputs_totals,
        folder_names=[0,1,2,3],
        title=f"",
        out_path=str(out_dir / "tput.png"),
        ylabel="Average Throughput"
    )
    
    plot_reward_bar(
        reward_totals=outage_totals,
        folder_names=[0,1,2,3],
        title=f"",
        out_path=str(out_dir / "outage.png"),
        ylabel="Average Outage"
    )
    

    plot_reward_bar(
        reward_totals=belief_totals,
        folder_names=[0,1,2,3],
        title=f"",
        out_path=str(out_dir / "belief.png"),
        ylabel="Average Outage"
    )
    

    # ---- Console summary ----
    print("\nSaved plots:")
    print(f"  {out_dir / 'channel0_box.png'}")
    print(f"  {out_dir / 'channel1_box.png'}")
    print(f"  {out_dir / 'reward_bar.png'}")
    

if __name__ == "__main__":
    main()