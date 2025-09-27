#!/usr/bin/env python3
# plot_metrics.py
# Parse trial_*/train.log and plot metrics either per-epoch (concatenated) or per-trial (min/mean/max).

import argparse, re, sys
from pathlib import Path
from datetime import datetime

EPOCH_RE = re.compile(r"\[epoch\s+(\d+)\s*/\s*(\d+)\]")
KV_RE    = re.compile(r"([A-Za-z_][\w\./]*)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
TS_RE    = re.compile(r"^trial_(\d{8}-\d{6})$")  # trial_YYYYMMDD-HHMMSS
DEFAULT_METRICS = ["loss", "pol_loss", "val_loss", "entropy", "kl", "clipfrac"]

def trial_sort_key(p: Path):
    m = TS_RE.match(p.name)
    if m:
        try:
            return int(datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").timestamp())
        except Exception:
            pass
    try:
        return int(p.stat().st_mtime)
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser(description="Plot learning metrics from train.log")
    ap.add_argument("--root", required=True, help="Directory with trial_* folders")
    ap.add_argument("--recent", type=int, default=0, help="Use most recent N trials (0=all)")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                    help=f"Comma-separated metric names (default: {','.join(DEFAULT_METRICS)})")
    ap.add_argument("--out", default="metrics.png", help="Output figure path (PNG)")
    ap.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    ap.add_argument("--gap", type=int, default=3, help="Epoch gap between trials (unit=epoch)")
    ap.add_argument("--show", action="store_true", help="Also show figure")
    ap.add_argument("--unit", choices=["epoch", "trial"], default="epoch",
                    help="Plot per-epoch (concatenated) or per-trial aggregates (min/mean/max)")
    ap.add_argument("--trial-agg", choices=["all", "mean", "max", "min"], default="all",
                    help="Which aggregates to plot when --unit trial")
    ap.add_argument("--label-trials", action="store_true",
                    help="Label x-axis with trial folder names when --unit trial")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERR: root not found: {root}", file=sys.stderr); sys.exit(1)

    # Discover trials: newest first -> keep N -> reverse to oldest->newest
    trials = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("trial_")]
    trials.sort(key=trial_sort_key, reverse=True)
    if args.recent > 0:
        trials = trials[:args.recent]
    trials = list(reversed(trials))

    wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not wanted:
        print("ERR: no metrics requested", file=sys.stderr); sys.exit(1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4.8), dpi=args.dpi)

    if args.unit == "epoch":
        # Concatenate epochs across trials with gaps
        series = {m: {"x": [], "y": []} for m in wanted}
        x_offset, used_trials = 0, 0
        for tdir in trials:
            logp = tdir / "train.log"
            if not logp.exists(): continue
            last_epoch = 0
            with open(logp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    me = EPOCH_RE.search(line)
                    if not me: continue
                    epoch = int(me.group(1))
                    last_epoch = max(last_epoch, epoch)
                    pairs = KV_RE.findall(line)
                    if not pairs: continue
                    kv = dict(pairs)
                    for met in wanted:
                        if met in kv:
                            try:
                                val = float(kv[met])
                                series[met]["x"].append(x_offset + epoch)
                                series[met]["y"].append(val)
                            except ValueError:
                                pass
            if last_epoch > 0:
                x_offset += last_epoch + args.gap
                used_trials += 1

        any_plotted = False
        for met in wanted:
            xs, ys = series[met]["x"], series[met]["y"]
            if xs and ys:
                plt.plot(xs, ys, label=met); any_plotted = True
        plt.xlabel("concatenated epoch (oldest → newest, gaps between trials)")
        plt.ylabel("value")
        plt.title(f"Learning metrics · {used_trials} trial(s)")
        if any_plotted: plt.legend(loc="best")

    else:  # args.unit == "trial"
        # Aggregate within each trial: min/mean/max for each metric
        # x-axis is trial index (1..T), optionally label with trial folder name
        trial_names = []
        per_metric = {m: {"min": [], "mean": [], "max": []} for m in wanted}
        used_idx = []
        t_idx = 0
        for tdir in trials:
            logp = tdir / "train.log"
            if not logp.exists(): continue
            t_idx += 1
            trial_names.append(tdir.name)
            used_idx.append(t_idx)
            # collect values per metric in this trial
            vals = {m: [] for m in wanted}
            with open(logp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    pairs = KV_RE.findall(line)
                    if not pairs: continue
                    kv = dict(pairs)
                    for met in wanted:
                        if met in kv:
                            try:
                                vals[met].append(float(kv[met]))
                            except ValueError:
                                pass
            # compute aggregates (if any values exist)
            for met in wanted:
                vs = vals[met]
                if vs:
                    vmin = min(vs); vmax = max(vs); vmean = sum(vs)/len(vs)
                else:
                    vmin = vmax = vmean = float("nan")
                per_metric[met]["min"].append(vmin)
                per_metric[met]["mean"].append(vmean)
                per_metric[met]["max"].append(vmax)

        # Decide which aggregates to draw
        aggs = ["min", "mean", "max"] if args.trial_agg == "all" else [args.trial_agg]
        any_plotted = False
        for met in wanted:
            for agg in aggs:
                ys = per_metric[met][agg]
                if any(x == x for x in ys):  # any non-NaN
                    plt.plot(used_idx, ys, label=f"{met}({agg})")
                    any_plotted = True

        plt.xlabel("trial index (oldest → newest)")
        plt.ylabel("value")
        plt.title(f"Per-trial aggregates ({'/'.join(aggs)}) · {len(used_idx)} trial(s)")
        if args.label-trials if False else args.label_trials:
            plt.xticks(used_idx, trial_names, rotation=45, ha="right")
        if any_plotted: plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved figure to: {args.out}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
