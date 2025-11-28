#!/usr/bin/env python3
"""
Fast plot of a single TensorBoard scalar with optional CSV caching.

Examples
--------
# Plot (with cache auto-created/updated)
python plot_tb_scalar.py /path/to/events --tag data/return --out return.png

# Only update/export CSV, no plot
python plot_tb_scalar.py /path/to/events --tag data/return --export-only

# Explicit cache path
python plot_tb_scalar.py /path/to/events --tag data/return --cache /tmp/return.csv
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util  # TF2+; simple_value still supported via .HasField

# --------------------- filesystem helpers ---------------------

def find_event_file(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        # newest first
        candidates = sorted(p.rglob("events.out.tfevents.*"),
                            key=lambda x: x.stat().st_mtime, reverse=True)
        if candidates:
            return str(candidates[0])
    raise FileNotFoundError(f"No TensorBoard event file found at {path!r}")

def default_cache_path(event_file: str, tag: str) -> str:
    safe_tag = tag.replace("/", "_")
    return str(Path(event_file).with_suffix(f".{safe_tag}.csv"))

# --------------------- CSV cache helpers ---------------------

CSV_HEADER = ["x", "y", "wall_time", "tag"]

def read_cache(csv_path: str, tag: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xs, ys, wall_times). Empty arrays if no file."""
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return (np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))
    xs, ys, wts = [], [], []
    with p.open("r", newline="") as f:
        r = csv.DictReader(f)
        # tolerate older headers: expect x,y,wall_time,tag
        for row in r:
            if row.get("tag") != tag:
                continue
            try:
                xs.append(float(row["x"]))
                ys.append(float(row["y"]))
                wts.append(float(row.get("wall_time", "nan")))
            except (KeyError, ValueError):
                continue
    if not xs:
        return (np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(wts, dtype=np.float64)

def append_cache(csv_path: str, rows: Iterable[Tuple[float, float, float, str]]) -> None:
    p = Path(csv_path)
    write_header = not p.exists() or p.stat().st_size == 0
    with p.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_HEADER)
        for x, y, wt, tag in rows:
            w.writerow([repr(float(x)), repr(float(y)), repr(float(wt)), tag])

# --------------------- loader ---------------------

def stream_tag(event_file: str, tag: str, x_mode: str,
               x_min_exclusive: Optional[float] = None, no_uptate_cache = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stream the event file and return arrays (xs, ys, wall_times) for a single tag.
    If x_min_exclusive is set, only values with x > x_min_exclusive are kept.
    Deduplicates consecutive identical x by keeping the last.
    """
    loader = event_file_loader.EventFileLoader(event_file)
    xs: List[float] = []
    ys: List[float] = []
    wts: List[float] = []
    
    if no_uptate_cache:
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(wts, dtype=np.float64)

    def maybe_push(x_val: float, y_val: float, wall_t: float):
        if x_min_exclusive is not None and not (x_val > x_min_exclusive):
            return
        if xs and x_val == xs[-1]:
            ys[-1] = y_val
            wts[-1] = wall_t
        else:
            xs.append(x_val); ys.append(y_val); wts.append(wall_t)

    for ev in loader.Load():
        if ev.WhichOneof("what") != "summary":
            continue
        x_val = ev.step if x_mode == "step" else float(ev.wall_time)
        for v in ev.summary.value:
            if v.tag != tag:
                continue
            if v.HasField("simple_value"):
                y_val = float(v.simple_value)
            elif v.HasField("tensor"):
                arr = tensor_util.make_ndarray(v.tensor)
                y_val = float(arr.reshape(1)[0])
            else:
                continue
            if not np.isfinite(y_val):
                continue
            maybe_push(float(x_val), y_val, float(ev.wall_time))

    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), np.asarray(wts, dtype=np.float64)

# --------------------- smoothing ---------------------

def ema_smooth(y: np.ndarray, alpha: float) -> np.ndarray:
    if y.size == 0: return y
    if not (0 <= alpha < 1): raise ValueError("EMA alpha must be in [0,1).")
    out = np.empty_like(y, dtype=np.float64)
    out[0] = y[0]
    for i in range(1, y.size):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * y[i]
    return out

def moving_average(y: np.ndarray, window: Optional[int]) -> np.ndarray:
    if not window or window <= 1: return y.copy()
    if window % 2 == 0: window += 1
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(ypad, kernel, mode="valid")

def median_smooth(y: np.ndarray, window: int) -> np.ndarray:
    """
    Centered median filter with edge padding so output length == input length.
    Ensures an odd window. No NaNs should be present upstream.
    """
    if not window or window <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1

    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")

    # Fast path using sliding_window_view (NumPy >= 1.20)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(ypad, window_shape=window)
        # sw shape: (len(y), window)
        return np.median(sw, axis=-1).astype(np.float64, copy=False)
    except Exception:
        # Portable fallback (slower but safe)
        out = np.empty_like(y, dtype=np.float64)
        for i in range(y.size):
            s = i
            e = i + window
            out[i] = np.median(ypad[s:e])
        return out

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Fast TB scalar plot with CSV cache.")
    ap.add_argument("event_path", type=str, help="Event file or directory containing it.")
    ap.add_argument("--tag", type=str, default="data/return", help="Scalar tag to plot.")
    ap.add_argument("--out", type=str, default=None, help="Output image path (default: '<tag>.png').")
    ap.add_argument("--show", action="store_true", help="Show the plot interactively.")
    ap.add_argument("--ema", type=float, default=0.9, help="EMA factor in [0,1).")
    ap.add_argument("--window", type=int, default=None, help="Centered moving-average window (odd). Overrides --ema.")
    ap.add_argument("--x", choices=["step", "wall"], default="step", help="X axis.")
    ap.add_argument("--title", type=str, default=None, help="Custom plot title.")
    ap.add_argument("--dpi", type=int, default=160, help="Figure DPI.")

    # cache options
    ap.add_argument("--cache", type=str, default=None, help="CSV cache path (default: auto).")
    ap.add_argument("--no-cache", action="store_true", help="Disable cache (load from event only).")
    ap.add_argument("--update-cache", action="store_true", default=True, help="Append new rows to cache (default on).")
    ap.add_argument("--no-update-cache", action="store_true")
    ap.add_argument("--export-only", action="store_true", help="Only write/update CSV, no plot.")
    args = ap.parse_args()

    event_file = find_event_file(args.event_path)
    cache_path = None if args.no_cache else (args.cache or default_cache_path(event_file, args.tag))

    # Load existing cache (if any)
    cached_xs = cached_ys = cached_wt = np.array([], dtype=np.float64)
    last_x = None
    if cache_path:
        cached_xs, cached_ys, cached_wt = read_cache(cache_path, args.tag)
        if cached_xs.size > 0:
            # last x defines the append boundary (strictly greater than)
            last_x = float(cached_xs[-1])

    # Stream ONLY new points beyond last_x (if any)
    xs_new, ys_new, wts_new = stream_tag(event_file, args.tag, args.x, x_min_exclusive=last_x, no_uptate_cache=args.no_update_cache)

    # Optionally append to cache
    if cache_path and args.update_cache and xs_new.size > 0:
        rows = ((x, y, wt, args.tag) for x, y, wt in zip(xs_new, ys_new, wts_new))
        append_cache(cache_path, rows)

    # If export-only, we’re done.
    if args.export_only:
        if cache_path:
            print(f"CSV updated at: {cache_path}")
        else:
            # If user disabled cache but asked export-only, write once-off CSV next to event.
            tmp_csv = default_cache_path(event_file, args.tag)
            rows = []
            # combine cached and new if any cache was read; otherwise dump everything we have
            if cached_xs.size > 0:
                for x, y, wt in zip(cached_xs, cached_ys, cached_wt):
                    rows.append((x, y, wt, args.tag))
            for x, y, wt in zip(xs_new, ys_new, wts_new):
                rows.append((x, y, wt, args.tag))
            append_cache(tmp_csv, rows)
            print(f"CSV written at: {tmp_csv}")
        return

    # Build plotting series: prefer cache (which already includes history), then new
    if cache_path and (cached_xs.size > 0 or xs_new.size > 0):
        xs_all = cached_xs
        ys_all = cached_ys
        if xs_new.size > 0:
            xs_all = np.concatenate([cached_xs, xs_new])
            ys_all = np.concatenate([cached_ys, ys_new])
    else:
        xs_all, ys_all = xs_new, ys_new  # no cache; plot what we streamed now

    if xs_all.size == 0:
        raise KeyError(f"Tag {args.tag!r} not found (or no finite values) in {event_file}")

    # Smooth
    if args.window and args.window > 1:
        # y_smooth = moving_average(ys_all, args.window)
        # smooth_desc = f"MA (w={args.window})"
        y_smooth = median_smooth(ys_all, args.window)
        smooth_desc = f"Median (w={args.window})"
    else:
        y_smooth = ema_smooth(ys_all, args.ema)
        smooth_desc = f"EMA (α={args.ema})"

    # Output path
    if args.out is None:
        safe_tag = args.tag.replace("/", "_")
        args.out = f"{safe_tag}.png"

    # Plot
    plt.figure(figsize=(8, 6), dpi=args.dpi)
    plt.plot(xs_all, ys_all, linestyle="--", linewidth=2.0, alpha=0.3, label="raw")
    plt.plot(xs_all, y_smooth, linewidth=2.0, label=smooth_desc)
    plt.xlabel(args.x.title(), fontsize = 18)
    plt.ylabel(args.tag.split("/")[-1], fontsize = 18)
    title = args.title or f"{args.tag}  —  {os.path.basename(event_file)}"
    # plt.title(title)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.ylim(bottom = -1000, top = 1500)
    # plt.yscale("symlog", linthresh=5e2) 
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    if args.show:
        plt.show()
    else:
        print(f"Saved plot to: {args.out}")
        if cache_path:
            print(f"(Cache: {cache_path})")

if __name__ == "__main__":
    main()
