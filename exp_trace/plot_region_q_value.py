from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from exp_trace.plot_utils import (
    create_figure,
    apply_scientific_style,
    save_figure,
    downsample_series,
)
from exp_trace.plot_config import PlotTheme  # for colors / legend font


REGION_DIR = Path("net_util/logs/12_10_v3/region_q_min")
N_REGIONS = 5
OUTPUT_FIG = REGION_DIR / "region_qmin_pi_mean.pdf"

# ---- CONFIG ----
# Maximum training step to show; set to None to disable clipping
MAX_STEP: Optional[float] = 400000
# e.g. MAX_STEP = 30000.0


def load_step_value(path: Path) -> Tuple[List[float], List[float]]:
    """
    Load (step, value) from a JSON file.

    Supported formats:
    - [[timestamp, step, value], ...]
    - list[{"step": ..., "value": ...}]
    - {"step": [...], "value": [...]}
    """
    with path.open("r") as f:
        data = json.load(f)

    # Case 1: list of [ts, step, value]
    if isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
        steps = [float(d[1]) for d in data]
        values = [float(d[2]) for d in data]
        return steps, values

    # Case 2: list of dicts
    if isinstance(data, list) and data and isinstance(data[0], dict):
        steps = [float(d["step"]) for d in data]
        values = [float(d["value"]) for d in data]
        return steps, values

    # Case 3: dict with arrays
    if isinstance(data, dict) and "step" in data and "value" in data:
        steps = [float(s) for s in data["step"]]
        values = [float(v) for v in data["value"]]
        return steps, values

    raise ValueError(f"Unsupported JSON format in {path}")


def clip_by_max_step(
    steps: List[float],
    values: List[float],
    max_step: Optional[float],
) -> Tuple[List[float], List[float]]:
    """Clip (steps, values) to steps <= max_step if max_step is not None."""
    if max_step is None:
        return steps, values

    clipped_steps: List[float] = []
    clipped_values: List[float] = []
    for s, v in zip(steps, values):
        if s <= max_step:
            clipped_steps.append(s)
            clipped_values.append(v)
    return clipped_steps, clipped_values


def plot_regions(stride: int = 1, smooth_window: int = 1) -> None:
    """
    Plot qmin_pi_mean for N_REGIONS regions on one scientific-style figure.

    Parameters
    ----------
    stride : int
        Downsample stride (1 = no downsampling).
    smooth_window : int
        Moving average window (1 = no smoothing).
    """
    fig, ax = create_figure(size="medium")

    for region_idx in range(N_REGIONS):
        path = REGION_DIR / f"region_{region_idx}.json"
        if not path.exists():
            print(f"[WARN] {path} not found, skip.")
            continue

        steps, values = load_step_value(path)
        steps, values = clip_by_max_step(steps, values, MAX_STEP)

        if not steps:
            print(f"[WARN] {path} has no points after clipping, skip.")
            continue

        # optional downsample for large files
        if stride > 1:
            steps, values = downsample_series(steps, values, stride=stride)

        # optional simple moving average smoothing
        if smooth_window > 1 and len(values) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            values = np.convolve(values, kernel, mode="same").tolist()

        color = PlotTheme.get_color(region_idx)
        ax.plot(
            steps,
            values,
            label=f"Case {region_idx}",
            color=color,
        )

    apply_scientific_style(
        ax,
        xlabel="Training step",
        ylabel=r"$q_{\min}^{\pi}$ (mean)",
        # title=r"Region-wise $q_{\min}^{\pi}$",
        ylim=(-200,None),
        minor_ticks=True,
    )
    ax.legend(fontsize=PlotTheme.LEGEND_FONT_SIZE)

    fig.tight_layout()
    save_figure(fig, OUTPUT_FIG)
    print(f"[INFO] Saved figure to: {OUTPUT_FIG}")


if __name__ == "__main__":
    # tweak stride / smooth_window / MAX_STEP at top as needed
    plot_regions(stride=1, smooth_window=1)
