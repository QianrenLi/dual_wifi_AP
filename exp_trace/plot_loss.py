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
from exp_trace.plot_config import PlotTheme


LOSS_DIR = Path("net_util/logs/12_10_v3/loss")
OUTPUT_DIR = LOSS_DIR

MAX_STEP: Optional[float] = 400000  # e.g. 30000.0


def load_step_value(path: Path) -> Tuple[List[float], List[float]]:
    with path.open("r") as f:
        data = json.load(f)

    if isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
        steps = [float(d[1]) for d in data]
        values = [float(d[2]) for d in data]
        return steps, values

    if isinstance(data, list) and data and isinstance(data[0], dict):
        steps = [float(d["step"]) for d in data]
        values = [float(d["value"]) for d in data]
        return steps, values

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
    if max_step is None:
        return steps, values
    clipped_steps: List[float] = []
    clipped_values: List[float] = []
    for s, v in zip(steps, values):
        if s <= max_step:
            clipped_steps.append(s)
            clipped_values.append(v)
    return clipped_steps, clipped_values


def plot_loss(
    name: str,
    color_index: int,
    stride: int = 1,
    smooth_window: int = 1,
) -> None:
    json_path = LOSS_DIR / f"{name}.json"
    if not json_path.exists():
        print(f"[WARN] {json_path} not found, skip.")
        return

    steps, values = load_step_value(json_path)
    steps, values = clip_by_max_step(steps, values, MAX_STEP)

    if not steps:
        print(f"[WARN] {json_path} has no points after clipping, skip.")
        return

    if stride > 1:
        steps, values = downsample_series(steps, values, stride=stride)

    if smooth_window > 1 and len(values) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        values = np.convolve(values, kernel, mode="same").tolist()

    fig, ax = create_figure(size="medium")
    color = PlotTheme.get_color(color_index)

    ax.plot(steps, values, color=color, label=f"{name} loss")

    pretty_name = name.capitalize()
    apply_scientific_style(
        ax,
        xlabel="Training step",
        ylabel=f"{pretty_name} loss",
        # title=f"{pretty_name} loss",
        minor_ticks=True,
    )
    # ax.legend(fontsize=PlotTheme.LEGEND_FONT_SIZE)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{name}_loss.pdf"
    save_figure(fig, out_path)
    print(f"[INFO] Saved {name} loss figure to: {out_path}")


if __name__ == "__main__":
    plot_loss("actor", color_index=0, stride=1, smooth_window=1)
    plot_loss("belief", color_index=1, stride=1, smooth_window=1)
    plot_loss("critic", color_index=2, stride=1, smooth_window=1)
