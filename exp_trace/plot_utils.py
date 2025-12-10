"""
Common utilities and helper functions for plotting.

This module provides reusable functions for common plotting tasks,
reducing code duplication and ensuring consistency across all plots.
"""

from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from exp_trace.plot_config import PlotTheme


# Common regex patterns
_TS_RE = re.compile(r"^trial_(\d{8}-\d{6})$")  # trial_YYYYMMDD-HHMMSS


def trial_sort_key(p: Path) -> Tuple[int, str]:
    """
    Sort trial directories by timestamp.

    Parameters
    ----------
    p : Path
        Path to trial directory

    Returns
    -------
    tuple
        (timestamp, name) for sorting
    """
    m = _TS_RE.match(p.name)
    if m:
        try:
            ts = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S")
            return (int(ts.timestamp()), p.name)
        except Exception:
            pass
    try:
        return (int(p.stat().st_mtime), p.name)
    except Exception:
        return (0, p.name)


def list_trials(root: Path, recent: int = 0,
               pattern: str = "IL_*") -> List[Path]:
    """
    List trial directories in chronological order.

    Parameters
    ----------
    root : Path
        Root directory containing trials
    recent : int, optional
        Number of most recent trials to include (0 = all)
    pattern : str, optional
        Glob pattern for trial directories (default: "IL_*")

    Returns
    -------
    list
        List of trial directories sorted by timestamp (oldest first)
    """
    trials = [p for p in root.iterdir()
             if p.is_dir() and p.name.startswith(pattern)]
    trials.sort(key=trial_sort_key, reverse=True)  # newest first

    if recent and recent > 0:
        trials = trials[:recent]

    trials.reverse()  # oldest -> newest for temporal continuity
    return trials


def create_figure(size: str = "medium",
                nrows: int = 1,
                ncols: int = 1,
                sharex: bool = False,
                sharey: bool = False,
                **kwargs) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Create a figure with consistent sizing and styling.

    Parameters
    ----------
    size : str, optional
        Base figure size name (default: "medium")
    nrows : int, optional
        Number of subplot rows (default: 1)
    ncols : int, optional
        Number of subplot columns (default: 1)
    sharex : bool, optional
        Whether subplots should share x-axis (default: False)
    sharey : bool, optional
        Whether subplots should share y-axis (default: False)
    **kwargs
        Additional arguments passed to plt.subplots

    Returns
    -------
    tuple
        (figure, axes)
    """
    figsize = PlotTheme.get_figure_size(size, nrows, ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        **kwargs
    )

    # Apply grid styling to all axes
    if nrows == 1 and ncols == 1:
        PlotTheme.apply_grid_style(axes)
        axes.tick_params(which="both", direction="in")
    else:
        # axes is a numpy array when multiple subplots
        for ax in axes.flat if hasattr(axes, 'flat') else [axes]:
            PlotTheme.apply_grid_style(ax)
            ax.tick_params(which="both", direction="in")

    return fig, axes


def save_figure(fig: plt.Figure,
               path: Union[str, Path],
               dpi: Optional[int] = None,
               bbox_inches: str = "tight",
               **kwargs) -> None:
    """
    Save a figure with consistent settings.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    path : str or Path
        Output file path
    dpi : int, optional
        Resolution (default: from theme)
    bbox_inches : str, optional
        Bounding box setting (default: "tight")
    **kwargs
        Additional arguments passed to fig.savefig
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if dpi is None:
        dpi = PlotTheme.DPI_PUBLICATION

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        **kwargs
    )


def apply_scientific_style(ax: plt.Axes,
                          xlabel: Optional[str] = None,
                          ylabel: Optional[str] = None,
                          title: Optional[str] = None,
                          xlim: Optional[Tuple[float, float]] = None,
                          ylim: Optional[Tuple[float, float]] = None,
                          minor_ticks: bool = True) -> None:
    """
    Apply scientific styling to an axis.

    Parameters
    ----------
    ax : plt.Axes
        Axis to style
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Axis title
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    minor_ticks : bool, optional
        Whether to show minor ticks (default: True)
    """
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=PlotTheme.FONT_SIZE_LARGE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=PlotTheme.FONT_SIZE_LARGE)
    if title:
        ax.set_title(title, fontsize=PlotTheme.FONT_SIZE_XLARGE)

    # Set limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Add minor ticks
    if minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Apply grid
    PlotTheme.apply_grid_style(ax, major=True)
    PlotTheme.apply_grid_style(ax, major=False)

    # Hide spines for scientific look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # tick font
    ax.tick_params(axis='both', which='major', labelsize=PlotTheme.FONT_SIZE_MEDIUM)


def flatten_dict_of_lists(d: Dict[str, List[float]]) -> np.ndarray:
    """
    Flatten a dictionary of lists into a single numpy array.

    Parameters
    ----------
    d : dict
        Dictionary where values are lists of floats

    Returns
    -------
    np.ndarray
        Flattened array of all values
    """
    if not d:
        return np.array([], dtype=float)

    parts = []
    for v in d.values():
        a = np.asarray(v, dtype=float)
        parts.append(a)

    return np.concatenate(parts) if parts else np.array([], dtype=float)


def aggregate_values(arr: np.ndarray,
                    mode: str = "mean") -> float:
    """
    Aggregate array values using specified method.

    Parameters
    ----------
    arr : np.ndarray
        Input array
    mode : str, optional
        Aggregation method ("mean", "median", "sum", "std")
        (default: "mean")

    Returns
    -------
    float
        Aggregated value
    """
    if arr.size == 0:
        return float("nan")

    if mode == "mean":
        return float(np.nanmean(arr))
    elif mode == "median":
        return float(np.nanmedian(arr))
    elif mode == "sum":
        return float(np.nansum(arr))
    elif mode == "std":
        return float(np.nanstd(arr, ddof=1))
    else:
        return float(np.nanmean(arr))


def moving_average(data: List[float],
                  window: int,
                  mode: str = "same") -> List[float]:
    """
    Compute moving average of a data series.

    Parameters
    ----------
    data : list
        Input data series
    window : int
        Window size for averaging
    mode : str, optional
        Convolution mode ("same", "valid", "full") (default: "same")

    Returns
    -------
    list
        Smoothed data series
    """
    if window <= 1 or len(data) < window:
        return data

    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode=mode).tolist()


def discounted_running_sum(data: List[float],
                          gamma: float) -> List[float]:
    """
    Compute discounted running sum of a series.

    Parameters
    ----------
    data : list
        Input data series
    gamma : float
        Discount factor (0 < gamma <= 1)

    Returns
    -------
    list
        Discounted running sum
    """
    acc = 0.0
    out: List[float] = []

    for r in data:
        acc = acc * gamma + r
        out.append(acc)

    return out


def downsample_series(x: List[float], y: List[float],
                     stride: int) -> Tuple[List[float], List[float]]:
    """
    Downsample a data series by stride.

    Parameters
    ----------
    x : list
        X values
    y : list
        Y values
    stride : int
        Downsampling stride

    Returns
    -------
    tuple
        (downsampled_x, downsampled_y)
    """
    if stride <= 1:
        return x, y

    # Downsample
    x_down = x[::stride]
    y_down = y[::stride]

    # Ensure the last point is included
    if x_down and x_down[-1] != x[-1]:
        x_down.append(x[-1])
        y_down.append(y[-1])

    return x_down, y_down


def create_box_plot_with_stats(ax: plt.Axes,
                             data: List[np.ndarray],
                             labels: Sequence[str],
                             showfliers: bool = False,
                             patch_artist: bool = True,
                             **kwargs) -> Dict[str, Any]:
    """
    Create a box plot with statistical styling.

    Parameters
    ----------
    ax : plt.Axes
        Axis to plot on
    data : list
        List of data arrays
    labels : sequence
        Labels for each box
    showfliers : bool, optional
        Whether to show outliers (default: False)
    patch_artist : bool, optional
        Whether to fill boxes (default: True)
    **kwargs
        Additional arguments passed to ax.boxplot

    Returns
    -------
    dict
        Dictionary containing box plot elements
    """
    bp = ax.boxplot(data,
                   tick_labels=labels,
                   showfliers=showfliers,
                   patch_artist=patch_artist,
                   **kwargs)

    # Style the boxes
    if patch_artist:
        colors = [PlotTheme.COLORS[f"primary{i % 10}"]
                 for i in range(len(data))]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    # Style the whiskers and caps
    for whisker in bp['whiskers']:
        whisker.set_linestyle('-')
        whisker.set_linewidth(1.5)

    for cap in bp['caps']:
        cap.set_linewidth(1.5)

    # Style the median lines
    for median in bp['medians']:
        median.set_linewidth(2)
        median.set_color('black')

    return bp


def add_statistical_annotation(ax: plt.Axes,
                             x1: float, x2: float,
                             y: float,
                             text: str,
                             height: float = 0.05,
                             fontsize: int = 12) -> None:
    """
    Add statistical significance annotation between two points.

    Parameters
    ----------
    ax : plt.Axes
        Axis to annotate
    x1, x2 : float
        X positions to connect
    y : float
        Y position of the annotation
    text : str
        Text to display (e.g., p-value)
    height : float, optional
        Height of the annotation brackets (default: 0.05)
    fontsize : int, optional
        Font size for the annotation (default: 12)
    """
    # Get y-axis limits to scale the height
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    actual_height = height * yrange

    # Draw the annotation
    ax.plot([x1, x1, x2, x2],
           [y, y + actual_height, y + actual_height, y],
           'k-', linewidth=1)

    # Add the text
    ax.text((x1 + x2) / 2, y + actual_height,
           text, ha='center', va='bottom',
           fontsize=fontsize)


def error_bar_format(data: np.ndarray,
                    ci: float = 95) -> Tuple[float, float]:
    """
    Calculate confidence intervals for error bars.

    Parameters
    ----------
    data : np.ndarray
        Data array
    ci : float, optional
        Confidence interval percentage (default: 95)

    Returns
    -------
    tuple
        (mean, error_margin)
    """
    if data.size == 0:
        return 0.0, 0.0

    mean = np.nanmean(data)

    if ci == 95:
        # 95% CI = 1.96 * SEM
        sem = np.nanstd(data, ddof=1) / np.sqrt(np.sum(~np.isnan(data)))
        error = 1.96 * sem
    else:
        # For other CI values, use percentile method
        alpha = 100 - ci
        lower = np.percentile(data, alpha / 2)
        upper = np.percentile(data, 100 - alpha / 2)
        error = max(mean - lower, upper - mean)

    return float(mean), float(error)
