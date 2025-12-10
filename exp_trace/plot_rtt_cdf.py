#!/usr/bin/env python3
"""
Plot RTT Cumulative Distribution Functions from log files.

Usage:
  python plot_rtt_cdf.py file1.txt file2.txt ... --output rtt_cdf.png
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

from exp_trace.plot_config import PlotTheme
from exp_trace.plot_utils import create_figure, save_figure, apply_scientific_style


def read_rtts(file_path: Path) -> np.ndarray:
    """
    Read RTT values (2nd column) from a text file.

    Parameters
    ----------
    file_path : Path
        Path to RTT log file

    Returns
    -------
    np.ndarray
        Array of RTT values in seconds
    """
    rtts = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    rtts.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(rtts)


def plot_cdf(ax: plt.Axes, rtts: np.ndarray, label: str,
             color: Optional[str] = None) -> None:
    """
    Plot CDF for a given RTT array.

    Parameters
    ----------
    ax : plt.Axes
        Axis to plot on
    rtts : np.ndarray
        RTT values in seconds
    label : str
        Legend label
    color : str, optional
        Plot line color (default: from theme)
    """
    if rtts.size == 0:
        return

    rtts_sorted = np.sort(rtts)
    cdf = np.arange(1, len(rtts_sorted) + 1) / len(rtts_sorted)

    # Convert to milliseconds for display
    rtts_ms = rtts_sorted * 1000

    # Use theme color if none provided
    if color is None:
        # Get next color from cycler
        color = ax._get_lines.get_next_color()

    ax.plot(rtts_ms, cdf, label=label, color=color, linewidth=2)


def plot_rtt_cdfs(log_files: List[Path],
                  output_path: Optional[Path] = None,
                  title: str = "RTT Cumulative Distribution",
                  figsize: str = "medium") -> None:
    """
    Plot RTT CDFs from multiple log files.

    Parameters
    ----------
    log_files : list of Path
        List of RTT log file paths
    output_path : Path, optional
        Output file path (default: None = show plot)
    title : str, optional
        Plot title (default: "RTT Cumulative Distribution")
    figsize : str, optional
        Figure size name (default: "medium")
    """
    fig, ax = create_figure(size=figsize)

    colors = [PlotTheme.get_color(i) for i in range(len(log_files))]

    for i, log_file in enumerate(log_files):
        rtts = read_rtts(log_file)
        if rtts.size == 0:
            print(f"[WARN] No valid RTTs found in {log_file}")
            continue

        # Use parent directory name as label
        label = log_file.parent.stem
        plot_cdf(ax, rtts, label=label, color=colors[i])

    # Apply scientific styling
    apply_scientific_style(
        ax,
        xlabel="RTT (ms)",
        ylabel="CDF",
        title=title,
        minor_ticks=True
    )

    # Add legend
    ax.legend(title="Log Files", fontsize=PlotTheme.FONT_SIZE_SMALL)

    if output_path:
        save_figure(fig, output_path, dpi=PlotTheme.DPI_PUBLICATION)
        print(f"[ok] Saved: {output_path}")
    else:
        plt.show()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot RTT CDFs from log files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_rtt_cdf.py file1.txt file2.txt
  python plot_rtt_cdf.py file1.txt --output rtt_cdf.png
  python plot_rtt_cdf.py *.txt --output rtt_cdf.pdf --title "My RTT Analysis"
        """)
    parser.add_argument("log_files", nargs="+", help="One or more RTT log text files")
    parser.add_argument("-o", "--output", type=Path, help="Output file path (e.g., rtt_cdf.png)")
    parser.add_argument("-t", "--title", default="RTT Cumulative Distribution",
                       help="Plot title (default: 'RTT Cumulative Distribution')")
    parser.add_argument("-s", "--size", default="medium", choices=["small", "medium", "large", "wide"],
                       help="Figure size (default: 'medium')")

    args = parser.parse_args()

    # Convert to Path objects
    log_files = [Path(f) for f in args.log_files]

    # Plot
    plot_rtt_cdfs(log_files, args.output, args.title, args.size)


if __name__ == "__main__":
    main()
