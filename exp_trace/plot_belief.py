from typing import Any, Dict, Iterable, Optional, Callable, Union, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt

from exp_trace.util import ExpTraceReader
from exp_trace.plot_config import PlotTheme
from exp_trace.plot_utils import create_figure, save_figure, apply_scientific_style, create_box_plot_with_stats

# ------------------------------- Plotting ------------------------------- #
def plot_belief_box_per_il(il_to_beliefs: Dict[int, List[float]],
                           title: str,
                           out_path: Path,
                           showfliers: bool = False,
                           figsize: str = "medium"):
    """
    Make ONE box figure: each box is the belief distribution for one IL (latest run).
    X tick labels are IL indices.

    Parameters
    ----------
    il_to_beliefs : dict
        Dictionary mapping IL IDs to belief value lists
    title : str
        Plot title
    out_path : Path
        Output file path
    showfliers : bool, optional
        Whether to show outliers (default: False)
    figsize : str, optional
        Figure size name (default: "medium")
    """
    if not il_to_beliefs:
        print("[info] Nothing to plot.")
        return

    ils = sorted(il_to_beliefs.keys())
    data = [il_to_beliefs[i] if il_to_beliefs[i] else [0.0] for i in ils]

    # Create figure with scientific styling
    fig, ax = create_figure(size=figsize)

    # Create styled box plot
    create_box_plot_with_stats(
        ax, data, labels=[str(i) for i in ils],
        showfliers=showfliers,
        patch_artist=True
    )

    # Apply scientific styling
    apply_scientific_style(
        ax,
        ylabel="Belief",
        title=title,
        minor_ticks=False
    )

    # Adjust tick sizes
    ax.tick_params(axis="x", labelsize=PlotTheme.FONT_SIZE_MEDIUM)
    ax.tick_params(axis="y", labelsize=PlotTheme.FONT_SIZE_MEDIUM)

    # Save figure
    save_figure(fig, out_path, dpi=PlotTheme.DPI_PUBLICATION)
    print(f"[ok] Saved: {out_path}")

# ----------------------------------- CLI ----------------------------------- #
def main():
    """
    For each --meta-folder:
      - pick the latest run per IL
      - load belief sequences
    Merge across all provided meta-folders by IL id, then
    plot ONE box figure where each box corresponds to one IL index.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Plot belief box of latest run per IL (x-label = IL index).")
    ap.add_argument("--meta-folder", required=True,
                    help="One or more meta-folders to scan.")
    ap.add_argument("--pattern", default="IL_*_trial_*",
                    help="Glob pattern for run folders (default: IL_*_trial_*)")
    ap.add_argument("--out-dir", default="plots",
                    help="Directory to save the figure (default: plots)")
    ap.add_argument("--filename", default="belief_box_latest_per_IL.png",
                    help="Output filename for the box plot")
    ap.add_argument("--by-name", action="store_true",
                    help="Pick latest by lexicographic name instead of modification time")
    ap.add_argument("--showfliers", action="store_true",
                    help="Show outliers in the box plot")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    data_reader = ExpTraceReader(args.meta_folder)

    il_to_beliefs = {}

    for il_id, run_folder in data_reader.latest_folders.items():
        rollout = data_reader.get_data(run_folder)[0]
        if rollout is None:
            print(f"[warn] no rollout.jsonl in {run_folder}")
            continue
        beliefs = rollout.belief
        if not beliefs:
            print(f"[warn] parsed 0 belief entries from {rollout}")
            continue
        il_to_beliefs.setdefault(il_id, []).extend(beliefs)

    # Plot one box figure: x = IL index
    if il_to_beliefs:
        out_path = out_dir / args.filename
        title = "Belief distribution — latest run per IL"
        plot_belief_box_per_il(il_to_beliefs, title=title, out_path=out_path, showfliers=args.showfliers)
    else:
        print("[info] No beliefs found; nothing to plot.")

if __name__ == "__main__":
    main()
