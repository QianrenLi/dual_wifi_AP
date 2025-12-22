from typing import Any, Dict, Iterable, Optional, Callable, Union, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from exp_trace.util import ExpTraceReader
from exp_trace.plot_config import PlotTheme
from exp_trace.plot_utils import create_figure, save_figure, apply_scientific_style

# ------------------------------- Plotting ------------------------------- #
def plot_belief_comparison(meta_folder_data: List[Tuple[str, Dict[int, List[float]], str]],
                           title: str,
                           out_path: Path,
                           showfliers: bool = False,
                           figsize: str = "medium"):
    """
    Create a comparison plot of belief distributions from two meta-folders.

    Parameters
    ----------
    meta_folder_data : list of tuples
        Each tuple contains (meta_folder_name, il_to_beliefs, style)
        where style is "solid" or "dashed"
    title : str
        Plot title
    out_path : Path
        Output file path
    showfliers : bool, optional
        Whether to show outliers (default: False)
    figsize : str, optional
        Figure size name (default: "medium")
    """
    if not meta_folder_data or len(meta_folder_data) != 2:
        print("[error] Need exactly two meta-folders for comparison")
        return

    # Get all unique IL indices
    all_ils = set()
    for _, il_to_beliefs, _ in meta_folder_data:
        all_ils.update(il_to_beliefs.keys())
    all_ils = sorted(all_ils)

    if not all_ils:
        print("[info] Nothing to plot.")
        return

    # Create figure with scientific styling
    fig, ax = create_figure(size=figsize)

    # Plot each meta-folder's data
    for meta_idx, (meta_name, il_to_beliefs, line_style) in enumerate(meta_folder_data):
        # Prepare data for this meta-folder
        data = []
        positions = []
        il_colors = []  # Store color for each IL

        for i, il_id in enumerate(all_ils):
            beliefs = il_to_beliefs.get(il_id, [0.0])
            data.append(beliefs)

            # Position: first meta-folder at exact IL indices, second at +0.5
            if meta_idx == 0:
                pos = float(il_id)  # 0, 1, 2, ... (exact IL index)
            else:
                pos = float(il_id + 0.5)  # 0.5, 1.5, 2.5, ... (shifted)
            positions.append(pos)

            # Assign unique color per IL index (cycle through PlotTheme colors)
            color = PlotTheme.get_color(il_id % 10)  # Use IL index to determine color
            il_colors.append(color)

        # Create box plot for this meta-folder
        edge_style = '-' if line_style == "solid" else '--'

        bp = ax.boxplot(data,
                       positions=positions,
                       widths=0.35,  # Narrower to accommodate two boxes per IL
                       patch_artist=True,
                       showfliers=showfliers,
                    #    boxprops=dict(facecolor='gray', alpha=0.6,  # Will be overridden per box
                                #    edgecolor='black', linestyle=edge_style, linewidth=1.2),
                       medianprops=dict(color='black', linewidth=1.2, linestyle=edge_style),
                       whiskerprops=dict(color='black', linewidth=1.2, linestyle=edge_style),
                       capprops=dict(color='black', linewidth=1.2, linestyle=edge_style))

        # Color each box individually based on IL index
        for patch, color in zip(bp['boxes'], il_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black')  # Set edge color explicitly to black
            patch.set_linewidth(1.2)      # Set line width
            patch.set_linestyle(edge_style)  # Set line style (dashed or solid)

    # Set x-axis labels at the exact IL positions (first meta-folder positions)
    ax.set_xticks(all_ils)
    ax.set_xticklabels([str(il_id) for il_id in all_ils])

    # Create legend
    legend_elements = []
    for i, (meta_name, _, line_style) in enumerate(meta_folder_data):
        from matplotlib.patches import Patch
        legend_elements.append(
            Patch(facecolor='white', alpha=0.6,
                 edgecolor='black',
                 linestyle='-' if line_style == "solid" else '--',
                 label=meta_name)
        )
    ax.legend(handles=legend_elements, loc='best')

    # Apply scientific styling
    apply_scientific_style(
        ax,
        xlabel="Interference Traffics (per 100 Mbps)",
        ylabel="Belief",
        title=None,
        minor_ticks=True
    )

    # Adjust tick sizes
    ax.tick_params(axis="x", labelsize=PlotTheme.FONT_SIZE_MEDIUM)
    ax.tick_params(axis="y", labelsize=PlotTheme.FONT_SIZE_MEDIUM)

    # Adjust x-axis limits to show all boxes clearly
    if len(all_ils) > 0:
        min_il = min(all_ils)
        max_il = max(all_ils)
        # Add margin: 0.5 on left, 1.0 on right (to accommodate +0.5 shift)
        ax.set_xlim(min_il - 0.5, max_il + 1.0)

    # Save figure
    save_figure(fig, out_path, dpi=PlotTheme.DPI_PUBLICATION)
    print(f"[ok] Saved: {out_path}")

# ----------------------------------- CLI ----------------------------------- #
def main():
    """
    Compare belief distributions from two meta-folders.
    The second meta-folder's boxes are shifted 0.5 to the right and have dashed edges.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Compare belief distributions from two meta-folders.")
    ap.add_argument("--meta-folder-1", required=True,
                    help="First meta-folder to scan.")
    ap.add_argument("--meta-folder-2", required=True,
                    help="Second meta-folder to scan.")
    ap.add_argument("--name-1", default="Meta-folder 1",
                    help="Name for first meta-folder in legend (default: Meta-folder 1)")
    ap.add_argument("--name-2", default="Meta-folder 2",
                    help="Name for second meta-folder in legend (default: Meta-folder 2)")
    ap.add_argument("--pattern", default="IL_*_trial_*",
                    help="Glob pattern for run folders (default: IL_*_trial_*)")
    ap.add_argument("--out-dir", default="plots",
                    help="Directory to save the figure (default: plots)")
    ap.add_argument("--filename", default="belief_comparison.png",
                    help="Output filename for the comparison plot")
    ap.add_argument("--by-name", action="store_true",
                    help="Pick latest by lexicographic name instead of modification time")
    ap.add_argument("--showfliers", action="store_true",
                    help="Show outliers in the box plot")
    ap.add_argument("--last", nargs="+", type=int, default=[0],
                    help="Indices of last runs per IL (e.g., 0 1 for last and second last).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_folder_data = []

    # Process first meta-folder
    print(f"[info] Processing meta-folder 1: {args.meta_folder_1}")
    data_reader_1 = ExpTraceReader(args.meta_folder_1)
    il_to_beliefs_1 = {}
    seen_keys_1 = set()

    for k in args.last:
        per_il = data_reader_1.pick_last_k_per_il(k)
        for il_id, run_folder in sorted(per_il.items()):
            key = (il_id, str(run_folder))
            if key in seen_keys_1:
                continue
            seen_keys_1.add(key)

            rollout = data_reader_1.get_data(run_folder)[0]
            if rollout is None:
                print(f"[warn] no rollout.jsonl in {run_folder}")
                continue
            beliefs = rollout.belief
            if not beliefs:
                print(f"[warn] parsed 0 belief entries from {rollout}")
                continue
            il_to_beliefs_1.setdefault(il_id, []).extend(beliefs)

    if il_to_beliefs_1:
        meta_folder_data.append((args.name_1, il_to_beliefs_1, "solid"))
    else:
        print(f"[warn] No beliefs found in meta-folder 1: {args.meta_folder_1}")

    # Process second meta-folder
    print(f"[info] Processing meta-folder 2: {args.meta_folder_2}")
    data_reader_2 = ExpTraceReader(args.meta_folder_2)
    il_to_beliefs_2 = {}
    seen_keys_2 = set()

    for k in args.last:
        per_il = data_reader_2.pick_last_k_per_il(k)
        for il_id, run_folder in sorted(per_il.items()):
            key = (il_id, str(run_folder))
            if key in seen_keys_2:
                continue
            seen_keys_2.add(key)

            rollout = data_reader_2.get_data(run_folder)[0]
            if rollout is None:
                print(f"[warn] no rollout.jsonl in {run_folder}")
                continue
            beliefs = rollout.belief
            if not beliefs:
                print(f"[warn] parsed 0 belief entries from {rollout}")
                continue
            il_to_beliefs_2.setdefault(il_id, []).extend(beliefs)

    if il_to_beliefs_2:
        meta_folder_data.append((args.name_2, il_to_beliefs_2, "dashed"))
    else:
        print(f"[warn] No beliefs found in meta-folder 2: {args.meta_folder_2}")

    # Create comparison plot
    if meta_folder_data and len(meta_folder_data) == 2:
        out_path = out_dir / args.filename
        title = f"Belief Distribution Comparison: {args.name_1} vs {args.name_2}"
        plot_belief_comparison(meta_folder_data, title=title, out_path=out_path,
                             showfliers=args.showfliers, figsize="large")
    else:
        print("[error] Could not create comparison plot - need belief data from both meta-folders")

if __name__ == "__main__":
    main()