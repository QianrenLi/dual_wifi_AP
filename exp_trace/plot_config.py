"""
Centralized configuration for scientific plotting.

This module provides consistent styling and configuration for all plots
in the exp_trace package, ensuring a professional and scientific appearance.
"""

from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path


class PlotTheme:
    """
    A theme class for scientific plot styling.

    Provides consistent colors, fonts, sizes, and other styling parameters
    for creating publication-quality plots.
    """

    # Font configuration
    FONT_FAMILY = "sans-serif"
    FONT_SANS_SERIF = ["Helvetica", "Arial", "DejaVu Sans"]
    MATHTEXT_FONTSET = "stix"
    MATHEMATICS_FONT = "STIXGeneral"

    # Font sizes
    FONT_SIZE_SMALL = 10
    FONT_SIZE_MEDIUM = 12
    FONT_SIZE_LARGE = 14
    FONT_SIZE_XLARGE = 16
    FONT_SIZE_XXLARGE = 18

    # Figure sizes (width, height) in inches
    FIGURE_SIZES = {
        "small": (6, 4),
        "medium": (8, 5),
        "large": (10, 6),
        "wide": (12, 6),
        "square": (8, 8),
        "portrait": (7, 10),
        "portrait_small": (3.5, 5),
    }

    # DPI settings
    DPI_DRAFT = 120
    DPI_PRESENTATION = 150
    DPI_PUBLICATION = 300

    # Color palette (colorblind-friendly)
    # Indexed access: primary0, primary1, ..., primary9
    COLORS = {
        "primary0": "#1f77b4",     # Blue
        "primary1": "#ff7f0e",     # Orange
        "primary2": "#2ca02c",     # Green
        "primary3": "#d62728",     # Red
        "primary4": "#9467bd",     # Purple
        "primary5": "#8c564b",     # Brown
        "primary6": "#e377c2",     # Pink
        "primary7": "#7f7f7f",     # Gray
        "primary8": "#bcbd22",     # Olive
        "primary9": "#17becf",     # Cyan
    }

    # Line styles
    LINE_STYLES = {
        "solid": "-",
        "dashed": "--",
        "dotted": ":",
        "dash_dot": "-.",
    }

    # Marker styles
    MARKERS = {
        "circle": "o",
        "square": "s",
        "triangle": "^",
        "diamond": "D",
        "plus": "+",
        "cross": "x",
        "star": "*",
        "pentagon": "p",
    }

    # Grid settings
    GRID_LINESTYLE = "--"
    GRID_ALPHA = 0.4
    GRID_MINOR_ALPHA = 0.25

    # Legend settings
    LEGEND_FONT_SIZE = 12
    LEGEND_FRAME_ALPHA = 0.9

    # Tick settings
    TICK_DIRECTION = "in"
    TICK_LENGTH_MAJOR = 4
    TICK_LENGTH_MINOR = 2

    # Spine settings
    SPINE_LINEWIDTH = 0.8
    TOP_RIGHT_SPINE_VISIBLE = False  # Scientific style: hide top and right spines

    @classmethod
    def configure_matplotlib(cls, style: str = "publication") -> None:
        """
        Configure matplotlib with the scientific theme.

        Parameters
        ----------
        style : str, optional
            Plot style variant. Options:
            - "draft": Lower DPI, faster rendering
            - "presentation": Medium DPI, balanced quality
            - "publication": High DPI, best quality (default)
        """
        # Font configuration
        plt.rcParams.update({
            "font.family": cls.FONT_FAMILY,
            "font.sans-serif": cls.FONT_SANS_SERIF,
            "font.size": cls.FONT_SIZE_MEDIUM,
            "axes.labelsize": cls.FONT_SIZE_MEDIUM,
            "axes.titlesize": cls.FONT_SIZE_XLARGE,
            "xtick.labelsize": cls.FONT_SIZE_SMALL,
            "ytick.labelsize": cls.FONT_SIZE_SMALL,
            "legend.fontsize": cls.LEGEND_FONT_SIZE,
            "figure.titlesize": cls.FONT_SIZE_XXLARGE,
        })

        # Math text configuration
        matplotlib.rcParams['mathtext.fontset'] = cls.MATHTEXT_FONTSET
        matplotlib.rcParams['font.family'] = cls.MATHEMATICS_FONT

        # DPI based on style
        if style == "draft":
            dpi = cls.DPI_DRAFT
        elif style == "presentation":
            dpi = cls.DPI_PRESENTATION
        else:  # publication
            dpi = cls.DPI_PUBLICATION

        plt.rcParams.update({
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        })

        # Tick configuration
        plt.rcParams.update({
            "xtick.direction": cls.TICK_DIRECTION,
            "ytick.direction": cls.TICK_DIRECTION,
            "xtick.major.size": cls.TICK_LENGTH_MAJOR,
            "ytick.major.size": cls.TICK_LENGTH_MAJOR,
            "xtick.minor.size": cls.TICK_LENGTH_MINOR,
            "ytick.minor.size": cls.TICK_LENGTH_MINOR,
        })

        # Spine configuration
        plt.rcParams.update({
            "axes.linewidth": cls.SPINE_LINEWIDTH,
            "axes.spines.top": cls.TOP_RIGHT_SPINE_VISIBLE,
            "axes.spines.right": cls.TOP_RIGHT_SPINE_VISIBLE,
        })

    @classmethod
    def get_color(cls, index: int) -> str:
        """
        Get a color from the palette by index.

        Parameters
        ----------
        index : int
            Color index (will wrap around if out of range)

        Returns
        -------
        str
            Hex color code
        """
        color_keys = list(cls.COLORS.keys())
        key = color_keys[index % len(color_keys)]
        return cls.COLORS[key]

    @classmethod
    def get_figure_size(cls, size_name: str,
                       nrows: int = 1,
                       ncols: int = 1) -> Tuple[float, float]:
        """
        Get figure size for multiple subplots.

        Parameters
        ----------
        size_name : str
            Base size name from FIGURE_SIZES
        nrows : int, optional
            Number of subplot rows
        ncols : int, optional
            Number of subplot columns

        Returns
        -------
        tuple
            (width, height) in inches
        """
        base_width, base_height = cls.FIGURE_SIZES.get(size_name, cls.FIGURE_SIZES["medium"])

        # Scale height based on number of rows
        # Add some padding between subplots
        height_padding = 0.5 * (nrows - 1)
        total_height = base_height * nrows + height_padding

        # Scale width based on number of columns
        # Add some padding between subplots
        width_padding = 0.5 * (ncols - 1)
        total_width = base_width * ncols + width_padding

        return (total_width, total_height)

    @classmethod
    def apply_grid_style(cls, ax, major: bool = True,
                        linestyle: Optional[str] = None,
                        alpha: Optional[float] = None) -> None:
        """
        Apply consistent grid styling to an axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to style
        major : bool, optional
            Whether to style major grid (default: True)
        linestyle : str, optional
            Line style for grid (default: from theme)
        alpha : float, optional
            Transparency for grid (default: from theme)
        """
        linestyle = linestyle or cls.GRID_LINESTYLE
        alpha = alpha or (cls.GRID_ALPHA if major else cls.GRID_MINOR_ALPHA)

        if major:
            ax.grid(True, axis="both", which="major",
                   linestyle=linestyle, alpha=alpha, zorder=0)
        else:
            ax.grid(True, axis="both", which="minor",
                   linestyle=":", alpha=alpha, zorder=0)


# Pre-defined themes for common use cases
THEMES = {
    "default": PlotTheme,
    "publication": PlotTheme,
    "colorblind": PlotTheme,  # Already uses colorblind-friendly palette
}


def configure_theme(theme_name: str = "default",
                  style: str = "publication") -> None:
    """
    Configure a plotting theme.

    Parameters
    ----------
    theme_name : str, optional
        Name of the theme to use (default: "default")
    style : str, optional
        Style variant ("draft", "presentation", "publication")
    """
    theme_class = THEMES.get(theme_name, PlotTheme)
    theme_class.configure_matplotlib(style)


# Auto-configure the default theme when module is imported
configure_theme("default", "publication")
