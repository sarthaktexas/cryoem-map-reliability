"""Nature-journal matplotlib styling: rcParams, palettes, and figure helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Global rcParams (applied on import)
# ---------------------------------------------------------------------------

_NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "lines.linewidth": 0.75,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

mpl.rcParams.update(_NATURE_RC)

# macOS Helvetica/Arial often trigger benign fontTools table-parse warnings on save.
logging.getLogger("fontTools").setLevel(logging.ERROR)

# Muted Nature-style qualitative colors (blues, reds, greens, oranges — no neons).
_CATEGORICAL = [
    "#4E79A7",  # blue
    "#E15759",  # red
    "#59A14F",  # green
    "#F28E2B",  # orange
    "#76B7B2",  # teal
    "#EDC948",  # yellow-gold
    "#B07AA1",  # purple
    "#9C755F",  # brown
]

# RdBu-style diverging palette for heatmaps.
_DIVERGING = mpl.colormaps["RdBu_r"].copy()

# Yellow → dark red sequential (matches mean |coupling| colorbars; YlOrRd family).
_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "nature_sequential",
    ["#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C", "#FD8D3C", "#FC4E2A", "#E31A1C", "#B10026"],
)

PALETTES: dict[str, list[str] | Colormap] = {
    "categorical": _CATEGORICAL,
    "diverging": _DIVERGING,
    "sequential": _SEQUENTIAL,
}


def apply(ax: plt.Axes) -> None:
    """Strip top/right spines and set tick params to match Nature style."""
    if hasattr(ax, "spines"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=6,
            width=0.5,
            length=2,
        )
        ax.tick_params(axis="both", which="minor", width=0.5)
    else:
        # Axes3D: no spine API; tick styling only.
        ax.tick_params(labelsize=6, width=0.5, length=2)


def label_panel(ax: plt.Axes, letter: str, *, x: float = -0.1, y: float = 1.05) -> None:
    """Bold panel label (a, b, c…) at upper-left in Nature position."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def label_panel_3d(ax, letter: str) -> None:
    """Panel label for mplot3d axes (text2D in axes coordinates)."""
    ax.text2D(
        -0.08,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


WORD_PNG_DPI = 600


def _has_3d_axes(fig: plt.Figure) -> bool:
    """True when any subplot is an mplot3d Axes3D instance."""
    for ax in fig.get_axes():
        if getattr(ax, "name", None) == "3d":
            return True
        if not hasattr(ax, "spines"):
            return True
    return False


def savefig(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = WORD_PNG_DPI,
    **kwargs,
) -> None:
    """
    Export figures for thesis / publication.

    - **2D figures:** vector PDF + 600 dpi PNG (Word).
    - **3D figures (Axes3D):** 600 dpi PNG only (mplot3d PDF is unwieldy).

    ``dpi`` is accepted for API compatibility; PNG is always written at ``WORD_PNG_DPI``.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    save_kw = dict(bbox_inches="tight", facecolor="white", **kwargs)
    if not _has_3d_axes(fig):
        fig.savefig(stem.with_suffix(".pdf"), **save_kw)
    fig.savefig(stem.with_suffix(".png"), dpi=WORD_PNG_DPI, **save_kw)


__all__ = ["PALETTES", "WORD_PNG_DPI", "apply", "label_panel", "label_panel_3d", "savefig"]
