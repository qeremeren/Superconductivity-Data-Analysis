"""Shared figure style: one palette and one set of mark specs for every phase.

Colors follow the entity across all figures (a family is always the same hue).
Palettes were checked with the dataviz skill's validator (light surface):
families use categorical slots 1-3 (pass all-pairs); split kinds use violet and
magenta (pass all-pairs; magenta is below 3:1 contrast, so it is always
direct-labeled and the data are in results/ as CSV/JSON).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import data  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

FAMILY_COLORS = {data.CUPRATE: "#2a78d6", data.IRON_BASED: "#eb6834", data.OTHER: "#1baf7a"}
FAMILY_LABELS = {data.CUPRATE: "Cuprates", data.IRON_BASED: "Iron-based", data.OTHER: "Other"}
SPLIT_COLORS = {"random": "#e87ba4", "grouped": "#4a3aa7"}

LINE_WIDTH = 2.0
MARKER_SIZE = 6  # points; >= 8 px diameter at the saved DPI
DPI = 200


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "lines.linewidth": LINE_WIDTH,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
