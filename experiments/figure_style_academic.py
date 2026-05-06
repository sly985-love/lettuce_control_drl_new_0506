# -*- coding: utf-8 -*-
"""Shared academic-style helpers for manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


COLORS = {
    "ink": "#22313f",
    "gray": "#6f7b87",
    "light_gray": "#e7edf2",
    "grid": "#d8e1e8",
    "navy": "#2f5f8f",
    "blue": "#4f86c6",
    "teal": "#5aa6a6",
    "green": "#78a55a",
    "gold": "#d4a64f",
    "brick": "#c86d5d",
    "plum": "#8f86c7",
    "sand": "#f4efe4",
    "olive": "#97a55f",
    "sky": "#6eb1d6",
    "rose": "#d88a7b",
}


def apply_academic_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Calibri", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "axes.unicode_minus": False,
            "font.size": 9.0,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.8,
            "ytick.major.size": 3.8,
            "xtick.minor.size": 2.2,
            "ytick.minor.size": 2.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "legend.frameon": False,
            "legend.labelcolor": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.65,
            "grid.alpha": 0.85,
            "lines.linewidth": 1.7,
            "patch.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax: plt.Axes, grid_axis: str = "y", add_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", colors=COLORS["ink"], pad=2.5)
    ax.tick_params(axis="both", which="minor", colors=COLORS["gray"])
    ax.set_axisbelow(True)
    if add_grid:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.6, alpha=0.85)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.03,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        fontweight="bold",
        color=COLORS["ink"],
    )


def apply_heatmap_frame(ax: plt.Axes, nrows: int, ncols: int) -> None:
    ax.set_xticks([x - 0.5 for x in range(1, ncols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, nrows)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)


def set_hour_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.set_xlim(0.0, 24.0)


def set_day_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=12, integer=True))


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
