"""Matplotlib styling and save helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


OLLAMA_COLOR = "#ff6b6b"
VLLM_COLOR = "#4ecdc4"

DPI_SCREEN = 100
DPI_PUBLISH = 300


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": DPI_SCREEN,
            "savefig.dpi": DPI_PUBLISH,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
            "errorbar.capsize": 3,
        }
    )


def log_axis_y(ax):
    ax.set_yscale("log")


def save_figure(fig: plt.Figure, path_png: Path, include_pdf: bool, path_pdf: Path | None):
    path_png = Path(path_png)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=DPI_PUBLISH, bbox_inches="tight", facecolor="white")
    if include_pdf and path_pdf is not None:
        path_pdf = Path(path_pdf)
        path_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_pdf, format="pdf", bbox_inches="tight", facecolor="white")
