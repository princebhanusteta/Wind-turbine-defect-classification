from __future__ import annotations

"""We provide a small shared plotting utility layer for clean and consistent project figures."""

from pathlib import Path

import matplotlib.pyplot as plt


def applyPlotStyle() -> None:
    """We apply one consistent project-wide Matplotlib style for clean and spacious figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (8.5, 5.5),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.titlepad": 12,
            "axes.labelpad": 8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10.5,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.borderpad": 0.6,
            "legend.labelspacing": 0.5,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.6,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def styleAxes(
    ax: plt.Axes,
    title: str,
    xLabel: str,
    yLabel: str,
) -> None:
    """We apply a clean and readable axis style to one plot."""
    ax.set_title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.margins(x=0.02, y=0.06)


def addLegend(
    ax: plt.Axes,
    title: str | None = None,
    location: str = "best",
    ncol: int = 1,
) -> None:
    """We add a readable legend only when labeled artists are present on the axis."""
    handles, labels = ax.get_legend_handles_labels()

    if not handles:
        return

    ax.legend(
        title=title,
        loc=location,
        ncol=ncol,
    )


def rotateCategoryLabels(
    ax: plt.Axes,
    rotation: int = 25,
    horizontalAlignment: str = "right",
) -> None:
    """We rotate category tick labels when needed so they stay readable and spacious."""
    for currentLabel in ax.get_xticklabels():
        currentLabel.set_rotation(rotation)
        currentLabel.set_ha(horizontalAlignment)


def ensureFigureDir(outputDir: Path) -> None:
    """We create the target figure directory before saving figures."""
    outputDir.mkdir(parents=True, exist_ok=True)


def buildFigurePath(outputDir: Path, figureFileName: str) -> Path:
    """We build a clean figure path and enforce a PNG extension for saved plots."""
    fileName = figureFileName if figureFileName.endswith(".png") else f"{figureFileName}.png"
    return outputDir / fileName


def saveFigure(
    fig: plt.Figure,
    outputDir: Path,
    figureFileName: str,
    closeFigure: bool = False,
) -> Path:
    """We save one figure to disk without redrawing it in a second plotting cell."""
    ensureFigureDir(outputDir)
    figurePath = buildFigurePath(outputDir=outputDir, figureFileName=figureFileName)

    fig.savefig(figurePath)

    if closeFigure:
        plt.close(fig)

    return figurePath