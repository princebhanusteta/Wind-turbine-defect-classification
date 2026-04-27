"""We expose the shared project utility functions."""

from .plotUtils import addLegend, applyPlotStyle, buildFigurePath, ensureFigureDir, rotateCategoryLabels, saveFigure, styleAxes
from .projectSetup import ensureProjectDirs, getProjectDirs, printProjectSummary, setGlobalSeed

__all__ = [
    "setGlobalSeed",
    "getProjectDirs",
    "ensureProjectDirs",
    "printProjectSummary",
    "applyPlotStyle",
    "styleAxes",
    "addLegend",
    "rotateCategoryLabels",
    "ensureFigureDir",
    "buildFigurePath",
    "saveFigure",
]