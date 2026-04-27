import os
import random
from pathlib import Path

import numpy as np
import torch

from src.config import ProjectConfig


def setGlobalSeed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def getProjectDirs(projectConfig: ProjectConfig) -> list[Path]:
    return [
        projectConfig.dataRawDir,
        projectConfig.wtDatasetDir,
        projectConfig.imageDir,
        projectConfig.annotationDir,
        projectConfig.secondAnnotatorDir,
        projectConfig.dataProcessedDir,
        projectConfig.auditDir,
        projectConfig.cropsDir,
        projectConfig.manifestsDir,
        projectConfig.outputsDir,
        projectConfig.figuresDir,
        projectConfig.metricsDir,
        projectConfig.samplePredictionsDir,
        projectConfig.tablesDir,
        projectConfig.reportAssetsDir,
        projectConfig.reportFiguresDir,
        projectConfig.reportTablesDir,
        projectConfig.notebooksDir,
    ]


def ensureProjectDirs(projectConfig: ProjectConfig) -> None:
    for currentDir in getProjectDirs(projectConfig):
        currentDir.mkdir(parents=True, exist_ok=True)


def printProjectSummary(projectConfig: ProjectConfig) -> None:
    print("projectRoot:", projectConfig.projectRoot)
    print("seed:", projectConfig.seed)
    print("device:", projectConfig.device)
    print("imageDir:", projectConfig.imageDir)
    print("annotationDir:", projectConfig.annotationDir)
    print("splitFilePath:", projectConfig.splitFilePath)
    print("classFilePath:", projectConfig.classFilePath)
    print("cropsDir:", projectConfig.cropsDir)
    print("metricsDir:", projectConfig.metricsDir)