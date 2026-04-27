from __future__ import annotations

"""We build the strict modeling manifest used for leakage-aware training and evaluation."""

import json
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig


def loadCropManifest(manifestPath: Path) -> pd.DataFrame:
    """We load the crop manifest that was created during crop extraction."""
    cropManifestDf = pd.read_csv(manifestPath)
    return cropManifestDf


def splitIncludedAndExcludedCrops(cropManifestDf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """We separate the strict modeling crops from crops excluded due to exact-duplicate risk."""
    excludedDuplicateDf = cropManifestDf.loc[
        cropManifestDf["isExactDuplicateImage"] == 1
    ].copy()

    strictModelingDf = cropManifestDf.loc[
        cropManifestDf["isExactDuplicateImage"] == 0
    ].copy()

    strictModelingDf = strictModelingDf.reset_index(drop=True)
    excludedDuplicateDf = excludedDuplicateDf.reset_index(drop=True)

    return strictModelingDf, excludedDuplicateDf


def buildModelingClassCounts(strictModelingDf: pd.DataFrame, classNames: list[str]) -> pd.DataFrame:
    """We build class-level crop counts for the strict modeling manifest."""
    classCountsDf = (
        strictModelingDf.groupby("className")
        .size()
        .rename("cropCount")
        .reindex(classNames, fill_value=0)
        .reset_index()
        .rename(columns={"index": "className"})
    )

    return classCountsDf


def buildModelingSplitCounts(strictModelingDf: pd.DataFrame) -> pd.DataFrame:
    """We build split-level crop counts for the strict modeling manifest."""
    splitCountsDf = (
        strictModelingDf.groupby("subset")
        .size()
        .rename("cropCount")
        .reindex(["train", "val", "test"], fill_value=0)
        .reset_index()
        .rename(columns={"index": "subset"})
    )

    return splitCountsDf


def buildModelingClassBySplitCounts(
    strictModelingDf: pd.DataFrame,
    classNames: list[str],
) -> pd.DataFrame:
    """We build class-by-split counts for the strict modeling manifest."""
    classBySplitCountsDf = (
        strictModelingDf.groupby(["subset", "className"])
        .size()
        .rename("cropCount")
        .reset_index()
    )

    allSubsetClassPairs = pd.MultiIndex.from_product(
        [["train", "val", "test"], classNames],
        names=["subset", "className"],
    )

    classBySplitCountsDf = (
        classBySplitCountsDf.set_index(["subset", "className"])
        .reindex(allSubsetClassPairs, fill_value=0)
        .reset_index()
    )

    return classBySplitCountsDf


def buildModelingSummary(
    cropManifestDf: pd.DataFrame,
    strictModelingDf: pd.DataFrame,
    excludedDuplicateDf: pd.DataFrame,
) -> dict:
    """We summarize what was kept and what was excluded for strict modeling."""
    return {
        "rawCropCount": int(len(cropManifestDf)),
        "strictModelingCropCount": int(len(strictModelingDf)),
        "excludedDuplicateCropCount": int(len(excludedDuplicateDf)),
        "excludedCrossSplitDuplicateCropCount": int(excludedDuplicateDf["isCrossSplitDuplicateImage"].sum()),
        "strictUniqueSourceImageCount": int(strictModelingDf["sourceImageFileName"].nunique()),
        "strictTrainCropCount": int((strictModelingDf["subset"] == "train").sum()),
        "strictValCropCount": int((strictModelingDf["subset"] == "val").sum()),
        "strictTestCropCount": int((strictModelingDf["subset"] == "test").sum()),
    }


def saveModelingArtifacts(
    strictModelingDf: pd.DataFrame,
    excludedDuplicateDf: pd.DataFrame,
    classCountsDf: pd.DataFrame,
    splitCountsDf: pd.DataFrame,
    classBySplitCountsDf: pd.DataFrame,
    modelingSummary: dict,
    manifestsDir: Path,
) -> dict[str, str]:
    """We save the strict modeling manifest and its supporting summary artifacts."""
    manifestsDir.mkdir(parents=True, exist_ok=True)

    strictManifestPath = manifestsDir / "modelingManifestStrict.csv"
    excludedDuplicatePath = manifestsDir / "excludedDuplicateCrops.csv"
    classCountsPath = manifestsDir / "modelingClassCountsStrict.csv"
    splitCountsPath = manifestsDir / "modelingSplitCountsStrict.csv"
    classBySplitCountsPath = manifestsDir / "modelingClassBySplitCountsStrict.csv"
    summaryPath = manifestsDir / "modelingManifestSummaryStrict.json"

    strictModelingDf.to_csv(strictManifestPath, index=False)
    excludedDuplicateDf.to_csv(excludedDuplicatePath, index=False)
    classCountsDf.to_csv(classCountsPath, index=False)
    splitCountsDf.to_csv(splitCountsPath, index=False)
    classBySplitCountsDf.to_csv(classBySplitCountsPath, index=False)

    with open(summaryPath, "w", encoding="utf-8") as jsonFile:
        json.dump(modelingSummary, jsonFile, indent=2)

    return {
        "modelingManifestStrict": str(strictManifestPath),
        "excludedDuplicateCrops": str(excludedDuplicatePath),
        "modelingClassCountsStrict": str(classCountsPath),
        "modelingSplitCountsStrict": str(splitCountsPath),
        "modelingClassBySplitCountsStrict": str(classBySplitCountsPath),
        "modelingManifestSummaryStrict": str(summaryPath),
    }


def runModelingManifestBuilder(projectConfig: ProjectConfig) -> dict:
    """We create the strict modeling manifest by excluding exact-duplicate-risk crops before training."""
    cropManifestPath = projectConfig.manifestsDir / "cropManifest.csv"
    cropManifestDf = loadCropManifest(cropManifestPath)

    strictModelingDf, excludedDuplicateDf = splitIncludedAndExcludedCrops(cropManifestDf)

    classCountsDf = buildModelingClassCounts(
        strictModelingDf=strictModelingDf,
        classNames=projectConfig.classNames,
    )

    splitCountsDf = buildModelingSplitCounts(strictModelingDf)

    classBySplitCountsDf = buildModelingClassBySplitCounts(
        strictModelingDf=strictModelingDf,
        classNames=projectConfig.classNames,
    )

    modelingSummary = buildModelingSummary(
        cropManifestDf=cropManifestDf,
        strictModelingDf=strictModelingDf,
        excludedDuplicateDf=excludedDuplicateDf,
    )

    savedPathDict = saveModelingArtifacts(
        strictModelingDf=strictModelingDf,
        excludedDuplicateDf=excludedDuplicateDf,
        classCountsDf=classCountsDf,
        splitCountsDf=splitCountsDf,
        classBySplitCountsDf=classBySplitCountsDf,
        modelingSummary=modelingSummary,
        manifestsDir=projectConfig.manifestsDir,
    )

    return {
        "strictModelingDf": strictModelingDf,
        "excludedDuplicateDf": excludedDuplicateDf,
        "classCountsDf": classCountsDf,
        "splitCountsDf": splitCountsDf,
        "classBySplitCountsDf": classBySplitCountsDf,
        "modelingSummary": modelingSummary,
        "savedPathDict": savedPathDict,
    }