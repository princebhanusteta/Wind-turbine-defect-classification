from __future__ import annotations

"""We generate visual overlay checks so we can confirm that XML stem based matching is correct."""

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.config import ProjectConfig
from src.data.rawDataAudit import runRawDataAudit


def loadAuditTables(projectConfig: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """We load the annotation-level and image-level audit tables needed for overlay validation."""
    auditResult = runRawDataAudit(projectConfig)
    savedPathDict = auditResult["savedPathDict"]

    annotationDf = pd.read_csv(savedPathDict["annotationObjects"])
    imageLevelAuditDf = pd.read_csv(savedPathDict["imageLevelAudit"])

    return annotationDf, imageLevelAuditDf


def prepareOverlayOutputDir(outputDir: Path) -> None:
    """We create a clean overlay output directory so old overlay files do not remain mixed with new ones."""
    outputDir.mkdir(parents=True, exist_ok=True)

    for currentPath in outputDir.iterdir():
        if currentPath.is_file() and currentPath.suffix.lower() in {".png", ".csv"}:
            currentPath.unlink()


def selectOverlaySampleTable(
    imageLevelAuditDf: pd.DataFrame,
    maxMismatchSamples: int,
    maxRegularSamples: int,
    seed: int,
) -> pd.DataFrame:
    """We choose a reproducible mix of mismatch and non-mismatch images for visual inspection."""
    eligibleDf = imageLevelAuditDf.loc[imageLevelAuditDf["hasObjects"] == 1].copy()

    mismatchDf = eligibleDf.loc[eligibleDf["hasXmlFilenameMismatch"] == 1].copy()
    regularDf = eligibleDf.loc[eligibleDf["hasXmlFilenameMismatch"] == 0].copy()

    mismatchSampleCount = min(maxMismatchSamples, len(mismatchDf))
    regularSampleCount = min(maxRegularSamples, len(regularDf))

    mismatchSampleDf = (
        mismatchDf.sample(n=mismatchSampleCount, random_state=seed)
        if mismatchSampleCount > 0
        else mismatchDf.head(0).copy()
    )

    regularSampleDf = (
        regularDf.sample(n=regularSampleCount, random_state=seed)
        if regularSampleCount > 0
        else regularDf.head(0).copy()
    )

    overlaySampleDf = pd.concat([mismatchSampleDf, regularSampleDf], ignore_index=True)

    overlaySampleDf = overlaySampleDf.sort_values(
        ["hasXmlFilenameMismatch", "subset", "imageFileName"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return overlaySampleDf


def getImageObjectTable(annotationDf: pd.DataFrame, imageStem: str | int) -> pd.DataFrame:
    """We collect all annotation rows that belong to one source image stem."""
    imageStemString = str(imageStem)

    imageObjectDf = annotationDf.loc[
        annotationDf["sourceImageStem"].astype(str) == imageStemString
    ].copy()

    imageObjectDf = imageObjectDf.sort_values(["className", "xmin", "ymin"]).reset_index(drop=True)
    return imageObjectDf


def getClassColorMap() -> dict[str, str]:
    """We define one consistent overlay color per class so the saved examples are easier to read."""
    return {
        "craze": "red",
        "corrosion": "orange",
        "surface_injure": "yellow",
        "thunderstrike": "cyan",
        "crack": "lime",
        "hide_craze": "magenta",
    }


def drawBoundingBoxesOnImage(imagePath: Path, imageObjectDf: pd.DataFrame) -> Image.Image:
    """We draw class-labeled bounding boxes on top of one image and return the overlay image."""
    overlayImage = Image.open(imagePath).convert("RGB")
    drawObject = ImageDraw.Draw(overlayImage)
    fontObject = ImageFont.load_default()

    classColorMap = getClassColorMap()

    for objectRow in imageObjectDf.itertuples(index=False):
        xmin = int(objectRow.xmin)
        ymin = int(objectRow.ymin)
        xmax = int(objectRow.xmax)
        ymax = int(objectRow.ymax)

        className = str(objectRow.className)
        boxColor = classColorMap.get(className, "white")

        drawObject.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            outline=boxColor,
            width=3,
        )

        labelText = className
        textAnchorPoint = (xmin + 3, max(0, ymin - 14))
        drawObject.text(
            textAnchorPoint,
            labelText,
            fill=boxColor,
            font=fontObject,
        )

    return overlayImage


def buildOverlaySummaryRecord(
    imageRow: pd.Series,
    imageObjectDf: pd.DataFrame,
    overlayPath: Path,
) -> dict:
    """We create one summary record for each saved overlay example."""
    classNames = sorted(imageObjectDf["className"].astype(str).unique().tolist())
    classSummary = ", ".join(classNames)

    return {
        "imageFileName": imageRow["imageFileName"],
        "imageStem": str(imageRow["imageStem"]),
        "subset": imageRow["subset"],
        "objectCount": int(imageRow["objectCount"]),
        "distinctClassCount": int(imageRow["distinctClassCount"]),
        "hasXmlFilenameMismatch": int(imageRow["hasXmlFilenameMismatch"]),
        "overlayPath": str(overlayPath),
        "classSummary": classSummary,
    }


def saveOverlaySummaryTable(overlaySummaryDf: pd.DataFrame, outputDir: Path) -> Path:
    """We save the overlay summary table for later inspection and report writing."""
    summaryPath = outputDir / "overlayAuditSummary.csv"
    overlaySummaryDf.to_csv(summaryPath, index=False)
    return summaryPath


def runOverlayAudit(
    projectConfig: ProjectConfig,
    maxMismatchSamples: int = 8,
    maxRegularSamples: int = 8,
) -> dict:
    """We generate a reproducible set of overlay examples to visually validate our matching rule."""
    annotationDf, imageLevelAuditDf = loadAuditTables(projectConfig)

    overlayOutputDir = projectConfig.figuresDir / "overlayAudit"
    prepareOverlayOutputDir(overlayOutputDir)

    overlaySampleDf = selectOverlaySampleTable(
        imageLevelAuditDf=imageLevelAuditDf,
        maxMismatchSamples=maxMismatchSamples,
        maxRegularSamples=maxRegularSamples,
        seed=projectConfig.seed,
    )

    overlaySummaryRecords = []
    savedOverlayPathDict = {}

    # We save one overlay image per sampled source image so we can inspect box placement visually.
    for imageRow in overlaySampleDf.to_dict(orient="records"):
        imageFileName = str(imageRow["imageFileName"])
        imageStem = str(imageRow["imageStem"])

        imagePath = projectConfig.imageDir / imageFileName
        imageObjectDf = getImageObjectTable(annotationDf, imageStem=imageStem)

        overlayImage = drawBoundingBoxesOnImage(imagePath=imagePath, imageObjectDf=imageObjectDf)

        overlayFileName = f"{imageStem}_overlay.png"
        overlayPath = overlayOutputDir / overlayFileName
        overlayImage.save(overlayPath)

        overlaySummaryRecords.append(
            buildOverlaySummaryRecord(
                imageRow=pd.Series(imageRow),
                imageObjectDf=imageObjectDf,
                overlayPath=overlayPath,
            )
        )

        savedOverlayPathDict[imageFileName] = str(overlayPath)

    overlaySummaryDf = pd.DataFrame(overlaySummaryRecords)
    overlaySummaryPath = saveOverlaySummaryTable(overlaySummaryDf, overlayOutputDir)

    return {
        "overlaySummaryDf": overlaySummaryDf,
        "overlaySummaryPath": str(overlaySummaryPath),
        "savedOverlayPathDict": savedOverlayPathDict,
        "overlayOutputDir": str(overlayOutputDir),
    }