from __future__ import annotations

"""We build the crop dataset and crop manifest using the stem-based matching rule."""

import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config import ProjectConfig
from src.data.rawDataAudit import runRawDataAudit


def loadCropSourceTables(projectConfig: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """We load the audit tables that act as the trusted source for crop extraction."""
    auditResult = runRawDataAudit(projectConfig)
    savedPathDict = auditResult["savedPathDict"]

    annotationDf = pd.read_csv(savedPathDict["annotationObjects"])
    imageLevelAuditDf = pd.read_csv(savedPathDict["imageLevelAudit"])
    crossSplitDuplicatesDf = pd.read_csv(savedPathDict["crossSplitDuplicates"])

    annotationDf["sourceImageStem"] = annotationDf["sourceImageStem"].astype(str)
    imageLevelAuditDf["imageStem"] = imageLevelAuditDf["imageStem"].astype(str)

    if not crossSplitDuplicatesDf.empty:
        crossSplitDuplicatesDf["imageStem"] = crossSplitDuplicatesDf["imageStem"].astype(str)

    return annotationDf, imageLevelAuditDf, crossSplitDuplicatesDf


def prepareCropOutputDir(cropsDir: Path) -> None:
    """We create a clean crop output directory so old crop files do not remain mixed with the new build."""
    if cropsDir.exists():
        shutil.rmtree(cropsDir)

    cropsDir.mkdir(parents=True, exist_ok=True)


def ensureSubsetClassDirs(cropsDir: Path, subsetNames: list[str], classNames: list[str]) -> None:
    """We create the subset and class folder structure used for saving crop images."""
    for subsetName in subsetNames:
        for className in classNames:
            currentDir = cropsDir / subsetName / className
            currentDir.mkdir(parents=True, exist_ok=True)


def buildImageMetadataTable(
    imageLevelAuditDf: pd.DataFrame,
    crossSplitDuplicatesDf: pd.DataFrame,
) -> pd.DataFrame:
    """We build one image-level metadata table that carries split and duplicate information into the crop manifest."""
    imageMetadataDf = imageLevelAuditDf.copy()

    crossSplitStemSet = (
        set(crossSplitDuplicatesDf["imageStem"].astype(str).tolist())
        if not crossSplitDuplicatesDf.empty
        else set()
    )

    imageMetadataDf["isExactDuplicateImage"] = (imageMetadataDf["duplicateGroupSize"] > 1).astype(int)
    imageMetadataDf["isCrossSplitDuplicateImage"] = imageMetadataDf["imageStem"].isin(crossSplitStemSet).astype(int)

    selectedColumns = [
        "imageFileName",
        "imageStem",
        "subset",
        "objectCount",
        "distinctClassCount",
        "hasXmlFilenameMismatch",
        "duplicateGroupSize",
        "isExactDuplicateImage",
        "isCrossSplitDuplicateImage",
    ]

    imageMetadataDf = imageMetadataDf[selectedColumns].copy()
    imageMetadataDf = imageMetadataDf.sort_values("imageFileName").reset_index(drop=True)
    return imageMetadataDf


def clampBoundingBox(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    imageWidth: int,
    imageHeight: int,
    cropPadding: int,
) -> tuple[int, int, int, int]:
    """We clamp the bounding box to valid image boundaries after applying optional padding."""
    paddedXmin = max(0, xmin - cropPadding)
    paddedYmin = max(0, ymin - cropPadding)
    paddedXmax = min(imageWidth, xmax + cropPadding)
    paddedYmax = min(imageHeight, ymax + cropPadding)

    if paddedXmax <= paddedXmin or paddedYmax <= paddedYmin:
        raise ValueError(
            "Invalid crop box after clamping: "
            f"({paddedXmin}, {paddedYmin}, {paddedXmax}, {paddedYmax})"
        )

    return paddedXmin, paddedYmin, paddedXmax, paddedYmax


def createCropFileName(subset: str, imageStem: str, objectIndexInSourceImage: int, className: str) -> str:
    """We create a stable crop file name so every crop can be traced back to its source image and object index."""
    safeClassName = className.replace(" ", "_")
    return f"{subset}__{imageStem}__obj{objectIndexInSourceImage:03d}__{safeClassName}.png"


def saveCropImage(
    sourceImage: Image.Image,
    cropBox: tuple[int, int, int, int],
    cropPath: Path,
) -> tuple[int, int]:
    """We crop one region from the source image, save it, and return the crop width and height."""
    cropImage = sourceImage.crop(cropBox)
    cropImage.save(cropPath)

    cropWidth, cropHeight = cropImage.size
    return cropWidth, cropHeight


def buildCropRecord(
    annotationRow: pd.Series,
    imageMetaRow: pd.Series,
    cropPath: Path,
    cropWidth: int,
    cropHeight: int,
    cropBox: tuple[int, int, int, int],
    objectIndexInSourceImage: int,
) -> dict:
    """We create one manifest record for a saved crop."""
    cropXmin, cropYmin, cropXmax, cropYmax = cropBox

    return {
        "cropFileName": cropPath.name,
        "cropPath": str(cropPath),
        "subset": str(imageMetaRow["subset"]),
        "className": str(annotationRow["className"]),
        "sourceImageFileName": str(annotationRow["sourceImageFileName"]),
        "sourceImageStem": str(annotationRow["sourceImageStem"]),
        "xmlFileName": str(annotationRow["xmlFileName"]),
        "xmlStem": str(annotationRow["xmlStem"]),
        "xmlFilenameField": str(annotationRow["xmlFilenameField"]),
        "xmlFilenameFieldStem": str(annotationRow["xmlFilenameFieldStem"]),
        "objectIndexInSourceImage": int(objectIndexInSourceImage),
        "sourceImageObjectCount": int(imageMetaRow["objectCount"]),
        "sourceImageDistinctClassCount": int(imageMetaRow["distinctClassCount"]),
        "hasXmlFilenameMismatch": int(imageMetaRow["hasXmlFilenameMismatch"]),
        "duplicateGroupSize": int(imageMetaRow["duplicateGroupSize"]),
        "isExactDuplicateImage": int(imageMetaRow["isExactDuplicateImage"]),
        "isCrossSplitDuplicateImage": int(imageMetaRow["isCrossSplitDuplicateImage"]),
        "originalXmin": int(annotationRow["xmin"]),
        "originalYmin": int(annotationRow["ymin"]),
        "originalXmax": int(annotationRow["xmax"]),
        "originalYmax": int(annotationRow["ymax"]),
        "cropXmin": int(cropXmin),
        "cropYmin": int(cropYmin),
        "cropXmax": int(cropXmax),
        "cropYmax": int(cropYmax),
        "cropWidth": int(cropWidth),
        "cropHeight": int(cropHeight),
        "cropArea": int(cropWidth * cropHeight),
        "bboxWidth": int(annotationRow["bboxWidth"]),
        "bboxHeight": int(annotationRow["bboxHeight"]),
        "bboxArea": int(annotationRow["bboxArea"]),
    }


def buildCropManifest(
    projectConfig: ProjectConfig,
    annotationDf: pd.DataFrame,
    imageMetadataDf: pd.DataFrame,
    cropPadding: int,
) -> pd.DataFrame:
    """We extract all object crops and build the crop manifest using the source image split and duplicate metadata."""
    imageMetadataByStemDf = imageMetadataDf.set_index("imageStem", drop=False)

    cropRecords = []

    groupedAnnotationDf = (
        annotationDf.sort_values(["sourceImageFileName", "className", "xmin", "ymin"])
        .groupby(["sourceImageFileName", "sourceImageStem"], sort=True)
    )

    # We load each source image once, extract all of its object crops, and carry its image-level metadata into every crop row.
    for (sourceImageFileName, sourceImageStem), imageAnnotationDf in groupedAnnotationDf:
        sourceImageStemString = str(sourceImageStem)

        if sourceImageStemString not in imageMetadataByStemDf.index:
            raise KeyError(f"Missing image-level metadata for image stem: {sourceImageStemString}")

        imageMetaRow = imageMetadataByStemDf.loc[sourceImageStemString]
        subset = str(imageMetaRow["subset"])

        imagePath = projectConfig.imageDir / str(sourceImageFileName)
        sourceImage = Image.open(imagePath).convert("RGB")
        imageWidth, imageHeight = sourceImage.size

        imageAnnotationDf = imageAnnotationDf.reset_index(drop=True)

        for objectIndexInSourceImage, annotationRow in enumerate(
            imageAnnotationDf.to_dict(orient="records"),
            start=1,
        ):
            annotationRowSeries = pd.Series(annotationRow)
            className = str(annotationRowSeries["className"])

            cropBox = clampBoundingBox(
                xmin=int(annotationRowSeries["xmin"]),
                ymin=int(annotationRowSeries["ymin"]),
                xmax=int(annotationRowSeries["xmax"]),
                ymax=int(annotationRowSeries["ymax"]),
                imageWidth=imageWidth,
                imageHeight=imageHeight,
                cropPadding=cropPadding,
            )

            cropFileName = createCropFileName(
                subset=subset,
                imageStem=sourceImageStemString,
                objectIndexInSourceImage=objectIndexInSourceImage,
                className=className,
            )

            cropPath = projectConfig.cropsDir / subset / className / cropFileName

            cropWidth, cropHeight = saveCropImage(
                sourceImage=sourceImage,
                cropBox=cropBox,
                cropPath=cropPath,
            )

            cropRecords.append(
                buildCropRecord(
                    annotationRow=annotationRowSeries,
                    imageMetaRow=imageMetaRow,
                    cropPath=cropPath,
                    cropWidth=cropWidth,
                    cropHeight=cropHeight,
                    cropBox=cropBox,
                    objectIndexInSourceImage=objectIndexInSourceImage,
                )
            )

    cropManifestDf = pd.DataFrame(cropRecords)

    if cropManifestDf.empty:
        return cropManifestDf

    cropManifestDf = cropManifestDf.sort_values(
        ["subset", "className", "sourceImageFileName", "objectIndexInSourceImage"]
    ).reset_index(drop=True)

    return cropManifestDf


def buildCropCountTables(
    cropManifestDf: pd.DataFrame,
    classNames: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """We build class and split count tables for the extracted crop dataset."""
    classCountsDf = (
        cropManifestDf.groupby("className")
        .size()
        .rename("cropCount")
        .reindex(classNames, fill_value=0)
        .reset_index()
        .rename(columns={"index": "className"})
    )

    splitCountsDf = (
        cropManifestDf.groupby("subset")
        .size()
        .rename("cropCount")
        .reindex(["train", "val", "test"], fill_value=0)
        .reset_index()
        .rename(columns={"index": "subset"})
    )

    classBySplitCountsDf = (
        cropManifestDf.groupby(["subset", "className"])
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

    return classCountsDf, splitCountsDf, classBySplitCountsDf


def buildCropBuildSummary(cropManifestDf: pd.DataFrame, cropPadding: int) -> dict:
    """We summarize the crop build so later notebooks and the report can reference the key numbers directly."""
    return {
        "cropPadding": int(cropPadding),
        "cropCount": int(len(cropManifestDf)),
        "uniqueSourceImageCount": int(cropManifestDf["sourceImageFileName"].nunique()),
        "classCount": int(cropManifestDf["className"].nunique()),
        "subsetCount": int(cropManifestDf["subset"].nunique()),
        "mismatchCropCount": int(cropManifestDf["hasXmlFilenameMismatch"].sum()),
        "exactDuplicateCropCount": int(cropManifestDf["isExactDuplicateImage"].sum()),
        "crossSplitDuplicateCropCount": int(cropManifestDf["isCrossSplitDuplicateImage"].sum()),
        "minCropWidth": int(cropManifestDf["cropWidth"].min()),
        "maxCropWidth": int(cropManifestDf["cropWidth"].max()),
        "minCropHeight": int(cropManifestDf["cropHeight"].min()),
        "maxCropHeight": int(cropManifestDf["cropHeight"].max()),
    }


def saveCropArtifacts(
    cropManifestDf: pd.DataFrame,
    classCountsDf: pd.DataFrame,
    splitCountsDf: pd.DataFrame,
    classBySplitCountsDf: pd.DataFrame,
    cropBuildSummary: dict,
    manifestsDir: Path,
) -> dict[str, str]:
    """We save the crop manifest and summary artifacts into the manifests directory."""
    manifestsDir.mkdir(parents=True, exist_ok=True)

    cropManifestPath = manifestsDir / "cropManifest.csv"
    cropClassCountsPath = manifestsDir / "cropClassCounts.csv"
    cropSplitCountsPath = manifestsDir / "cropSplitCounts.csv"
    cropClassBySplitCountsPath = manifestsDir / "cropClassBySplitCounts.csv"
    cropBuildSummaryPath = manifestsDir / "cropBuildSummary.json"

    cropManifestDf.to_csv(cropManifestPath, index=False)
    classCountsDf.to_csv(cropClassCountsPath, index=False)
    splitCountsDf.to_csv(cropSplitCountsPath, index=False)
    classBySplitCountsDf.to_csv(cropClassBySplitCountsPath, index=False)

    with open(cropBuildSummaryPath, "w", encoding="utf-8") as jsonFile:
        json.dump(cropBuildSummary, jsonFile, indent=2)

    return {
        "cropManifest": str(cropManifestPath),
        "cropClassCounts": str(cropClassCountsPath),
        "cropSplitCounts": str(cropSplitCountsPath),
        "cropClassBySplitCounts": str(cropClassBySplitCountsPath),
        "cropBuildSummary": str(cropBuildSummaryPath),
    }


def runCropDatasetBuilder(
    projectConfig: ProjectConfig,
    cropPadding: int = 0,
) -> dict:
    """We extract the crop dataset, preserve the original split, and carry duplicate-risk metadata into the crop manifest."""
    annotationDf, imageLevelAuditDf, crossSplitDuplicatesDf = loadCropSourceTables(projectConfig)

    imageMetadataDf = buildImageMetadataTable(
        imageLevelAuditDf=imageLevelAuditDf,
        crossSplitDuplicatesDf=crossSplitDuplicatesDf,
    )

    prepareCropOutputDir(projectConfig.cropsDir)
    ensureSubsetClassDirs(
        cropsDir=projectConfig.cropsDir,
        subsetNames=["train", "val", "test"],
        classNames=projectConfig.classNames,
    )

    cropManifestDf = buildCropManifest(
        projectConfig=projectConfig,
        annotationDf=annotationDf,
        imageMetadataDf=imageMetadataDf,
        cropPadding=cropPadding,
    )

    classCountsDf, splitCountsDf, classBySplitCountsDf = buildCropCountTables(
        cropManifestDf=cropManifestDf,
        classNames=projectConfig.classNames,
    )

    cropBuildSummary = buildCropBuildSummary(
        cropManifestDf=cropManifestDf,
        cropPadding=cropPadding,
    )

    savedPathDict = saveCropArtifacts(
        cropManifestDf=cropManifestDf,
        classCountsDf=classCountsDf,
        splitCountsDf=splitCountsDf,
        classBySplitCountsDf=classBySplitCountsDf,
        cropBuildSummary=cropBuildSummary,
        manifestsDir=projectConfig.manifestsDir,
    )

    return {
        "cropManifestDf": cropManifestDf,
        "classCountsDf": classCountsDf,
        "splitCountsDf": splitCountsDf,
        "classBySplitCountsDf": classBySplitCountsDf,
        "cropBuildSummary": cropBuildSummary,
        "savedPathDict": savedPathDict,
    }