from __future__ import annotations

"""We implement the raw-data audit layer for the wind-turbine defect project."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

from src.config import ProjectConfig


def readClassDefinitions(classFilePath: Path) -> list[str]:
    """We read the class-definition file and return the ordered class list."""
    with open(classFilePath, "r", encoding="utf-8") as textFile:
        classNames = [line.strip() for line in textFile.readlines() if line.strip()]
    return classNames


def readSplitTable(splitFilePath: Path) -> pd.DataFrame:
    """We read and validate the official image-level split table."""
    splitDf = pd.read_csv(splitFilePath)

    expectedColumns = {"ImageID", "Subset"}
    if not expectedColumns.issubset(splitDf.columns):
        raise ValueError(f"Split file must contain columns: {expectedColumns}")

    splitDf = splitDf.rename(columns={"ImageID": "imageFileName", "Subset": "subset"})
    splitDf["imageFileName"] = splitDf["imageFileName"].astype(str).str.strip()
    splitDf["subset"] = splitDf["subset"].astype(str).str.strip().str.lower()
    splitDf["imageStem"] = splitDf["imageFileName"].apply(lambda currentName: Path(currentName).stem)

    validSubsets = {"train", "val", "test"}
    invalidSubsetMask = ~splitDf["subset"].isin(validSubsets)
    if invalidSubsetMask.any():
        invalidSubsets = sorted(splitDf.loc[invalidSubsetMask, "subset"].unique().tolist())
        raise ValueError(f"Invalid subset values found: {invalidSubsets}")

    splitDf = splitDf.sort_values(["subset", "imageFileName"]).reset_index(drop=True)
    return splitDf


def getFileStemSet(targetDir: Path, validSuffixes: tuple[str, ...]) -> set[str]:
    """We collect the file stems from a directory for the given valid suffixes."""
    return {
        currentPath.stem
        for currentPath in targetDir.iterdir()
        if currentPath.is_file() and currentPath.suffix.lower() in validSuffixes
    }


def computeFileHash(filePath: Path, chunkSize: int = 1024 * 1024) -> str:
    """We compute a SHA-256 hash for a file so we can detect exact duplicates."""
    hashObject = hashlib.sha256()

    with open(filePath, "rb") as binaryFile:
        while True:
            chunk = binaryFile.read(chunkSize)
            if not chunk:
                break
            hashObject.update(chunk)

    return hashObject.hexdigest()


def getElementText(parentElement: ET.Element | None, tagName: str, defaultValue: str = "") -> str:
    """We safely read text from an XML child element and return a fallback when missing."""
    if parentElement is None:
        return defaultValue

    childElement = parentElement.find(tagName)
    if childElement is None or childElement.text is None:
        return defaultValue

    return childElement.text.strip()


def safeParseInt(value: str, defaultValue: int = 0) -> int:
    """We safely convert a value to an integer and return a fallback on failure."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return defaultValue


def buildImageInventoryTable(imageDir: Path) -> pd.DataFrame:
    """We build the raw image inventory table with file size and exact-duplicate hashes."""
    imagePaths = sorted(
        [
            currentPath
            for currentPath in imageDir.iterdir()
            if currentPath.is_file() and currentPath.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    imageRecords = []
    for imagePath in imagePaths:
        imageRecords.append(
            {
                "imageFileName": imagePath.name,
                "imageStem": imagePath.stem,
                "imagePath": str(imagePath),
                "fileSizeBytes": imagePath.stat().st_size,
                "fileHash": computeFileHash(imagePath),
            }
        )

    imageInventoryDf = pd.DataFrame(imageRecords)

    if imageInventoryDf.empty:
        imageInventoryDf["duplicateGroupSize"] = pd.Series(dtype="int64")
        return imageInventoryDf

    duplicateGroupSizes = imageInventoryDf["fileHash"].value_counts()
    imageInventoryDf["duplicateGroupSize"] = imageInventoryDf["fileHash"].map(duplicateGroupSizes).astype(int)

    imageInventoryDf = imageInventoryDf.sort_values("imageFileName").reset_index(drop=True)
    return imageInventoryDf


def parseAnnotationFile(xmlPath: Path) -> list[dict]:
    """We parse one Pascal VOC XML file into object-level records using xmlStem as the trusted image key."""
    tree = ET.parse(xmlPath)
    root = tree.getroot()

    xmlFileName = xmlPath.name
    xmlStem = xmlPath.stem

    xmlFilenameField = getElementText(root, "filename", "")
    xmlFilenameFieldStem = Path(xmlFilenameField).stem if xmlFilenameField else ""

    sizeElement = root.find("size")
    imageWidth = safeParseInt(getElementText(sizeElement, "width", "0"))
    imageHeight = safeParseInt(getElementText(sizeElement, "height", "0"))

    sourceImageStem = xmlStem
    sourceImageFileName = f"{xmlStem}.jpg"

    objectRecords = []
    for objectElement in root.findall("object"):
        bboxElement = objectElement.find("bndbox")

        xmin = safeParseInt(getElementText(bboxElement, "xmin", "0"))
        ymin = safeParseInt(getElementText(bboxElement, "ymin", "0"))
        xmax = safeParseInt(getElementText(bboxElement, "xmax", "0"))
        ymax = safeParseInt(getElementText(bboxElement, "ymax", "0"))

        bboxWidth = max(0, xmax - xmin)
        bboxHeight = max(0, ymax - ymin)
        bboxArea = bboxWidth * bboxHeight

        objectRecords.append(
            {
                "xmlFileName": xmlFileName,
                "xmlStem": xmlStem,
                "sourceImageFileName": sourceImageFileName,
                "sourceImageStem": sourceImageStem,
                "xmlFilenameField": xmlFilenameField,
                "xmlFilenameFieldStem": xmlFilenameFieldStem,
                "isXmlFilenameMismatch": int(xmlFilenameFieldStem != "" and xmlFilenameFieldStem != xmlStem),
                "className": getElementText(objectElement, "name", ""),
                "pose": getElementText(objectElement, "pose", ""),
                "truncated": safeParseInt(getElementText(objectElement, "truncated", "0")),
                "difficult": safeParseInt(getElementText(objectElement, "difficult", "0")),
                "imageWidth": imageWidth,
                "imageHeight": imageHeight,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "bboxWidth": bboxWidth,
                "bboxHeight": bboxHeight,
                "bboxArea": bboxArea,
                "isValidBbox": int(xmax > xmin and ymax > ymin),
            }
        )

    return objectRecords


def buildAnnotationTable(annotationDir: Path) -> pd.DataFrame:
    """We parse all XML files into one object-level annotation table."""
    xmlPaths = sorted(
        [
            currentPath
            for currentPath in annotationDir.iterdir()
            if currentPath.is_file() and currentPath.suffix.lower() == ".xml"
        ]
    )

    objectRecords = []
    for xmlPath in xmlPaths:
        objectRecords.extend(parseAnnotationFile(xmlPath))

    if not objectRecords:
        expectedColumns = [
            "xmlFileName",
            "xmlStem",
            "sourceImageFileName",
            "sourceImageStem",
            "xmlFilenameField",
            "xmlFilenameFieldStem",
            "isXmlFilenameMismatch",
            "className",
            "pose",
            "truncated",
            "difficult",
            "imageWidth",
            "imageHeight",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "bboxWidth",
            "bboxHeight",
            "bboxArea",
            "isValidBbox",
        ]
        return pd.DataFrame(columns=expectedColumns)

    annotationDf = pd.DataFrame(objectRecords)
    annotationDf = annotationDf.sort_values(
        ["xmlFileName", "className", "xmin", "ymin"]
    ).reset_index(drop=True)
    return annotationDf


def buildImageLevelAuditTable(
    imageInventoryDf: pd.DataFrame,
    splitDf: pd.DataFrame,
    annotationDf: pd.DataFrame,
) -> pd.DataFrame:
    """We aggregate object-level information into one image-level audit table using image stems as keys."""
    objectCountDf = (
        annotationDf.groupby("sourceImageStem")
        .size()
        .rename("objectCount")
        .reset_index()
    )

    distinctClassCountDf = (
        annotationDf.groupby("sourceImageStem")["className"]
        .nunique()
        .rename("distinctClassCount")
        .reset_index()
    )

    invalidBboxCountDf = (
        (annotationDf["isValidBbox"] == 0)
        .groupby(annotationDf["sourceImageStem"])
        .sum()
        .rename("invalidBboxCount")
        .reset_index()
    )

    xmlMismatchCountDf = (
        annotationDf.groupby("sourceImageStem")["isXmlFilenameMismatch"]
        .max()
        .rename("hasXmlFilenameMismatch")
        .reset_index()
    )

    # We merge real image files with the official split using the actual image stem.
    imageLevelAuditDf = imageInventoryDf.merge(
        splitDf[["imageFileName", "imageStem", "subset"]].rename(
            columns={"imageFileName": "splitImageFileName"}
        ),
        on="imageStem",
        how="left",
        validate="1:1",
    )

    # We then enrich the same table with object counts derived from xmlStem-based annotation mapping.
    imageLevelAuditDf = imageLevelAuditDf.merge(
        objectCountDf,
        left_on="imageStem",
        right_on="sourceImageStem",
        how="left",
        validate="1:1",
    )

    imageLevelAuditDf = imageLevelAuditDf.merge(
        distinctClassCountDf,
        on="sourceImageStem",
        how="left",
        validate="1:1",
    )

    imageLevelAuditDf = imageLevelAuditDf.merge(
        invalidBboxCountDf,
        on="sourceImageStem",
        how="left",
        validate="1:1",
    )

    imageLevelAuditDf = imageLevelAuditDf.merge(
        xmlMismatchCountDf,
        on="sourceImageStem",
        how="left",
        validate="1:1",
    )

    imageLevelAuditDf["objectCount"] = imageLevelAuditDf["objectCount"].fillna(0).astype(int)
    imageLevelAuditDf["distinctClassCount"] = imageLevelAuditDf["distinctClassCount"].fillna(0).astype(int)
    imageLevelAuditDf["invalidBboxCount"] = imageLevelAuditDf["invalidBboxCount"].fillna(0).astype(int)
    imageLevelAuditDf["hasXmlFilenameMismatch"] = imageLevelAuditDf["hasXmlFilenameMismatch"].fillna(0).astype(int)
    imageLevelAuditDf["hasObjects"] = (imageLevelAuditDf["objectCount"] > 0).astype(int)
    imageLevelAuditDf["isSplitFileNameMismatch"] = (
        imageLevelAuditDf["splitImageFileName"].notna()
        & (imageLevelAuditDf["imageFileName"] != imageLevelAuditDf["splitImageFileName"])
    ).astype(int)

    imageLevelAuditDf = imageLevelAuditDf.drop(columns=["sourceImageStem"])
    imageLevelAuditDf = imageLevelAuditDf.sort_values("imageFileName").reset_index(drop=True)
    return imageLevelAuditDf


def buildDuplicateTables(imageLevelAuditDf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """We build exact-duplicate and cross-split-duplicate tables from hashed images."""
    exactDuplicateDf = imageLevelAuditDf.loc[
        imageLevelAuditDf["duplicateGroupSize"] > 1,
        [
            "imageFileName",
            "imageStem",
            "subset",
            "fileSizeBytes",
            "fileHash",
            "duplicateGroupSize",
        ],
    ].copy()

    exactDuplicateDf = exactDuplicateDf.sort_values(["fileHash", "imageFileName"]).reset_index(drop=True)

    if exactDuplicateDf.empty:
        crossSplitDuplicateDf = exactDuplicateDf.copy()
        return exactDuplicateDf, crossSplitDuplicateDf

    subsetCountPerHash = (
        exactDuplicateDf.groupby("fileHash")["subset"]
        .nunique(dropna=True)
        .rename("subsetCount")
    )

    crossSplitHashes = subsetCountPerHash[subsetCountPerHash > 1].index.tolist()

    crossSplitDuplicateDf = exactDuplicateDf.loc[
        exactDuplicateDf["fileHash"].isin(crossSplitHashes)
    ].copy()

    crossSplitDuplicateDf = crossSplitDuplicateDf.merge(
        subsetCountPerHash.rename("subsetCount"),
        on="fileHash",
        how="left",
        validate="m:1",
    )

    crossSplitDuplicateDf = crossSplitDuplicateDf.sort_values(
        ["fileHash", "subset", "imageFileName"]
    ).reset_index(drop=True)

    return exactDuplicateDf, crossSplitDuplicateDf


def buildStemMismatchTables(
    imageDir: Path,
    annotationDir: Path,
    secondAnnotatorDir: Path,
    splitDf: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """We compare image, XML, second-annotator, and split stems to find missing pairings."""
    imageStemSet = getFileStemSet(imageDir, (".jpg", ".jpeg", ".png"))
    annotationStemSet = getFileStemSet(annotationDir, (".xml",))
    secondAnnotatorStemSet = getFileStemSet(secondAnnotatorDir, (".xml",))
    splitStemSet = set(splitDf["imageStem"].tolist())

    mismatchTables = {
        "imagesMissingAnnotation": pd.DataFrame({"imageStem": sorted(imageStemSet - annotationStemSet)}),
        "annotationsMissingImage": pd.DataFrame({"imageStem": sorted(annotationStemSet - imageStemSet)}),
        "imagesMissingSplit": pd.DataFrame({"imageStem": sorted(imageStemSet - splitStemSet)}),
        "splitMissingImage": pd.DataFrame({"imageStem": sorted(splitStemSet - imageStemSet)}),
        "mainAnnotationMissingSecondAnnotator": pd.DataFrame({"imageStem": sorted(annotationStemSet - secondAnnotatorStemSet)}),
        "secondAnnotatorMissingMainAnnotation": pd.DataFrame({"imageStem": sorted(secondAnnotatorStemSet - annotationStemSet)}),
    }

    return mismatchTables


def buildXmlFilenameMismatchTable(annotationDf: pd.DataFrame) -> pd.DataFrame:
    """We summarize XML files whose internal filename field does not match the XML stem."""
    xmlFilenameMismatchDf = (
        annotationDf.loc[
            annotationDf["isXmlFilenameMismatch"] == 1,
            [
                "xmlFileName",
                "xmlStem",
                "xmlFilenameField",
                "xmlFilenameFieldStem",
                "sourceImageFileName",
                "sourceImageStem",
            ],
        ]
        .drop_duplicates()
        .sort_values(["xmlFileName", "xmlFilenameField"])
        .reset_index(drop=True)
    )

    return xmlFilenameMismatchDf


def buildClassCountTables(
    annotationDf: pd.DataFrame,
    splitDf: pd.DataFrame,
    classNames: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """We build class-level and split-level count tables for the audit summary."""
    annotationWithSplitDf = annotationDf.merge(
        splitDf[["imageStem", "subset"]],
        left_on="sourceImageStem",
        right_on="imageStem",
        how="left",
        validate="m:1",
    )

    classCountsDf = (
        annotationWithSplitDf.groupby("className")
        .size()
        .rename("objectCount")
        .reindex(classNames, fill_value=0)
        .reset_index()
        .rename(columns={"index": "className"})
    )

    splitCountsDf = (
        splitDf.groupby("subset")
        .size()
        .rename("imageCount")
        .reindex(["train", "val", "test"], fill_value=0)
        .reset_index()
        .rename(columns={"index": "subset"})
    )

    classBySplitCountsDf = (
        annotationWithSplitDf.groupby(["subset", "className"])
        .size()
        .rename("objectCount")
        .reset_index()
    )

    # We reindex to the full subset-class grid so missing combinations appear explicitly as zeros.
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


def saveAuditTables(
    auditTables: dict[str, pd.DataFrame],
    auditSummary: dict,
    auditDir: Path,
) -> dict[str, str]:
    """We save all audit tables and the JSON summary into the audit directory."""
    auditDir.mkdir(parents=True, exist_ok=True)

    savedPathDict = {}
    for tableName, tableDf in auditTables.items():
        outputPath = auditDir / f"{tableName}.csv"
        tableDf.to_csv(outputPath, index=False)
        savedPathDict[tableName] = str(outputPath)

    summaryPath = auditDir / "rawAuditSummary.json"
    with open(summaryPath, "w", encoding="utf-8") as jsonFile:
        json.dump(auditSummary, jsonFile, indent=2)

    savedPathDict["rawAuditSummary"] = str(summaryPath)
    return savedPathDict


def runRawDataAudit(projectConfig: ProjectConfig) -> dict:
    """We run the full non-visual raw-data audit and save stable audit artifacts."""
    classNamesFromFile = readClassDefinitions(projectConfig.classFilePath)
    splitDf = readSplitTable(projectConfig.splitFilePath)
    imageInventoryDf = buildImageInventoryTable(projectConfig.imageDir)
    annotationDf = buildAnnotationTable(projectConfig.annotationDir)

    imageLevelAuditDf = buildImageLevelAuditTable(
        imageInventoryDf=imageInventoryDf,
        splitDf=splitDf,
        annotationDf=annotationDf,
    )

    exactDuplicateDf, crossSplitDuplicateDf = buildDuplicateTables(imageLevelAuditDf)

    mismatchTables = buildStemMismatchTables(
        imageDir=projectConfig.imageDir,
        annotationDir=projectConfig.annotationDir,
        secondAnnotatorDir=projectConfig.secondAnnotatorDir,
        splitDf=splitDf,
    )

    xmlFilenameMismatchDf = buildXmlFilenameMismatchTable(annotationDf)

    classCountsDf, splitCountsDf, classBySplitCountsDf = buildClassCountTables(
        annotationDf=annotationDf,
        splitDf=splitDf,
        classNames=classNamesFromFile,
    )

    auditSummary = {
        "seed": projectConfig.seed,
        "device": projectConfig.device,
        "classNamesFromFile": classNamesFromFile,
        "classDefinitionMatchesConfig": classNamesFromFile == projectConfig.classNames,
        "imageFileCount": int(len(imageInventoryDf)),
        "annotationXmlCount": int(len(getFileStemSet(projectConfig.annotationDir, (".xml",)))),
        "secondAnnotatorXmlCount": int(len(getFileStemSet(projectConfig.secondAnnotatorDir, (".xml",)))),
        "splitRowCount": int(len(splitDf)),
        "uniqueSplitImageCount": int(splitDf["imageFileName"].nunique()),
        "annotationObjectCount": int(len(annotationDf)),
        "imagesWithObjectsCount": int((imageLevelAuditDf["hasObjects"] == 1).sum()),
        "imagesWithoutObjectsCount": int((imageLevelAuditDf["hasObjects"] == 0).sum()),
        "invalidBboxObjectCount": int((annotationDf["isValidBbox"] == 0).sum()),
        "objectsWithUnknownClassCount": int((~annotationDf["className"].isin(classNamesFromFile)).sum()),
        "missingSubsetCount": int(imageLevelAuditDf["subset"].isna().sum()),
        "xmlFilenameMismatchObjectCount": int(annotationDf["isXmlFilenameMismatch"].sum()),
        "xmlFilenameMismatchXmlCount": int(xmlFilenameMismatchDf["xmlFileName"].nunique()),
        "exactDuplicateImageCount": int(len(exactDuplicateDf)),
        "exactDuplicateGroupCount": int(exactDuplicateDf["fileHash"].nunique()) if not exactDuplicateDf.empty else 0,
        "crossSplitDuplicateImageCount": int(len(crossSplitDuplicateDf)),
        "crossSplitDuplicateGroupCount": int(crossSplitDuplicateDf["fileHash"].nunique()) if not crossSplitDuplicateDf.empty else 0,
        "imagesMissingAnnotationCount": int(len(mismatchTables["imagesMissingAnnotation"])),
        "annotationsMissingImageCount": int(len(mismatchTables["annotationsMissingImage"])),
        "imagesMissingSplitCount": int(len(mismatchTables["imagesMissingSplit"])),
        "splitMissingImageCount": int(len(mismatchTables["splitMissingImage"])),
        "mainAnnotationMissingSecondAnnotatorCount": int(len(mismatchTables["mainAnnotationMissingSecondAnnotator"])),
        "secondAnnotatorMissingMainAnnotationCount": int(len(mismatchTables["secondAnnotatorMissingMainAnnotation"])),
    }

    auditTables = {
        "splitTable": splitDf,
        "imageInventory": imageInventoryDf,
        "annotationObjects": annotationDf,
        "imageLevelAudit": imageLevelAuditDf,
        "xmlFilenameMismatches": xmlFilenameMismatchDf,
        "classCounts": classCountsDf,
        "splitCounts": splitCountsDf,
        "classBySplitCounts": classBySplitCountsDf,
        "exactDuplicates": exactDuplicateDf,
        "crossSplitDuplicates": crossSplitDuplicateDf,
        **mismatchTables,
    }

    savedPathDict = saveAuditTables(
        auditTables=auditTables,
        auditSummary=auditSummary,
        auditDir=projectConfig.auditDir,
    )

    return {
        "auditSummary": auditSummary,
        "savedPathDict": savedPathDict,
    }