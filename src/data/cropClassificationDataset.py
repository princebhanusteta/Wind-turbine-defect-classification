from __future__ import annotations

"""We expose the strict modeling crops as a clean PyTorch classification dataset."""

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REQUIRED_MODELING_COLUMNS = [
    "cropFileName",
    "cropPath",
    "subset",
    "className",
    "sourceImageFileName",
    "sourceImageStem",
    "hasXmlFilenameMismatch",
    "isExactDuplicateImage",
    "isCrossSplitDuplicateImage",
    "cropWidth",
    "cropHeight",
]


def validateModelingManifestColumns(modelingManifestDf: pd.DataFrame) -> None:
    """We verify that the modeling manifest contains all required columns before dataset creation."""
    missingColumns = [
        columnName
        for columnName in REQUIRED_MODELING_COLUMNS
        if columnName not in modelingManifestDf.columns
    ]

    if missingColumns:
        raise ValueError(f"Modeling manifest is missing required columns: {missingColumns}")


def loadModelingManifest(manifestPath: Path) -> pd.DataFrame:
    """We load the strict modeling manifest from disk."""
    modelingManifestDf = pd.read_csv(manifestPath)
    validateModelingManifestColumns(modelingManifestDf)

    modelingManifestDf["className"] = modelingManifestDf["className"].astype(str)
    modelingManifestDf["subset"] = modelingManifestDf["subset"].astype(str)
    modelingManifestDf["cropPath"] = modelingManifestDf["cropPath"].astype(str)

    return modelingManifestDf


def filterManifestBySubset(modelingManifestDf: pd.DataFrame, subsetName: str) -> pd.DataFrame:
    """We select one subset from the modeling manifest and return a clean indexed table."""
    validSubsetNames = {"train", "val", "test"}
    if subsetName not in validSubsetNames:
        raise ValueError(f"subsetName must be one of {sorted(validSubsetNames)}, got: {subsetName}")

    subsetDf = modelingManifestDf.loc[
        modelingManifestDf["subset"] == subsetName
    ].copy()

    subsetDf = subsetDf.reset_index(drop=True)
    return subsetDf


def buildClassToIndexMap(classNames: list[str]) -> dict[str, int]:
    """We build a stable class-to-index mapping from the fixed project class order."""
    return {
        className: classIndex
        for classIndex, className in enumerate(classNames)
    }


def buildIndexToClassMap(classNames: list[str]) -> dict[int, str]:
    """We build the inverse index-to-class mapping from the fixed project class order."""
    return {
        classIndex: className
        for classIndex, className in enumerate(classNames)
    }


class CropClassificationDataset(Dataset):
    """We provide one strict modeling subset as a PyTorch image classification dataset."""

    def __init__(
        self,
        modelingManifestDf: pd.DataFrame,
        classNames: list[str],
        transform: Callable[[Image.Image], Any] | None = None,
        returnMetadata: bool = False,
    ) -> None:
        """We store the manifest, class mapping, and optional transform for later sample loading."""
        validateModelingManifestColumns(modelingManifestDf)

        self.modelingManifestDf = modelingManifestDf.copy().reset_index(drop=True)
        self.classNames = list(classNames)
        self.classToIndexMap = buildClassToIndexMap(self.classNames)
        self.indexToClassMap = buildIndexToClassMap(self.classNames)
        self.transform = transform
        self.returnMetadata = returnMetadata

    def __len__(self) -> int:
        """We return the number of crops available in this dataset view."""
        return len(self.modelingManifestDf)

    def __getitem__(self, sampleIndex: int) -> Any:
        """We load one crop image, convert its class name to an index, and optionally return metadata."""
        sampleRow = self.modelingManifestDf.iloc[sampleIndex]

        imagePath = Path(str(sampleRow["cropPath"]))
        imageObject = Image.open(imagePath).convert("RGB")

        if self.transform is not None:
            imageObject = self.transform(imageObject)

        className = str(sampleRow["className"])
        classIndex = self.classToIndexMap[className]

        if not self.returnMetadata:
            return imageObject, classIndex

        metadataDict = {
            "cropFileName": str(sampleRow["cropFileName"]),
            "subset": str(sampleRow["subset"]),
            "className": className,
            "sourceImageFileName": str(sampleRow["sourceImageFileName"]),
            "sourceImageStem": str(sampleRow["sourceImageStem"]),
            "cropWidth": int(sampleRow["cropWidth"]),
            "cropHeight": int(sampleRow["cropHeight"]),
            "hasXmlFilenameMismatch": int(sampleRow["hasXmlFilenameMismatch"]),
            "isExactDuplicateImage": int(sampleRow["isExactDuplicateImage"]),
            "isCrossSplitDuplicateImage": int(sampleRow["isCrossSplitDuplicateImage"]),
        }

        return imageObject, classIndex, metadataDict


def buildSubsetDataset(
    manifestPath: Path,
    classNames: list[str],
    subsetName: str,
    transform: Callable[[Image.Image], Any] | None = None,
    returnMetadata: bool = False,
) -> CropClassificationDataset:
    """We build one subset dataset directly from the strict modeling manifest."""
    modelingManifestDf = loadModelingManifest(manifestPath)
    subsetDf = filterManifestBySubset(modelingManifestDf, subsetName=subsetName)

    datasetObject = CropClassificationDataset(
        modelingManifestDf=subsetDf,
        classNames=classNames,
        transform=transform,
        returnMetadata=returnMetadata,
    )

    return datasetObject


def buildNeuralDatasets(
    manifestPath: Path,
    classNames: list[str],
    trainTransform: Callable[[Image.Image], Any],
    evalTransform: Callable[[Image.Image], Any],
    returnMetadata: bool = False,
) -> dict[str, CropClassificationDataset]:
    """We build the train, validation, and test datasets for neural image models."""
    modelingManifestDf = loadModelingManifest(manifestPath)

    trainDf = filterManifestBySubset(modelingManifestDf, subsetName="train")
    valDf = filterManifestBySubset(modelingManifestDf, subsetName="val")
    testDf = filterManifestBySubset(modelingManifestDf, subsetName="test")

    datasetDict = {
        "train": CropClassificationDataset(
            modelingManifestDf=trainDf,
            classNames=classNames,
            transform=trainTransform,
            returnMetadata=returnMetadata,
        ),
        "val": CropClassificationDataset(
            modelingManifestDf=valDf,
            classNames=classNames,
            transform=evalTransform,
            returnMetadata=returnMetadata,
        ),
        "test": CropClassificationDataset(
            modelingManifestDf=testDf,
            classNames=classNames,
            transform=evalTransform,
            returnMetadata=returnMetadata,
        ),
    }

    return datasetDict


def buildNeuralDataLoaders(
    datasetDict: dict[str, CropClassificationDataset],
    batchSize: int = 32,
    numWorkers: int = 0,
    seed: int = 27,
) -> dict[str, DataLoader]:
    """We build deterministic train, validation, and test data loaders for neural models."""
    expectedKeys = {"train", "val", "test"}
    datasetKeys = set(datasetDict.keys())

    if datasetKeys != expectedKeys:
        raise ValueError(
            f"datasetDict must contain exactly {sorted(expectedKeys)}, got: {sorted(datasetKeys)}"
        )

    if batchSize <= 0:
        raise ValueError(f"batchSize must be positive, got: {batchSize}")

    if numWorkers < 0:
        raise ValueError(f"numWorkers must be non-negative, got: {numWorkers}")

    trainGenerator = torch.Generator()
    trainGenerator.manual_seed(seed)

    dataLoaderDict = {
        "train": DataLoader(
            datasetDict["train"],
            batch_size=batchSize,
            shuffle=True,
            num_workers=numWorkers,
            generator=trainGenerator,
            persistent_workers=numWorkers > 0,
        ),
        "val": DataLoader(
            datasetDict["val"],
            batch_size=batchSize,
            shuffle=False,
            num_workers=numWorkers,
            persistent_workers=numWorkers > 0,
        ),
        "test": DataLoader(
            datasetDict["test"],
            batch_size=batchSize,
            shuffle=False,
            num_workers=numWorkers,
            persistent_workers=numWorkers > 0,
        ),
    }

    return dataLoaderDict