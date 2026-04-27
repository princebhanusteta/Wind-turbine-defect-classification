from __future__ import annotations

"""We implement the tuned RGB HOG plus Logistic Regression baseline for leakage-aware crop classification."""

import json
import warnings
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ProjectConfig
from src.features import getBaselineTransform


def loadStrictModelingManifest(projectConfig: ProjectConfig) -> pd.DataFrame:
    """We load the strict modeling manifest that acts as the safe training and evaluation source."""
    manifestPath = projectConfig.manifestsDir / "modelingManifestStrict.csv"
    modelingManifestDf = pd.read_csv(manifestPath)

    modelingManifestDf["subset"] = modelingManifestDf["subset"].astype(str)
    modelingManifestDf["className"] = modelingManifestDf["className"].astype(str)
    modelingManifestDf["cropPath"] = modelingManifestDf["cropPath"].astype(str)

    return modelingManifestDf


def buildClassMaps(classNames: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """We build stable class-name and class-index mappings from the fixed project class order."""
    classToIndexMap = {
        className: classIndex
        for classIndex, className in enumerate(classNames)
    }

    indexToClassMap = {
        classIndex: className
        for classIndex, className in enumerate(classNames)
    }

    return classToIndexMap, indexToClassMap


def getDefaultTuningSearchSpace() -> dict[str, list[Any]]:
    """We define the default focused 30-trial RGB HOG plus Logistic Regression tuning space."""
    return {
        "imageSizeList": [112, 128, 160],
        "cValueList": [0.01, 0.03, 0.1, 0.25, 1.0],
        "pixelsPerCellList": [(8, 8)],
        "orientationsList": [9, 12],
        "colorModeList": ["rgb"],
    }


def normalizeTuningSearchSpace(searchSpace: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """We fill optional tuning keys with clean defaults so the search space stays backward compatible."""
    normalizedSearchSpace = dict(searchSpace)

    if "colorModeList" not in normalizedSearchSpace:
        normalizedSearchSpace["colorModeList"] = ["rgb"]

    return normalizedSearchSpace


def validateTuningSearchSpace(searchSpace: dict[str, list[Any]]) -> None:
    """We verify that the tuning space contains all required parameter lists and valid values."""
    requiredKeys = [
        "imageSizeList",
        "cValueList",
        "pixelsPerCellList",
        "orientationsList",
        "colorModeList",
    ]

    missingKeys = [currentKey for currentKey in requiredKeys if currentKey not in searchSpace]
    if missingKeys:
        raise ValueError(f"Tuning search space is missing keys: {missingKeys}")

    if not searchSpace["imageSizeList"]:
        raise ValueError("imageSizeList must not be empty.")
    if not searchSpace["cValueList"]:
        raise ValueError("cValueList must not be empty.")
    if not searchSpace["pixelsPerCellList"]:
        raise ValueError("pixelsPerCellList must not be empty.")
    if not searchSpace["orientationsList"]:
        raise ValueError("orientationsList must not be empty.")
    if not searchSpace["colorModeList"]:
        raise ValueError("colorModeList must not be empty.")

    for imageSize in searchSpace["imageSizeList"]:
        if int(imageSize) <= 0:
            raise ValueError(f"All image sizes must be positive, got: {imageSize}")

    for cValue in searchSpace["cValueList"]:
        if float(cValue) <= 0:
            raise ValueError(f"All C values must be positive, got: {cValue}")

    for pixelsPerCell in searchSpace["pixelsPerCellList"]:
        if len(tuple(pixelsPerCell)) != 2:
            raise ValueError(f"pixelsPerCell must have length 2, got: {pixelsPerCell}")

        pixelsPerCellX, pixelsPerCellY = tuple(pixelsPerCell)
        if int(pixelsPerCellX) <= 0 or int(pixelsPerCellY) <= 0:
            raise ValueError(f"pixelsPerCell values must be positive, got: {pixelsPerCell}")

    for orientationCount in searchSpace["orientationsList"]:
        if int(orientationCount) <= 0:
            raise ValueError(f"All orientation counts must be positive, got: {orientationCount}")

    validColorModes = {"grayscale", "rgb"}
    for colorMode in searchSpace["colorModeList"]:
        if str(colorMode) not in validColorModes:
            raise ValueError(f"colorMode must be one of {sorted(validColorModes)}, got: {colorMode}")


def buildHogParamDict(
    orientations: int,
    pixelsPerCell: tuple[int, int],
) -> dict[str, Any]:
    """We build one HOG parameter dictionary from the selected tuning values."""
    hogParamDict = {
        "orientations": int(orientations),
        "pixelsPerCell": (int(pixelsPerCell[0]), int(pixelsPerCell[1])),
        "cellsPerBlock": (2, 2),
        "blockNorm": "L2-Hys",
    }

    return hogParamDict


def preprocessBaselineImage(
    imagePath: Path,
    imageSize: int,
    colorMode: str,
) -> np.ndarray:
    """We resize, pad, and convert one crop image to either grayscale or RGB for HOG extraction."""
    if colorMode not in {"grayscale", "rgb"}:
        raise ValueError(f"colorMode must be 'grayscale' or 'rgb', got: {colorMode}")

    baselineTransform = getBaselineTransform(
        imageSize=imageSize,
        convertToGrayscale=(colorMode == "grayscale"),
    )

    imageObject = Image.open(imagePath).convert("RGB")
    transformedImage = baselineTransform(imageObject)

    imageArray = np.asarray(transformedImage, dtype=np.float32) / 255.0
    return imageArray


def extractHogFeatureVector(
    imageArray: np.ndarray,
    hogParams: dict[str, Any],
    colorMode: str,
) -> np.ndarray:
    """We extract one HOG feature vector from a preprocessed grayscale or RGB image array."""
    if colorMode == "grayscale":
        channelAxis = None
    elif colorMode == "rgb":
        channelAxis = -1
    else:
        raise ValueError(f"colorMode must be 'grayscale' or 'rgb', got: {colorMode}")

    featureVector = hog(
        imageArray,
        orientations=int(hogParams["orientations"]),
        pixels_per_cell=tuple(hogParams["pixelsPerCell"]),
        cells_per_block=tuple(hogParams["cellsPerBlock"]),
        block_norm=str(hogParams["blockNorm"]),
        feature_vector=True,
        channel_axis=channelAxis,
    )

    return featureVector.astype(np.float32)


def buildSubsetFeatures(
    modelingManifestDf: pd.DataFrame,
    subsetName: str,
    classToIndexMap: dict[str, int],
    imageSize: int,
    hogParams: dict[str, Any],
    colorMode: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """We build the HOG feature matrix and target vector for one dataset subset."""
    subsetDf = modelingManifestDf.loc[
        modelingManifestDf["subset"] == subsetName
    ].copy()

    subsetDf = subsetDf.reset_index(drop=True)

    featureList = []
    targetList = []

    # We process every crop row in manifest order so the later prediction tables stay traceable.
    for sampleRow in subsetDf.to_dict(orient="records"):
        imagePath = Path(str(sampleRow["cropPath"]))
        imageArray = preprocessBaselineImage(
            imagePath=imagePath,
            imageSize=imageSize,
            colorMode=colorMode,
        )

        featureVector = extractHogFeatureVector(
            imageArray=imageArray,
            hogParams=hogParams,
            colorMode=colorMode,
        )

        className = str(sampleRow["className"])
        classIndex = classToIndexMap[className]

        featureList.append(featureVector)
        targetList.append(classIndex)

    featureMatrix = np.vstack(featureList).astype(np.float32)
    targetArray = np.asarray(targetList, dtype=np.int64)

    return featureMatrix, targetArray, subsetDf


def fitLogisticRegressionModel(
    trainFeatures: np.ndarray,
    trainTargets: np.ndarray,
    seed: int,
    cValue: float,
    tol: float,
    maxIter: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """We fit one Logistic Regression trial and record whether convergence warnings occurred."""
    modelPipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(cValue),
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                    max_iter=int(maxIter),
                    tol=float(tol),
                ),
            ),
        ]
    )

    with warnings.catch_warnings(record=True) as caughtWarnings:
        warnings.simplefilter("always", ConvergenceWarning)
        modelPipeline.fit(trainFeatures, trainTargets)

    convergenceWarningCount = sum(
        1
        for currentWarning in caughtWarnings
        if issubclass(currentWarning.category, ConvergenceWarning)
    )

    fitMeta = {
        "hadConvergenceWarning": int(convergenceWarningCount > 0),
        "convergenceWarningCount": int(convergenceWarningCount),
    }

    return modelPipeline, fitMeta


def predictWithLogisticRegressionModel(
    modelPipeline: Pipeline,
    featureMatrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """We generate class predictions and class-probability scores for one feature matrix."""
    predictedClassIndices = modelPipeline.predict(featureMatrix)
    probabilityScores = modelPipeline.predict_proba(featureMatrix)

    return predictedClassIndices.astype(np.int64), probabilityScores.astype(np.float32)


def buildPredictionTable(
    subsetDf: pd.DataFrame,
    targetArray: np.ndarray,
    predictedClassIndices: np.ndarray,
    scoreMatrix: np.ndarray,
    indexToClassMap: dict[int, str],
) -> pd.DataFrame:
    """We create a prediction table that keeps model outputs traceable back to crop-level metadata."""
    predictedClassNames = [
        indexToClassMap[int(classIndex)]
        for classIndex in predictedClassIndices.tolist()
    ]

    maxScores = scoreMatrix.max(axis=1)

    predictionDf = subsetDf.copy().reset_index(drop=True)
    predictionDf["trueClassIndex"] = targetArray.astype(int)
    predictionDf["predictedClassIndex"] = predictedClassIndices.astype(int)
    predictionDf["predictedClassName"] = predictedClassNames
    predictionDf["predictedScore"] = maxScores.astype(float)
    predictionDf["isCorrectPrediction"] = (
        predictionDf["trueClassIndex"] == predictionDf["predictedClassIndex"]
    ).astype(int)

    return predictionDf


def buildMetricSummary(trueLabels: Any, predictedLabels: Any) -> dict[str, float]:
    """We compute the main multiclass classification metrics used for baseline selection and inspection."""
    metricSummary = {
        "accuracy": float(accuracy_score(trueLabels, predictedLabels)),
        "macroPrecision": float(
            precision_score(trueLabels, predictedLabels, average="macro", zero_division=0)
        ),
        "macroRecall": float(
            recall_score(trueLabels, predictedLabels, average="macro", zero_division=0)
        ),
        "macroF1": float(
            f1_score(trueLabels, predictedLabels, average="macro", zero_division=0)
        ),
        "weightedF1": float(
            f1_score(trueLabels, predictedLabels, average="weighted", zero_division=0)
        ),
    }

    return metricSummary


def buildFeatureCacheKey(
    imageSize: int,
    pixelsPerCell: tuple[int, int],
    orientations: int,
    colorMode: str,
) -> str:
    """We build a stable cache key for one HOG feature-setting combination."""
    return (
        f"size{int(imageSize)}__"
        f"ppc{int(pixelsPerCell[0])}x{int(pixelsPerCell[1])}__"
        f"ori{int(orientations)}__"
        f"color{str(colorMode)}"
    )


def buildTuningFeatureCache(
    modelingManifestDf: pd.DataFrame,
    classToIndexMap: dict[str, int],
    searchSpace: dict[str, list[Any]],
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """We precompute the train and validation HOG features for every unique feature-setting combination."""
    featureCacheDict: dict[str, dict[str, Any]] = {}

    uniqueFeatureSettings = list(
        product(
            searchSpace["imageSizeList"],
            searchSpace["pixelsPerCellList"],
            searchSpace["orientationsList"],
            searchSpace["colorModeList"],
        )
    )

    totalFeatureSettingCount = len(uniqueFeatureSettings)

    for featureSettingIndex, (imageSize, pixelsPerCell, orientations, colorMode) in enumerate(uniqueFeatureSettings, start=1):
        pixelsPerCellTuple = (int(pixelsPerCell[0]), int(pixelsPerCell[1]))
        hogParams = buildHogParamDict(
            orientations=int(orientations),
            pixelsPerCell=pixelsPerCellTuple,
        )

        cacheKey = buildFeatureCacheKey(
            imageSize=int(imageSize),
            pixelsPerCell=pixelsPerCellTuple,
            orientations=int(orientations),
            colorMode=str(colorMode),
        )

        if verbose:
            print(
                f"[HOG cache] Setting {featureSettingIndex}/{totalFeatureSettingCount} | "
                f"imageSize={int(imageSize)} | "
                f"pixelsPerCell={pixelsPerCellTuple} | "
                f"orientations={int(orientations)} | "
                f"colorMode={str(colorMode)}"
            )

        trainFeatures, trainTargets, trainDf = buildSubsetFeatures(
            modelingManifestDf=modelingManifestDf,
            subsetName="train",
            classToIndexMap=classToIndexMap,
            imageSize=int(imageSize),
            hogParams=hogParams,
            colorMode=str(colorMode),
        )

        valFeatures, valTargets, valDf = buildSubsetFeatures(
            modelingManifestDf=modelingManifestDf,
            subsetName="val",
            classToIndexMap=classToIndexMap,
            imageSize=int(imageSize),
            hogParams=hogParams,
            colorMode=str(colorMode),
        )

        featureCacheDict[cacheKey] = {
            "imageSize": int(imageSize),
            "pixelsPerCell": pixelsPerCellTuple,
            "orientations": int(orientations),
            "colorMode": str(colorMode),
            "hogParams": hogParams,
            "trainFeatures": trainFeatures,
            "trainTargets": trainTargets,
            "trainDf": trainDf,
            "valFeatures": valFeatures,
            "valTargets": valTargets,
            "valDf": valDf,
        }

    return featureCacheDict


def buildTuningSummaryDf(
    featureCacheDict: dict[str, dict[str, Any]],
    searchSpace: dict[str, list[Any]],
    seed: int,
    logisticTol: float,
    logisticMaxIter: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """We evaluate every allowed HOG plus Logistic Regression configuration and summarize only train and validation performance."""
    tuningRows = []

    sortedCacheItems = list(featureCacheDict.items())
    totalTrialCount = len(sortedCacheItems) * len(searchSpace["cValueList"])
    currentTrialNumber = 0

    for cacheKey, cacheEntry in sortedCacheItems:
        trainFeatures = cacheEntry["trainFeatures"]
        trainTargets = cacheEntry["trainTargets"]
        valFeatures = cacheEntry["valFeatures"]
        valTargets = cacheEntry["valTargets"]

        for cValue in searchSpace["cValueList"]:
            currentTrialNumber += 1

            if verbose:
                print(
                    f"[HOG tuning] Trial {currentTrialNumber}/{totalTrialCount} | "
                    f"imageSize={int(cacheEntry['imageSize'])} | "
                    f"pixelsPerCell={cacheEntry['pixelsPerCell']} | "
                    f"orientations={int(cacheEntry['orientations'])} | "
                    f"colorMode={str(cacheEntry['colorMode'])} | "
                    f"C={float(cValue)}"
                )

            modelPipeline, fitMeta = fitLogisticRegressionModel(
                trainFeatures=trainFeatures,
                trainTargets=trainTargets,
                seed=seed,
                cValue=float(cValue),
                tol=float(logisticTol),
                maxIter=int(logisticMaxIter),
            )

            trainPredictedIndices, _ = predictWithLogisticRegressionModel(
                modelPipeline=modelPipeline,
                featureMatrix=trainFeatures,
            )

            valPredictedIndices, _ = predictWithLogisticRegressionModel(
                modelPipeline=modelPipeline,
                featureMatrix=valFeatures,
            )

            trainMetricSummary = buildMetricSummary(
                trueLabels=trainTargets,
                predictedLabels=trainPredictedIndices,
            )

            valMetricSummary = buildMetricSummary(
                trueLabels=valTargets,
                predictedLabels=valPredictedIndices,
            )

            tuningRows.append(
                {
                    "trialNumber": int(currentTrialNumber),
                    "featureCacheKey": cacheKey,
                    "imageSize": int(cacheEntry["imageSize"]),
                    "pixelsPerCellX": int(cacheEntry["pixelsPerCell"][0]),
                    "pixelsPerCellY": int(cacheEntry["pixelsPerCell"][1]),
                    "pixelsPerCellLabel": f"{int(cacheEntry['pixelsPerCell'][0])}x{int(cacheEntry['pixelsPerCell'][1])}",
                    "orientations": int(cacheEntry["orientations"]),
                    "colorMode": str(cacheEntry["colorMode"]),
                    "cValue": float(cValue),
                    "logisticTol": float(logisticTol),
                    "logisticMaxIter": int(logisticMaxIter),
                    "featureDimension": int(trainFeatures.shape[1]),
                    "hadConvergenceWarning": int(fitMeta["hadConvergenceWarning"]),
                    "convergenceWarningCount": int(fitMeta["convergenceWarningCount"]),
                    "trainAccuracy": float(trainMetricSummary["accuracy"]),
                    "trainMacroPrecision": float(trainMetricSummary["macroPrecision"]),
                    "trainMacroRecall": float(trainMetricSummary["macroRecall"]),
                    "trainMacroF1": float(trainMetricSummary["macroF1"]),
                    "trainWeightedF1": float(trainMetricSummary["weightedF1"]),
                    "valAccuracy": float(valMetricSummary["accuracy"]),
                    "valMacroPrecision": float(valMetricSummary["macroPrecision"]),
                    "valMacroRecall": float(valMetricSummary["macroRecall"]),
                    "valMacroF1": float(valMetricSummary["macroF1"]),
                    "valWeightedF1": float(valMetricSummary["weightedF1"]),
                }
            )

    tuningSummaryDf = pd.DataFrame(tuningRows)

    if (tuningSummaryDf["hadConvergenceWarning"] == 0).any():
        tuningSummaryDf["isSelectionEligible"] = (
            tuningSummaryDf["hadConvergenceWarning"] == 0
        ).astype(int)
    else:
        tuningSummaryDf["isSelectionEligible"] = 1

    tuningSummaryDf = tuningSummaryDf.sort_values(
        by=[
            "isSelectionEligible",
            "valMacroF1",
            "valMacroRecall",
            "valWeightedF1",
            "valAccuracy",
            "imageSize",
            "cValue",
        ],
        ascending=[False, False, False, False, False, True, True],
    ).reset_index(drop=True)

    tuningSummaryDf["rankBySelection"] = np.arange(1, len(tuningSummaryDf) + 1)

    return tuningSummaryDf


def buildBestMetricSummaryDf(
    trainMetricSummary: dict[str, float],
    valMetricSummary: dict[str, float],
    testMetricSummary: dict[str, float],
) -> pd.DataFrame:
    """We combine the best-config train, validation, and test metrics into one compact table."""
    metricSummaryDf = pd.DataFrame(
        {
            "metricName": list(trainMetricSummary.keys()),
            "trainValue": list(trainMetricSummary.values()),
            "valValue": list(valMetricSummary.values()),
            "testValue": list(testMetricSummary.values()),
        }
    )

    return metricSummaryDf


def saveBaselineArtifacts(
    projectConfig: ProjectConfig,
    bestModelPipeline: Pipeline,
    tuningSummaryDf: pd.DataFrame,
    metricSummaryDf: pd.DataFrame,
    valPredictionDf: pd.DataFrame,
    testPredictionDf: pd.DataFrame,
    baselineConfig: dict[str, Any],
) -> dict[str, str]:
    """We save only the selected best baseline artifacts and the global tuning summary."""
    modelOutputDir = projectConfig.outputsDir / "models" / "hogLogisticRegression"
    tableOutputDir = projectConfig.tablesDir / "hogLogisticRegression"

    modelOutputDir.mkdir(parents=True, exist_ok=True)
    tableOutputDir.mkdir(parents=True, exist_ok=True)

    modelPath = modelOutputDir / "hogLogisticRegression.joblib"
    configPath = modelOutputDir / "hogLogisticRegressionConfig.json"
    tuningSummaryPath = tableOutputDir / "hogLogisticRegressionTuningSummary.csv"
    metricSummaryPath = tableOutputDir / "hogLogisticRegressionMetricSummary.csv"
    valPredictionPath = tableOutputDir / "hogLogisticRegressionValPredictions.csv"
    testPredictionPath = tableOutputDir / "hogLogisticRegressionTestPredictions.csv"

    joblib.dump(bestModelPipeline, modelPath)

    with open(configPath, "w", encoding="utf-8") as jsonFile:
        json.dump(baselineConfig, jsonFile, indent=2)

    tuningSummaryDf.to_csv(tuningSummaryPath, index=False)
    metricSummaryDf.to_csv(metricSummaryPath, index=False)
    valPredictionDf.to_csv(valPredictionPath, index=False)
    testPredictionDf.to_csv(testPredictionPath, index=False)

    return {
        "modelPath": str(modelPath),
        "configPath": str(configPath),
        "tuningSummaryPath": str(tuningSummaryPath),
        "metricSummaryPath": str(metricSummaryPath),
        "valPredictionPath": str(valPredictionPath),
        "testPredictionPath": str(testPredictionPath),
    }


def runTunedHogLogisticRegressionBaseline(
    projectConfig: ProjectConfig,
    searchSpace: dict[str, list[Any]] | None = None,
    logisticTol: float = 1e-3,
    logisticMaxIter: int = 2000,
    verbose: bool = True,
) -> dict[str, Any]:
    """We tune the HOG plus Logistic Regression baseline on validation macro F1 and save only the selected best artifacts."""
    if searchSpace is None:
        searchSpace = getDefaultTuningSearchSpace()

    searchSpace = normalizeTuningSearchSpace(searchSpace)
    validateTuningSearchSpace(searchSpace)

    modelingManifestDf = loadStrictModelingManifest(projectConfig)
    classToIndexMap, indexToClassMap = buildClassMaps(projectConfig.classNames)

    featureCacheDict = buildTuningFeatureCache(
        modelingManifestDf=modelingManifestDf,
        classToIndexMap=classToIndexMap,
        searchSpace=searchSpace,
        verbose=verbose,
    )

    tuningSummaryDf = buildTuningSummaryDf(
        featureCacheDict=featureCacheDict,
        searchSpace=searchSpace,
        seed=projectConfig.seed,
        logisticTol=logisticTol,
        logisticMaxIter=logisticMaxIter,
        verbose=verbose,
    )

    bestTrialRow = tuningSummaryDf.iloc[0].to_dict()
    bestFeatureCacheKey = str(bestTrialRow["featureCacheKey"])
    bestFeatureEntry = featureCacheDict[bestFeatureCacheKey]

    bestImageSize = int(bestTrialRow["imageSize"])
    bestPixelsPerCell = (
        int(bestTrialRow["pixelsPerCellX"]),
        int(bestTrialRow["pixelsPerCellY"]),
    )
    bestOrientations = int(bestTrialRow["orientations"])
    bestColorMode = str(bestTrialRow["colorMode"])
    bestCValue = float(bestTrialRow["cValue"])

    bestHogParams = buildHogParamDict(
        orientations=bestOrientations,
        pixelsPerCell=bestPixelsPerCell,
    )

    trainFeatures = bestFeatureEntry["trainFeatures"]
    trainTargets = bestFeatureEntry["trainTargets"]
    trainDf = bestFeatureEntry["trainDf"]

    valFeatures = bestFeatureEntry["valFeatures"]
    valTargets = bestFeatureEntry["valTargets"]
    valDf = bestFeatureEntry["valDf"]

    testFeatures, testTargets, testDf = buildSubsetFeatures(
        modelingManifestDf=modelingManifestDf,
        subsetName="test",
        classToIndexMap=classToIndexMap,
        imageSize=bestImageSize,
        hogParams=bestHogParams,
        colorMode=bestColorMode,
    )

    bestModelPipeline, finalFitMeta = fitLogisticRegressionModel(
        trainFeatures=trainFeatures,
        trainTargets=trainTargets,
        seed=projectConfig.seed,
        cValue=bestCValue,
        tol=float(logisticTol),
        maxIter=int(logisticMaxIter),
    )

    trainPredictedIndices, trainScoreMatrix = predictWithLogisticRegressionModel(
        modelPipeline=bestModelPipeline,
        featureMatrix=trainFeatures,
    )

    valPredictedIndices, valScoreMatrix = predictWithLogisticRegressionModel(
        modelPipeline=bestModelPipeline,
        featureMatrix=valFeatures,
    )

    testPredictedIndices, testScoreMatrix = predictWithLogisticRegressionModel(
        modelPipeline=bestModelPipeline,
        featureMatrix=testFeatures,
    )

    trainPredictionDf = buildPredictionTable(
        subsetDf=trainDf,
        targetArray=trainTargets,
        predictedClassIndices=trainPredictedIndices,
        scoreMatrix=trainScoreMatrix,
        indexToClassMap=indexToClassMap,
    )

    valPredictionDf = buildPredictionTable(
        subsetDf=valDf,
        targetArray=valTargets,
        predictedClassIndices=valPredictedIndices,
        scoreMatrix=valScoreMatrix,
        indexToClassMap=indexToClassMap,
    )

    testPredictionDf = buildPredictionTable(
        subsetDf=testDf,
        targetArray=testTargets,
        predictedClassIndices=testPredictedIndices,
        scoreMatrix=testScoreMatrix,
        indexToClassMap=indexToClassMap,
    )

    trainMetricSummary = buildMetricSummary(
        trueLabels=trainPredictionDf["className"],
        predictedLabels=trainPredictionDf["predictedClassName"],
    )

    valMetricSummary = buildMetricSummary(
        trueLabels=valPredictionDf["className"],
        predictedLabels=valPredictionDf["predictedClassName"],
    )

    testMetricSummary = buildMetricSummary(
        trueLabels=testPredictionDf["className"],
        predictedLabels=testPredictionDf["predictedClassName"],
    )

    metricSummaryDf = buildBestMetricSummaryDf(
        trainMetricSummary=trainMetricSummary,
        valMetricSummary=valMetricSummary,
        testMetricSummary=testMetricSummary,
    )

    baselineConfig = {
        "modelName": "hogLogisticRegression",
        "selectionMetric": "valMacroF1",
        "seed": int(projectConfig.seed),
        "logisticTol": float(logisticTol),
        "logisticMaxIter": int(logisticMaxIter),
        "searchSpace": searchSpace,
        "bestConfig": {
            "imageSize": bestImageSize,
            "cValue": bestCValue,
            "orientations": bestOrientations,
            "pixelsPerCell": [int(bestPixelsPerCell[0]), int(bestPixelsPerCell[1])],
            "cellsPerBlock": [2, 2],
            "blockNorm": "L2-Hys",
            "colorMode": bestColorMode,
        },
        "bestTrialMeta": {
            "rankBySelection": int(bestTrialRow["rankBySelection"]),
            "hadConvergenceWarning": int(bestTrialRow["hadConvergenceWarning"]),
            "convergenceWarningCount": int(bestTrialRow["convergenceWarningCount"]),
            "isSelectionEligible": int(bestTrialRow["isSelectionEligible"]),
        },
        "finalRefitMeta": finalFitMeta,
        "featureDimension": int(trainFeatures.shape[1]),
        "bestTrainMetrics": trainMetricSummary,
        "bestValMetrics": valMetricSummary,
        "bestTestMetrics": testMetricSummary,
    }

    savedPathDict = saveBaselineArtifacts(
        projectConfig=projectConfig,
        bestModelPipeline=bestModelPipeline,
        tuningSummaryDf=tuningSummaryDf,
        metricSummaryDf=metricSummaryDf,
        valPredictionDf=valPredictionDf,
        testPredictionDf=testPredictionDf,
        baselineConfig=baselineConfig,
    )

    return {
        "bestModelPipeline": bestModelPipeline,
        "tuningSummaryDf": tuningSummaryDf,
        "metricSummaryDf": metricSummaryDf,
        "bestTrialRow": bestTrialRow,
        "baselineConfig": baselineConfig,
        "trainPredictionDf": trainPredictionDf,
        "valPredictionDf": valPredictionDf,
        "testPredictionDf": testPredictionDf,
        "savedPathDict": savedPathDict,
    }
