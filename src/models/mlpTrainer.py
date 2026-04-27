from __future__ import annotations

"""We train, tune, evaluate, and save one clean HOG-MLP baseline on the strict crop manifest."""

from dataclasses import asdict, dataclass
import json
import pickle
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.projectConfig import ProjectConfig
from src.data import filterManifestBySubset, loadModelingManifest
from src.data.cropClassificationDataset import buildClassToIndexMap
from src.features.imageTransforms import getBaselineTransform
from src.models.mlpModel import MlpClassifier, MlpConfig, buildDefaultMlpConfig


@dataclass(frozen=True)
class MlpTrainingConfig:
    """We store the HOG extraction and optimization hyperparameters for one shared MLP run."""

    imageSize: int = 144
    pixelsPerCell: tuple[int, int] = (16, 16)
    orientations: int = 9
    colorMode: str = "rgb"
    batchSize: int = 64
    learningRate: float = 1e-3
    weightDecay: float = 1e-3
    numEpochs: int = 60
    earlyStoppingPatience: int = 10
    labelSmoothing: float = 0.0
    useClassWeights: bool = True
    randomSeed: int = 27


@dataclass(frozen=True)
class MlpTrialSpec:
    """We store one complete tunable HOG-MLP trial specification."""

    trialName: str = "hogMlpTrial"
    hiddenDims: tuple[int, ...] = (512, 256)
    dropoutRate: float = 0.30
    useBatchNorm: bool = True
    imageSize: int = 144
    pixelsPerCell: tuple[int, int] = (16, 16)
    orientations: int = 9
    colorMode: str = "rgb"
    batchSize: int = 64
    learningRate: float = 1e-3
    weightDecay: float = 1e-3
    numEpochs: int = 60
    earlyStoppingPatience: int = 10
    labelSmoothing: float = 0.0
    useClassWeights: bool = True
    randomSeed: int = 27


def buildDefaultMlpTrainingConfig(projectConfig: ProjectConfig) -> MlpTrainingConfig:
    """We build one strong default HOG-MLP training setup from the shared project config."""
    return MlpTrainingConfig(
        imageSize=144,
        pixelsPerCell=(16, 16),
        orientations=9,
        colorMode="rgb",
        batchSize=64,
        learningRate=1e-3,
        weightDecay=1e-3,
        numEpochs=60,
        earlyStoppingPatience=10,
        labelSmoothing=0.0,
        useClassWeights=True,
        randomSeed=projectConfig.seed,
    )


def buildDefaultMlpTrialSpec(projectConfig: ProjectConfig) -> MlpTrialSpec:
    """We build one strong default HOG-MLP trial spec for notebook-driven tuning."""
    return MlpTrialSpec(
        trialName="hogMlpDefault",
        hiddenDims=(512, 256),
        dropoutRate=0.30,
        useBatchNorm=True,
        imageSize=144,
        pixelsPerCell=(16, 16),
        orientations=9,
        colorMode="rgb",
        batchSize=64,
        learningRate=1e-3,
        weightDecay=1e-3,
        numEpochs=60,
        earlyStoppingPatience=10,
        labelSmoothing=0.0,
        useClassWeights=True,
        randomSeed=projectConfig.seed,
    )


def buildTrainingConfigFromTrialSpec(trialSpec: MlpTrialSpec) -> MlpTrainingConfig:
    """We convert one full trial spec into the training configuration used by the trainer."""
    return MlpTrainingConfig(
        imageSize=trialSpec.imageSize,
        pixelsPerCell=trialSpec.pixelsPerCell,
        orientations=trialSpec.orientations,
        colorMode=trialSpec.colorMode,
        batchSize=trialSpec.batchSize,
        learningRate=trialSpec.learningRate,
        weightDecay=trialSpec.weightDecay,
        numEpochs=trialSpec.numEpochs,
        earlyStoppingPatience=trialSpec.earlyStoppingPatience,
        labelSmoothing=trialSpec.labelSmoothing,
        useClassWeights=trialSpec.useClassWeights,
        randomSeed=trialSpec.randomSeed,
    )


def buildModelConfigFromTrialSpec(
    projectConfig: ProjectConfig,
    inputFeatureDim: int,
    trialSpec: MlpTrialSpec,
) -> MlpConfig:
    """We convert one full trial spec into the final MLP architecture config."""
    return MlpConfig(
        inputFeatureDim=int(inputFeatureDim),
        numClasses=len(projectConfig.classNames),
        hiddenDims=tuple(trialSpec.hiddenDims),
        dropoutRate=float(trialSpec.dropoutRate),
        useBatchNorm=bool(trialSpec.useBatchNorm),
    )


def setTrialSeed(seed: int) -> None:
    """We reset the random seeds so each trial starts reproducibly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def buildHogParamDict(trainingConfig: MlpTrainingConfig) -> dict[str, Any]:
    """We collect the fixed HOG extraction settings for one MLP run."""
    if trainingConfig.colorMode not in {"gray", "rgb"}:
        raise ValueError(f"colorMode must be 'gray' or 'rgb', got: {trainingConfig.colorMode}")

    return {
        "imageSize": int(trainingConfig.imageSize),
        "pixelsPerCell": tuple(trainingConfig.pixelsPerCell),
        "orientations": int(trainingConfig.orientations),
        "colorMode": str(trainingConfig.colorMode),
        "cellsPerBlock": (2, 2),
        "blockNorm": "L2-Hys",
    }


def buildFeatureCacheKey(trainingConfig: MlpTrainingConfig) -> tuple[Any, ...]:
    """We build a stable cache key for one HOG feature setup."""
    return (
        int(trainingConfig.imageSize),
        tuple(trainingConfig.pixelsPerCell),
        int(trainingConfig.orientations),
        str(trainingConfig.colorMode),
    )


def preprocessBaselineImage(imagePath: Path, hogParamDict: dict[str, Any]) -> np.ndarray:
    """We load one crop, apply the shared baseline preprocessing, and return a numpy image array."""
    convertToGrayscale = hogParamDict["colorMode"] == "gray"
    transformObject = getBaselineTransform(
        imageSize=hogParamDict["imageSize"],
        convertToGrayscale=convertToGrayscale,
    )

    imageObject = Image.open(imagePath).convert("RGB")
    transformedImage = transformObject(imageObject)

    imageArray = np.asarray(transformedImage, dtype=np.float32)

    if convertToGrayscale and imageArray.ndim == 3:
        imageArray = np.squeeze(imageArray, axis=-1)

    return imageArray


def extractHogFeatureVector(imagePath: Path, hogParamDict: dict[str, Any]) -> np.ndarray:
    """We extract one HOG feature vector from a preprocessed crop image."""
    imageArray = preprocessBaselineImage(imagePath=imagePath, hogParamDict=hogParamDict)

    if hogParamDict["colorMode"] == "rgb":
        featureVector = hog(
            imageArray,
            orientations=hogParamDict["orientations"],
            pixels_per_cell=hogParamDict["pixelsPerCell"],
            cells_per_block=hogParamDict["cellsPerBlock"],
            block_norm=hogParamDict["blockNorm"],
            feature_vector=True,
            channel_axis=-1,
        )
    else:
        featureVector = hog(
            imageArray,
            orientations=hogParamDict["orientations"],
            pixels_per_cell=hogParamDict["pixelsPerCell"],
            cells_per_block=hogParamDict["cellsPerBlock"],
            block_norm=hogParamDict["blockNorm"],
            feature_vector=True,
        )

    return np.asarray(featureVector, dtype=np.float32)


def buildSubsetFeatures(
    subsetDf: pd.DataFrame,
    classToIndexMap: dict[str, int],
    hogParamDict: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """We build the HOG feature matrix, label vector, and aligned metadata for one subset."""
    featureList: list[np.ndarray] = []
    labelList: list[int] = []
    metadataRowList: list[dict[str, Any]] = []

    for _, sampleRow in subsetDf.iterrows():
        imagePath = Path(str(sampleRow["cropPath"]))
        featureVector = extractHogFeatureVector(imagePath=imagePath, hogParamDict=hogParamDict)

        className = str(sampleRow["className"])
        classIndex = classToIndexMap[className]

        featureList.append(featureVector)
        labelList.append(classIndex)
        metadataRowList.append(
            {
                "cropFileName": str(sampleRow["cropFileName"]),
                "subset": str(sampleRow["subset"]),
                "sourceImageFileName": str(sampleRow["sourceImageFileName"]),
                "sourceImageStem": str(sampleRow["sourceImageStem"]),
                "trueClassIndex": int(classIndex),
                "trueClassName": className,
                "cropWidth": int(sampleRow["cropWidth"]),
                "cropHeight": int(sampleRow["cropHeight"]),
                "hasXmlFilenameMismatch": int(sampleRow["hasXmlFilenameMismatch"]),
                "isExactDuplicateImage": int(sampleRow["isExactDuplicateImage"]),
                "isCrossSplitDuplicateImage": int(sampleRow["isCrossSplitDuplicateImage"]),
            }
        )

    featureMatrix = np.vstack(featureList).astype(np.float32)
    labelArray = np.asarray(labelList, dtype=np.int64)
    metadataDf = pd.DataFrame(metadataRowList)

    return featureMatrix, labelArray, metadataDf


def buildFeatureBundle(
    projectConfig: ProjectConfig,
    trainingConfig: MlpTrainingConfig,
) -> dict[str, Any]:
    """We build and standardize the strict train, validation, and test HOG feature bundles."""
    strictManifestPath = projectConfig.manifestsDir / "modelingManifestStrict.csv"
    modelingManifestDf = loadModelingManifest(strictManifestPath)

    trainDf = filterManifestBySubset(modelingManifestDf, subsetName="train")
    valDf = filterManifestBySubset(modelingManifestDf, subsetName="val")
    testDf = filterManifestBySubset(modelingManifestDf, subsetName="test")

    classToIndexMap = buildClassToIndexMap(projectConfig.classNames)
    hogParamDict = buildHogParamDict(trainingConfig)

    trainFeatureMatrix, trainLabelArray, trainMetadataDf = buildSubsetFeatures(
        subsetDf=trainDf,
        classToIndexMap=classToIndexMap,
        hogParamDict=hogParamDict,
    )
    valFeatureMatrix, valLabelArray, valMetadataDf = buildSubsetFeatures(
        subsetDf=valDf,
        classToIndexMap=classToIndexMap,
        hogParamDict=hogParamDict,
    )
    testFeatureMatrix, testLabelArray, testMetadataDf = buildSubsetFeatures(
        subsetDf=testDf,
        classToIndexMap=classToIndexMap,
        hogParamDict=hogParamDict,
    )

    scalerObject = StandardScaler()
    trainFeatureMatrix = scalerObject.fit_transform(trainFeatureMatrix).astype(np.float32)
    valFeatureMatrix = scalerObject.transform(valFeatureMatrix).astype(np.float32)
    testFeatureMatrix = scalerObject.transform(testFeatureMatrix).astype(np.float32)

    return {
        "hogParamDict": hogParamDict,
        "scalerObject": scalerObject,
        "trainFeatureMatrix": trainFeatureMatrix,
        "trainLabelArray": trainLabelArray,
        "trainMetadataDf": trainMetadataDf,
        "valFeatureMatrix": valFeatureMatrix,
        "valLabelArray": valLabelArray,
        "valMetadataDf": valMetadataDf,
        "testFeatureMatrix": testFeatureMatrix,
        "testLabelArray": testLabelArray,
        "testMetadataDf": testMetadataDf,
    }


def computeClassificationMetrics(
    trueLabels: list[int],
    predictedLabels: list[int],
) -> dict[str, float]:
    """We compute the main multiclass metrics used across the project."""
    accuracy = accuracy_score(trueLabels, predictedLabels)

    macroPrecision, macroRecall, macroF1, _ = precision_recall_fscore_support(
        trueLabels,
        predictedLabels,
        average="macro",
        zero_division=0,
    )
    weightedPrecision, weightedRecall, weightedF1, _ = precision_recall_fscore_support(
        trueLabels,
        predictedLabels,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macroPrecision": float(macroPrecision),
        "macroRecall": float(macroRecall),
        "macroF1": float(macroF1),
        "weightedPrecision": float(weightedPrecision),
        "weightedRecall": float(weightedRecall),
        "weightedF1": float(weightedF1),
    }


def buildTensorDataLoader(
    featureMatrix: np.ndarray,
    labelArray: np.ndarray,
    batchSize: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """We wrap one feature matrix and label vector into a deterministic tensor data loader."""
    featureTensor = torch.tensor(featureMatrix, dtype=torch.float32)
    labelTensor = torch.tensor(labelArray, dtype=torch.long)

    datasetObject = TensorDataset(featureTensor, labelTensor)

    dataLoaderKwargs: dict[str, Any] = {
        "dataset": datasetObject,
        "batch_size": batchSize,
        "shuffle": shuffle,
        "num_workers": 0,
    }

    if shuffle:
        generatorObject = torch.Generator()
        generatorObject.manual_seed(seed)
        dataLoaderKwargs["generator"] = generatorObject

    return DataLoader(**dataLoaderKwargs)


def computeClassWeightTensor(
    labelArray: np.ndarray,
    numClasses: int,
    device: torch.device | str,
) -> torch.Tensor:
    """We compute inverse-frequency class weights from the training labels."""
    classCounts = np.bincount(labelArray, minlength=numClasses).astype(np.float32)
    classCounts = np.maximum(classCounts, 1.0)

    classWeightArray = len(labelArray) / (numClasses * classCounts)
    classWeightTensor = torch.tensor(classWeightArray, dtype=torch.float32, device=device)

    return classWeightTensor


def runOneTrainingEpoch(
    model: MlpClassifier,
    dataLoader: DataLoader,
    lossFunction: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> tuple[float, dict[str, float]]:
    """We run one full training epoch on HOG features and return the average loss and metrics."""
    model.train()

    runningLoss = 0.0
    totalSamples = 0
    trueLabels: list[int] = []
    predictedLabels: list[int] = []

    for featureTensor, labelTensor in dataLoader:
        featureTensor = featureTensor.to(device)
        labelTensor = labelTensor.to(device)

        optimizer.zero_grad()

        logits = model(featureTensor)
        lossValue = lossFunction(logits, labelTensor)
        lossValue.backward()
        optimizer.step()

        batchSize = featureTensor.size(0)
        runningLoss += float(lossValue.item()) * batchSize
        totalSamples += batchSize

        predictedTensor = torch.argmax(logits, dim=1)
        trueLabels.extend(labelTensor.detach().cpu().tolist())
        predictedLabels.extend(predictedTensor.detach().cpu().tolist())

    averageLoss = runningLoss / max(totalSamples, 1)
    metricDict = computeClassificationMetrics(trueLabels, predictedLabels)

    return averageLoss, metricDict


@torch.no_grad()
def runOneEvaluationEpoch(
    model: MlpClassifier,
    dataLoader: DataLoader,
    lossFunction: nn.Module,
    device: torch.device | str,
) -> tuple[float, dict[str, float], list[int], list[int]]:
    """We run one full evaluation epoch on HOG features and return loss, metrics, and labels."""
    model.eval()

    runningLoss = 0.0
    totalSamples = 0
    trueLabels: list[int] = []
    predictedLabels: list[int] = []

    for featureTensor, labelTensor in dataLoader:
        featureTensor = featureTensor.to(device)
        labelTensor = labelTensor.to(device)

        logits = model(featureTensor)
        lossValue = lossFunction(logits, labelTensor)

        batchSize = featureTensor.size(0)
        runningLoss += float(lossValue.item()) * batchSize
        totalSamples += batchSize

        predictedTensor = torch.argmax(logits, dim=1)
        trueLabels.extend(labelTensor.detach().cpu().tolist())
        predictedLabels.extend(predictedTensor.detach().cpu().tolist())

    averageLoss = runningLoss / max(totalSamples, 1)
    metricDict = computeClassificationMetrics(trueLabels, predictedLabels)

    return averageLoss, metricDict, trueLabels, predictedLabels


@torch.no_grad()
def collectPredictionRows(
    model: MlpClassifier,
    featureMatrix: np.ndarray,
    metadataDf: pd.DataFrame,
    classNames: list[str],
    batchSize: int,
    device: torch.device | str,
) -> pd.DataFrame:
    """We collect aligned per-sample predictions from one feature matrix and metadata table."""
    model.eval()

    featureTensor = torch.tensor(featureMatrix, dtype=torch.float32)
    dataLoader = DataLoader(featureTensor, batch_size=batchSize, shuffle=False, num_workers=0)

    predictedIndexList: list[int] = []

    for batchTensor in dataLoader:
        batchTensor = batchTensor.to(device)
        logits = model(batchTensor)
        predictedTensor = torch.argmax(logits, dim=1).detach().cpu().tolist()
        predictedIndexList.extend(predictedTensor)

    predictionDf = metadataDf.copy().reset_index(drop=True)
    predictionDf["predictedClassIndex"] = predictedIndexList
    predictionDf["predictedClassName"] = [
        classNames[predictedIndex]
        for predictedIndex in predictedIndexList
    ]

    return predictionDf


def getMlpArtifactPaths(projectConfig: ProjectConfig) -> dict[str, Path]:
    """We define the fixed overwrite paths for the single saved HOG-MLP artifact set."""
    modelDir = projectConfig.outputsDir / "models" / "mlp"
    metricDir = projectConfig.metricsDir / "mlp"
    tableDir = projectConfig.tablesDir / "mlp"
    predictionDir = projectConfig.samplePredictionsDir / "mlp"

    modelDir.mkdir(parents=True, exist_ok=True)
    metricDir.mkdir(parents=True, exist_ok=True)
    tableDir.mkdir(parents=True, exist_ok=True)
    predictionDir.mkdir(parents=True, exist_ok=True)

    return {
        "modelPath": modelDir / "mlpClassifier.pt",
        "featureScalerPath": modelDir / "mlpFeatureScaler.pkl",
        "mlpConfigPath": tableDir / "mlpConfig.json",
        "trainingConfigPath": tableDir / "mlpTrainingConfig.json",
        "tuningSummaryPath": tableDir / "mlpTuningSummary.csv",
        "historyPath": metricDir / "mlpTrainingHistory.csv",
        "metricSummaryPath": metricDir / "mlpMetricSummary.csv",
        "valPredictionsPath": predictionDir / "mlpValPredictions.csv",
        "testPredictionsPath": predictionDir / "mlpTestPredictions.csv",
    }


def saveMlpArtifacts(
    projectConfig: ProjectConfig,
    model: MlpClassifier,
    scalerObject: StandardScaler,
    mlpConfig: MlpConfig,
    trainingConfig: MlpTrainingConfig,
    historyDf: pd.DataFrame,
    metricSummaryDf: pd.DataFrame,
    valPredictionDf: pd.DataFrame,
    testPredictionDf: pd.DataFrame,
) -> dict[str, Path]:
    """We save the fixed final HOG-MLP artifacts and overwrite the previous MLP run."""
    artifactPathDict = getMlpArtifactPaths(projectConfig)

    torch.save(model.state_dict(), artifactPathDict["modelPath"])

    with open(artifactPathDict["featureScalerPath"], "wb") as fileObject:
        pickle.dump(scalerObject, fileObject)

    with open(artifactPathDict["mlpConfigPath"], "w", encoding="utf-8") as fileObject:
        json.dump(asdict(mlpConfig), fileObject, indent=2)

    with open(artifactPathDict["trainingConfigPath"], "w", encoding="utf-8") as fileObject:
        json.dump(asdict(trainingConfig), fileObject, indent=2)

    historyDf.to_csv(artifactPathDict["historyPath"], index=False)
    metricSummaryDf.to_csv(artifactPathDict["metricSummaryPath"], index=False)
    valPredictionDf.to_csv(artifactPathDict["valPredictionsPath"], index=False)
    testPredictionDf.to_csv(artifactPathDict["testPredictionsPath"], index=False)

    return artifactPathDict


def printEpochLine(
    trialName: str,
    epochIndex: int,
    numEpochs: int,
    learningRate: float,
    trainLoss: float,
    valLoss: float,
    trainMetricDict: dict[str, float],
    valMetricDict: dict[str, float],
    bestValMacroF1: float,
) -> None:
    """We print one concise live epoch line for notebook-visible training progress."""
    print(
        f"[{trialName}] "
        f"epoch={epochIndex:03d}/{numEpochs:03d} | "
        f"lr={learningRate:.6f} | "
        f"trainLoss={trainLoss:.4f} | "
        f"valLoss={valLoss:.4f} | "
        f"trainMacroF1={trainMetricDict['macroF1']:.4f} | "
        f"valMacroF1={valMetricDict['macroF1']:.4f} | "
        f"bestValMacroF1={bestValMacroF1:.4f}"
    )


def extractSplitMetric(metricSummaryDf: pd.DataFrame, splitName: str, metricName: str) -> float:
    """We extract one scalar metric from the split-level metric summary table."""
    splitSeries = metricSummaryDf.loc[
        metricSummaryDf["split"] == splitName,
        metricName,
    ]

    if splitSeries.empty:
        raise ValueError(f"Missing split='{splitName}' or metric='{metricName}' in metric summary.")

    return float(splitSeries.iloc[0])


def runMlpExperiment(
    projectConfig: ProjectConfig,
    mlpConfig: MlpConfig | None = None,
    trainingConfig: MlpTrainingConfig | None = None,
    featureBundle: dict[str, Any] | None = None,
    saveArtifacts: bool = True,
    verbose: bool = True,
    trialName: str = "hogMlp",
) -> dict[str, Any]:
    """We run one clean end-to-end HOG-MLP experiment on the strict modeling manifest."""
    if trainingConfig is None:
        trainingConfig = buildDefaultMlpTrainingConfig(projectConfig)

    if featureBundle is None:
        featureBundle = buildFeatureBundle(
            projectConfig=projectConfig,
            trainingConfig=trainingConfig,
        )

    setTrialSeed(trainingConfig.randomSeed)

    trainFeatureMatrix = featureBundle["trainFeatureMatrix"]
    trainLabelArray = featureBundle["trainLabelArray"]
    trainMetadataDf = featureBundle["trainMetadataDf"]
    valFeatureMatrix = featureBundle["valFeatureMatrix"]
    valLabelArray = featureBundle["valLabelArray"]
    valMetadataDf = featureBundle["valMetadataDf"]
    testFeatureMatrix = featureBundle["testFeatureMatrix"]
    testLabelArray = featureBundle["testLabelArray"]
    testMetadataDf = featureBundle["testMetadataDf"]
    hogParamDict = featureBundle["hogParamDict"]
    scalerObject = featureBundle["scalerObject"]

    if mlpConfig is None:
        mlpConfig = buildDefaultMlpConfig(
            projectConfig=projectConfig,
            inputFeatureDim=int(trainFeatureMatrix.shape[1]),
        )

    trainLoader = buildTensorDataLoader(
        featureMatrix=trainFeatureMatrix,
        labelArray=trainLabelArray,
        batchSize=trainingConfig.batchSize,
        shuffle=True,
        seed=trainingConfig.randomSeed,
    )
    valLoader = buildTensorDataLoader(
        featureMatrix=valFeatureMatrix,
        labelArray=valLabelArray,
        batchSize=trainingConfig.batchSize,
        shuffle=False,
        seed=trainingConfig.randomSeed,
    )
    testLoader = buildTensorDataLoader(
        featureMatrix=testFeatureMatrix,
        labelArray=testLabelArray,
        batchSize=trainingConfig.batchSize,
        shuffle=False,
        seed=trainingConfig.randomSeed,
    )

    model = MlpClassifier(mlpConfig).to(projectConfig.device)

    if trainingConfig.useClassWeights:
        classWeightTensor = computeClassWeightTensor(
            labelArray=trainLabelArray,
            numClasses=len(projectConfig.classNames),
            device=projectConfig.device,
        )
    else:
        classWeightTensor = None

    lossFunction = nn.CrossEntropyLoss(
        weight=classWeightTensor,
        label_smoothing=trainingConfig.labelSmoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainingConfig.learningRate,
        weight_decay=trainingConfig.weightDecay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    historyRowList: list[dict[str, Any]] = []
    bestValMacroF1 = float("-inf")
    bestEpochIndex = -1
    bestStateDict: dict[str, torch.Tensor] | None = None
    epochsWithoutImprovement = 0

    if verbose:
        print(
            f"[{trialName}] start | "
            f"hog(imageSize={trainingConfig.imageSize}, pixelsPerCell={trainingConfig.pixelsPerCell}, "
            f"orientations={trainingConfig.orientations}, colorMode={trainingConfig.colorMode}) | "
            f"mlp(hiddenDims={mlpConfig.hiddenDims}, dropoutRate={mlpConfig.dropoutRate}, "
            f"useBatchNorm={mlpConfig.useBatchNorm}) | "
            f"opt(batchSize={trainingConfig.batchSize}, learningRate={trainingConfig.learningRate}, "
            f"weightDecay={trainingConfig.weightDecay}, labelSmoothing={trainingConfig.labelSmoothing}, "
            f"useClassWeights={trainingConfig.useClassWeights})"
        )

    for epochIndex in range(trainingConfig.numEpochs):
        trainLoss, trainMetricDict = runOneTrainingEpoch(
            model=model,
            dataLoader=trainLoader,
            lossFunction=lossFunction,
            optimizer=optimizer,
            device=projectConfig.device,
        )
        valLoss, valMetricDict, _, _ = runOneEvaluationEpoch(
            model=model,
            dataLoader=valLoader,
            lossFunction=lossFunction,
            device=projectConfig.device,
        )

        currentLearningRate = float(optimizer.param_groups[0]["lr"])
        scheduler.step(valMetricDict["macroF1"])

        currentBestValMacroF1 = max(bestValMacroF1, valMetricDict["macroF1"])

        historyRowList.append(
            {
                "epoch": epochIndex + 1,
                "learningRate": currentLearningRate,
                "trainLoss": trainLoss,
                "valLoss": valLoss,
                "trainAccuracy": trainMetricDict["accuracy"],
                "trainMacroPrecision": trainMetricDict["macroPrecision"],
                "trainMacroRecall": trainMetricDict["macroRecall"],
                "trainMacroF1": trainMetricDict["macroF1"],
                "trainWeightedF1": trainMetricDict["weightedF1"],
                "valAccuracy": valMetricDict["accuracy"],
                "valMacroPrecision": valMetricDict["macroPrecision"],
                "valMacroRecall": valMetricDict["macroRecall"],
                "valMacroF1": valMetricDict["macroF1"],
                "valWeightedF1": valMetricDict["weightedF1"],
            }
        )

        if verbose:
            printEpochLine(
                trialName=trialName,
                epochIndex=epochIndex + 1,
                numEpochs=trainingConfig.numEpochs,
                learningRate=currentLearningRate,
                trainLoss=trainLoss,
                valLoss=valLoss,
                trainMetricDict=trainMetricDict,
                valMetricDict=valMetricDict,
                bestValMacroF1=currentBestValMacroF1,
            )

        if valMetricDict["macroF1"] > bestValMacroF1:
            bestValMacroF1 = valMetricDict["macroF1"]
            bestEpochIndex = epochIndex + 1
            bestStateDict = {
                parameterName: parameterTensor.detach().cpu().clone()
                for parameterName, parameterTensor in model.state_dict().items()
            }
            epochsWithoutImprovement = 0
        else:
            epochsWithoutImprovement += 1

        if epochsWithoutImprovement >= trainingConfig.earlyStoppingPatience:
            if verbose:
                print(
                    f"[{trialName}] earlyStop | "
                    f"bestEpoch={bestEpochIndex} | bestValMacroF1={bestValMacroF1:.4f}"
                )
            break

    if bestStateDict is None:
        raise RuntimeError("We failed to store a best HOG-MLP checkpoint during training.")

    model.load_state_dict(bestStateDict)
    model = model.to(projectConfig.device)

    trainLoss, trainMetricDict, _, _ = runOneEvaluationEpoch(
        model=model,
        dataLoader=trainLoader,
        lossFunction=lossFunction,
        device=projectConfig.device,
    )
    valLoss, valMetricDict, _, _ = runOneEvaluationEpoch(
        model=model,
        dataLoader=valLoader,
        lossFunction=lossFunction,
        device=projectConfig.device,
    )
    testLoss, testMetricDict, _, _ = runOneEvaluationEpoch(
        model=model,
        dataLoader=testLoader,
        lossFunction=lossFunction,
        device=projectConfig.device,
    )

    metricSummaryDf = pd.DataFrame(
        [
            {"split": "train", "loss": trainLoss, **trainMetricDict},
            {"split": "val", "loss": valLoss, **valMetricDict},
            {"split": "test", "loss": testLoss, **testMetricDict},
        ]
    )

    historyDf = pd.DataFrame(historyRowList)

    valPredictionDf = collectPredictionRows(
        model=model,
        featureMatrix=valFeatureMatrix,
        metadataDf=valMetadataDf,
        classNames=projectConfig.classNames,
        batchSize=trainingConfig.batchSize,
        device=projectConfig.device,
    )
    testPredictionDf = collectPredictionRows(
        model=model,
        featureMatrix=testFeatureMatrix,
        metadataDf=testMetadataDf,
        classNames=projectConfig.classNames,
        batchSize=trainingConfig.batchSize,
        device=projectConfig.device,
    )

    if saveArtifacts:
        artifactPathDict = saveMlpArtifacts(
            projectConfig=projectConfig,
            model=model,
            scalerObject=scalerObject,
            mlpConfig=mlpConfig,
            trainingConfig=trainingConfig,
            historyDf=historyDf,
            metricSummaryDf=metricSummaryDf,
            valPredictionDf=valPredictionDf,
            testPredictionDf=testPredictionDf,
        )
    else:
        artifactPathDict = getMlpArtifactPaths(projectConfig)

    return {
        "model": model,
        "scalerObject": scalerObject,
        "historyDf": historyDf,
        "metricSummaryDf": metricSummaryDf,
        "valPredictionDf": valPredictionDf,
        "testPredictionDf": testPredictionDf,
        "trainMetadataDf": trainMetadataDf,
        "valMetadataDf": valMetadataDf,
        "testMetadataDf": testMetadataDf,
        "hogParamDict": hogParamDict,
        "mlpConfig": mlpConfig,
        "trainingConfig": trainingConfig,
        "bestEpoch": bestEpochIndex,
        "artifactPathDict": artifactPathDict,
    }


def runMlpHyperparameterSearch(
    projectConfig: ProjectConfig,
    trialSpecList: list[MlpTrialSpec],
    verbose: bool = True,
) -> dict[str, Any]:
    """We run a notebook-defined HOG-MLP search and save only the final best artifact set."""
    if len(trialSpecList) == 0:
        raise ValueError("trialSpecList must contain at least one trial.")

    featureCache: dict[tuple[Any, ...], dict[str, Any]] = {}
    tuningSummaryRowList: list[dict[str, Any]] = []

    bestResult: dict[str, Any] | None = None
    bestTrialSpec: MlpTrialSpec | None = None
    bestValMacroF1 = float("-inf")

    for trialIndex, trialSpec in enumerate(trialSpecList, start=1):
        trainingConfig = buildTrainingConfigFromTrialSpec(trialSpec)
        cacheKey = buildFeatureCacheKey(trainingConfig)

        if cacheKey not in featureCache:
            if verbose:
                print(f"[search] featureBuild | trial={trialIndex}/{len(trialSpecList)} | key={cacheKey}")
            featureCache[cacheKey] = buildFeatureBundle(
                projectConfig=projectConfig,
                trainingConfig=trainingConfig,
            )
        else:
            if verbose:
                print(f"[search] featureReuse | trial={trialIndex}/{len(trialSpecList)} | key={cacheKey}")

        featureBundle = featureCache[cacheKey]
        inputFeatureDim = int(featureBundle["trainFeatureMatrix"].shape[1])

        mlpConfig = buildModelConfigFromTrialSpec(
            projectConfig=projectConfig,
            inputFeatureDim=inputFeatureDim,
            trialSpec=trialSpec,
        )

        if verbose:
            print(
                f"[search] trialStart | index={trialIndex}/{len(trialSpecList)} | "
                f"name={trialSpec.trialName}"
            )

        trialResult = runMlpExperiment(
            projectConfig=projectConfig,
            mlpConfig=mlpConfig,
            trainingConfig=trainingConfig,
            featureBundle=featureBundle,
            saveArtifacts=False,
            verbose=verbose,
            trialName=trialSpec.trialName,
        )

        metricSummaryDf = trialResult["metricSummaryDf"]

        trainMacroF1 = extractSplitMetric(metricSummaryDf, "train", "macroF1")
        valMacroF1 = extractSplitMetric(metricSummaryDf, "val", "macroF1")
        testMacroF1 = extractSplitMetric(metricSummaryDf, "test", "macroF1")
        trainAccuracy = extractSplitMetric(metricSummaryDf, "train", "accuracy")
        valAccuracy = extractSplitMetric(metricSummaryDf, "val", "accuracy")
        testAccuracy = extractSplitMetric(metricSummaryDf, "test", "accuracy")

        tuningSummaryRowList.append(
            {
                "trialName": trialSpec.trialName,
                "hiddenDims": str(trialSpec.hiddenDims),
                "dropoutRate": float(trialSpec.dropoutRate),
                "useBatchNorm": bool(trialSpec.useBatchNorm),
                "imageSize": int(trialSpec.imageSize),
                "pixelsPerCell": str(trialSpec.pixelsPerCell),
                "orientations": int(trialSpec.orientations),
                "colorMode": str(trialSpec.colorMode),
                "batchSize": int(trialSpec.batchSize),
                "learningRate": float(trialSpec.learningRate),
                "weightDecay": float(trialSpec.weightDecay),
                "numEpochs": int(trialSpec.numEpochs),
                "earlyStoppingPatience": int(trialSpec.earlyStoppingPatience),
                "labelSmoothing": float(trialSpec.labelSmoothing),
                "useClassWeights": bool(trialSpec.useClassWeights),
                "bestEpoch": int(trialResult["bestEpoch"]),
                "trainAccuracy": trainAccuracy,
                "valAccuracy": valAccuracy,
                "testAccuracy": testAccuracy,
                "trainMacroF1": trainMacroF1,
                "valMacroF1": valMacroF1,
                "testMacroF1": testMacroF1,
            }
        )

        if verbose:
            print(
                f"[search] trialDone | name={trialSpec.trialName} | "
                f"valMacroF1={valMacroF1:.4f} | testMacroF1={testMacroF1:.4f}"
            )

        if valMacroF1 > bestValMacroF1:
            bestValMacroF1 = valMacroF1
            bestResult = trialResult
            bestTrialSpec = trialSpec

    if bestResult is None or bestTrialSpec is None:
        raise RuntimeError("We failed to select a best MLP trial.")

    tuningSummaryDf = pd.DataFrame(tuningSummaryRowList)
    tuningSummaryDf = tuningSummaryDf.sort_values(
        by=["valMacroF1", "testMacroF1", "valAccuracy"],
        ascending=False,
    ).reset_index(drop=True)

    artifactPathDict = saveMlpArtifacts(
        projectConfig=projectConfig,
        model=bestResult["model"],
        scalerObject=bestResult["scalerObject"],
        mlpConfig=bestResult["mlpConfig"],
        trainingConfig=bestResult["trainingConfig"],
        historyDf=bestResult["historyDf"],
        metricSummaryDf=bestResult["metricSummaryDf"],
        valPredictionDf=bestResult["valPredictionDf"],
        testPredictionDf=bestResult["testPredictionDf"],
    )

    tuningSummaryDf.to_csv(artifactPathDict["tuningSummaryPath"], index=False)

    if verbose:
        print(
            f"[search] bestTrial | name={bestTrialSpec.trialName} | "
            f"valMacroF1={bestValMacroF1:.4f} | "
            f"savedTo={artifactPathDict['modelPath']}"
        )

    bestResult["tuningSummaryDf"] = tuningSummaryDf
    bestResult["bestTrialSpec"] = bestTrialSpec
    bestResult["artifactPathDict"] = artifactPathDict

    return bestResult