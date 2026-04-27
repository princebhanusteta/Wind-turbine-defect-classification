from __future__ import annotations

"""We train, tune, evaluate, and save one clean compact residual CNN on the strict crop manifest."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from src.config.projectConfig import ProjectConfig
from src.data import buildNeuralDataLoaders, buildNeuralDatasets
from src.features.imageTransforms import getCnnEvalTransform, getCnnTrainTransform
from src.models.cnnModel import (
    ResidualCnnClassifier,
    ResidualCnnConfig,
    buildDefaultResidualCnnConfig,
)


@dataclass(frozen=True)
class ResidualCnnTrainingConfig:
    """We store the optimization and input settings for one residual CNN run."""

    imageSize: int = 160
    batchSize: int = 32
    learningRate: float = 3e-4
    weightDecay: float = 1e-4
    numEpochs: int = 80
    earlyStoppingPatience: int = 15
    labelSmoothing: float = 0.0
    useClassWeights: bool = True
    randomSeed: int = 27


@dataclass(frozen=True)
class ResidualCnnTrialSpec:
    """We store one complete tunable residual CNN trial specification."""

    trialName: str = "residualCnnTrial"
    imageSize: int = 160
    batchSize: int = 32
    learningRate: float = 3e-4
    weightDecay: float = 1e-4
    numEpochs: int = 80
    earlyStoppingPatience: int = 15
    labelSmoothing: float = 0.0
    useClassWeights: bool = True
    baseChannels: int = 32
    stageBlockCounts: tuple[int, ...] = (2, 2, 2, 2)
    dropoutRate: float = 0.20
    useBatchNorm: bool = True
    randomSeed: int = 27


def buildDefaultResidualCnnTrainingConfig(
    projectConfig: ProjectConfig,
) -> ResidualCnnTrainingConfig:
    """We build one strong default residual CNN training setup from the shared project config."""
    return ResidualCnnTrainingConfig(
        imageSize=160,
        batchSize=32,
        learningRate=3e-4,
        weightDecay=1e-4,
        numEpochs=80,
        earlyStoppingPatience=15,
        labelSmoothing=0.0,
        useClassWeights=True,
        randomSeed=projectConfig.seed,
    )


def buildDefaultResidualCnnTrialSpec(
    projectConfig: ProjectConfig,
) -> ResidualCnnTrialSpec:
    """We build one strong default residual CNN trial spec for notebook-driven tuning."""
    return ResidualCnnTrialSpec(
        trialName="residualCnnDefault",
        imageSize=160,
        batchSize=32,
        learningRate=3e-4,
        weightDecay=1e-4,
        numEpochs=80,
        earlyStoppingPatience=15,
        labelSmoothing=0.0,
        useClassWeights=True,
        baseChannels=32,
        stageBlockCounts=(2, 2, 2, 2),
        dropoutRate=0.20,
        useBatchNorm=True,
        randomSeed=projectConfig.seed,
    )


def buildTrainingConfigFromTrialSpec(
    trialSpec: ResidualCnnTrialSpec,
) -> ResidualCnnTrainingConfig:
    """We convert one full CNN trial spec into the training configuration used by the trainer."""
    return ResidualCnnTrainingConfig(
        imageSize=trialSpec.imageSize,
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
    trialSpec: ResidualCnnTrialSpec,
) -> ResidualCnnConfig:
    """We convert one full CNN trial spec into the model architecture config."""
    return ResidualCnnConfig(
        inputChannels=3,
        numClasses=len(projectConfig.classNames),
        baseChannels=int(trialSpec.baseChannels),
        stageBlockCounts=tuple(trialSpec.stageBlockCounts),
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


def computeClassWeightTensor(
    datasetObject: Any,
    numClasses: int,
    device: torch.device | str,
) -> torch.Tensor:
    """We compute inverse-frequency class weights from the training dataset labels."""
    labelSeries = datasetObject.modelingManifestDf["className"].astype(str)
    classCounts = np.zeros(numClasses, dtype=np.float32)

    for className, classIndex in datasetObject.classToIndexMap.items():
        classCounts[classIndex] = float((labelSeries == className).sum())

    classCounts = np.maximum(classCounts, 1.0)
    classWeightArray = len(labelSeries) / (numClasses * classCounts)

    classWeightTensor = torch.tensor(
        classWeightArray,
        dtype=torch.float32,
        device=device,
    )

    return classWeightTensor


def runOneTrainingEpoch(
    model: ResidualCnnClassifier,
    dataLoader: DataLoader,
    lossFunction: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> tuple[float, dict[str, float]]:
    """We run one full CNN training epoch and return the average loss and metrics."""
    model.train()

    runningLoss = 0.0
    totalSamples = 0
    trueLabels: list[int] = []
    predictedLabels: list[int] = []

    for imageTensor, labelTensor in dataLoader:
        imageTensor = imageTensor.to(device)
        labelTensor = labelTensor.to(device)

        optimizer.zero_grad()

        logits = model(imageTensor)
        lossValue = lossFunction(logits, labelTensor)
        lossValue.backward()
        optimizer.step()

        batchSize = imageTensor.size(0)
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
    model: ResidualCnnClassifier,
    dataLoader: DataLoader,
    lossFunction: nn.Module,
    device: torch.device | str,
) -> tuple[float, dict[str, float], list[int], list[int]]:
    """We run one full CNN evaluation epoch and return loss, metrics, and labels."""
    model.eval()

    runningLoss = 0.0
    totalSamples = 0
    trueLabels: list[int] = []
    predictedLabels: list[int] = []

    for imageTensor, labelTensor in dataLoader:
        imageTensor = imageTensor.to(device)
        labelTensor = labelTensor.to(device)

        logits = model(imageTensor)
        lossValue = lossFunction(logits, labelTensor)

        batchSize = imageTensor.size(0)
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
    model: ResidualCnnClassifier,
    dataLoader: DataLoader,
    classNames: list[str],
    device: torch.device | str,
) -> pd.DataFrame:
    """We collect per-sample prediction rows for later export and inspection."""
    model.eval()

    predictionRowList: list[dict[str, Any]] = []

    for imageTensor, labelTensor, metadataDict in dataLoader:
        imageTensor = imageTensor.to(device)

        logits = model(imageTensor)
        probabilityTensor = torch.softmax(logits, dim=1).detach().cpu()
        predictedTensor = torch.argmax(probabilityTensor, dim=1)
        confidenceTensor = torch.max(probabilityTensor, dim=1).values
        trueTensor = labelTensor.detach().cpu()

        batchSize = trueTensor.size(0)

        for batchIndex in range(batchSize):
            trueClassIndex = int(trueTensor[batchIndex].item())
            predictedClassIndex = int(predictedTensor[batchIndex].item())
            predictedConfidence = float(confidenceTensor[batchIndex].item())

            predictionRowList.append(
                {
                    "cropFileName": str(metadataDict["cropFileName"][batchIndex]),
                    "subset": str(metadataDict["subset"][batchIndex]),
                    "sourceImageFileName": str(metadataDict["sourceImageFileName"][batchIndex]),
                    "sourceImageStem": str(metadataDict["sourceImageStem"][batchIndex]),
                    "trueClassIndex": trueClassIndex,
                    "predictedClassIndex": predictedClassIndex,
                    "trueClassName": classNames[trueClassIndex],
                    "predictedClassName": classNames[predictedClassIndex],
                    "predictedConfidence": predictedConfidence,
                    "cropWidth": int(metadataDict["cropWidth"][batchIndex]),
                    "cropHeight": int(metadataDict["cropHeight"][batchIndex]),
                    "hasXmlFilenameMismatch": int(metadataDict["hasXmlFilenameMismatch"][batchIndex]),
                    "isExactDuplicateImage": int(metadataDict["isExactDuplicateImage"][batchIndex]),
                    "isCrossSplitDuplicateImage": int(metadataDict["isCrossSplitDuplicateImage"][batchIndex]),
                }
            )

    predictionDf = pd.DataFrame(predictionRowList)
    return predictionDf


def getResidualCnnArtifactPaths(projectConfig: ProjectConfig) -> dict[str, Path]:
    """We define the fixed overwrite paths for the single saved residual CNN artifact set."""
    modelDir = projectConfig.outputsDir / "models" / "residualCnn"
    metricDir = projectConfig.metricsDir / "residualCnn"
    tableDir = projectConfig.tablesDir / "residualCnn"
    predictionDir = projectConfig.samplePredictionsDir / "residualCnn"

    modelDir.mkdir(parents=True, exist_ok=True)
    metricDir.mkdir(parents=True, exist_ok=True)
    tableDir.mkdir(parents=True, exist_ok=True)
    predictionDir.mkdir(parents=True, exist_ok=True)

    return {
        "modelPath": modelDir / "residualCnnClassifier.pt",
        "cnnConfigPath": tableDir / "residualCnnConfig.json",
        "trainingConfigPath": tableDir / "residualCnnTrainingConfig.json",
        "tuningSummaryPath": tableDir / "residualCnnTuningSummary.csv",
        "historyPath": metricDir / "residualCnnTrainingHistory.csv",
        "metricSummaryPath": metricDir / "residualCnnMetricSummary.csv",
        "valPredictionsPath": predictionDir / "residualCnnValPredictions.csv",
        "testPredictionsPath": predictionDir / "residualCnnTestPredictions.csv",
    }


def saveResidualCnnArtifacts(
    projectConfig: ProjectConfig,
    model: ResidualCnnClassifier,
    cnnConfig: ResidualCnnConfig,
    trainingConfig: ResidualCnnTrainingConfig,
    historyDf: pd.DataFrame,
    metricSummaryDf: pd.DataFrame,
    valPredictionDf: pd.DataFrame,
    testPredictionDf: pd.DataFrame,
) -> dict[str, Path]:
    """We save the fixed final residual CNN artifacts and overwrite the previous CNN run."""
    artifactPathDict = getResidualCnnArtifactPaths(projectConfig)

    torch.save(model.state_dict(), artifactPathDict["modelPath"])

    with open(artifactPathDict["cnnConfigPath"], "w", encoding="utf-8") as fileObject:
        json.dump(asdict(cnnConfig), fileObject, indent=2)

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
    """We print one concise live epoch line for notebook-visible CNN training progress."""
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


def runResidualCnnExperiment(
    projectConfig: ProjectConfig,
    cnnConfig: ResidualCnnConfig | None = None,
    trainingConfig: ResidualCnnTrainingConfig | None = None,
    saveArtifacts: bool = True,
    verbose: bool = True,
    trialName: str = "residualCnn",
) -> dict[str, Any]:
    """We run one clean end-to-end residual CNN experiment on the strict modeling manifest."""
    if cnnConfig is None:
        cnnConfig = buildDefaultResidualCnnConfig(projectConfig)

    if trainingConfig is None:
        trainingConfig = buildDefaultResidualCnnTrainingConfig(projectConfig)

    setTrialSeed(trainingConfig.randomSeed)

    strictManifestPath = projectConfig.manifestsDir / "modelingManifestStrict.csv"

    trainTransform = getCnnTrainTransform(imageSize=trainingConfig.imageSize)
    evalTransform = getCnnEvalTransform(imageSize=trainingConfig.imageSize)

    datasetDict = buildNeuralDatasets(
        manifestPath=strictManifestPath,
        classNames=projectConfig.classNames,
        trainTransform=trainTransform,
        evalTransform=evalTransform,
        returnMetadata=False,
    )
    dataLoaderDict = buildNeuralDataLoaders(
        datasetDict=datasetDict,
        batchSize=trainingConfig.batchSize,
        numWorkers=projectConfig.numWorkers,
        seed=trainingConfig.randomSeed,
    )

    predictionDatasetDict = buildNeuralDatasets(
        manifestPath=strictManifestPath,
        classNames=projectConfig.classNames,
        trainTransform=evalTransform,
        evalTransform=evalTransform,
        returnMetadata=True,
    )
    predictionLoaderDict = buildNeuralDataLoaders(
        datasetDict=predictionDatasetDict,
        batchSize=trainingConfig.batchSize,
        numWorkers=projectConfig.numWorkers,
        seed=trainingConfig.randomSeed,
    )

    model = ResidualCnnClassifier(cnnConfig).to(projectConfig.device)

    if trainingConfig.useClassWeights:
        classWeightTensor = computeClassWeightTensor(
            datasetObject=datasetDict["train"],
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
        patience=4,
    )

    historyRowList: list[dict[str, Any]] = []
    bestValMacroF1 = float("-inf")
    bestEpochIndex = -1
    bestStateDict: dict[str, torch.Tensor] | None = None
    epochsWithoutImprovement = 0

    if verbose:
        print(
            f"[{trialName}] start | "
            f"cnn(baseChannels={cnnConfig.baseChannels}, stageBlockCounts={cnnConfig.stageBlockCounts}, "
            f"dropoutRate={cnnConfig.dropoutRate}, useBatchNorm={cnnConfig.useBatchNorm}) | "
            f"opt(imageSize={trainingConfig.imageSize}, batchSize={trainingConfig.batchSize}, "
            f"learningRate={trainingConfig.learningRate}, weightDecay={trainingConfig.weightDecay}, "
            f"labelSmoothing={trainingConfig.labelSmoothing}, useClassWeights={trainingConfig.useClassWeights})"
        )

    for epochIndex in range(trainingConfig.numEpochs):
        trainLoss, trainMetricDict = runOneTrainingEpoch(
            model=model,
            dataLoader=dataLoaderDict["train"],
            lossFunction=lossFunction,
            optimizer=optimizer,
            device=projectConfig.device,
        )
        valLoss, valMetricDict, _, _ = runOneEvaluationEpoch(
            model=model,
            dataLoader=dataLoaderDict["val"],
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
        raise RuntimeError("We failed to store a best residual CNN checkpoint during training.")

    model.load_state_dict(bestStateDict)
    model = model.to(projectConfig.device)

    trainLoss, trainMetricDict, _, _ = runOneEvaluationEpoch(
        model=model,
        dataLoader=dataLoaderDict["train"],
        lossFunction=lossFunction,
        device=projectConfig.device,
    )
    valLoss, valMetricDict, _, _ = runOneEvaluationEpoch(
        model=model,
        dataLoader=dataLoaderDict["val"],
        lossFunction=lossFunction,
        device=projectConfig.device,
    )
    testLoss, testMetricDict, _, _ = runOneEvaluationEpoch(
        model=model,
        dataLoader=dataLoaderDict["test"],
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
        dataLoader=predictionLoaderDict["val"],
        classNames=projectConfig.classNames,
        device=projectConfig.device,
    )
    testPredictionDf = collectPredictionRows(
        model=model,
        dataLoader=predictionLoaderDict["test"],
        classNames=projectConfig.classNames,
        device=projectConfig.device,
    )

    if saveArtifacts:
        artifactPathDict = saveResidualCnnArtifacts(
            projectConfig=projectConfig,
            model=model,
            cnnConfig=cnnConfig,
            trainingConfig=trainingConfig,
            historyDf=historyDf,
            metricSummaryDf=metricSummaryDf,
            valPredictionDf=valPredictionDf,
            testPredictionDf=testPredictionDf,
        )
    else:
        artifactPathDict = getResidualCnnArtifactPaths(projectConfig)

    return {
        "model": model,
        "historyDf": historyDf,
        "metricSummaryDf": metricSummaryDf,
        "valPredictionDf": valPredictionDf,
        "testPredictionDf": testPredictionDf,
        "cnnConfig": cnnConfig,
        "trainingConfig": trainingConfig,
        "bestEpoch": bestEpochIndex,
        "artifactPathDict": artifactPathDict,
    }


def runResidualCnnHyperparameterSearch(
    projectConfig: ProjectConfig,
    trialSpecList: list[ResidualCnnTrialSpec],
    verbose: bool = True,
) -> dict[str, Any]:
    """We run a notebook-defined residual CNN search and save only the final best artifact set."""
    if len(trialSpecList) == 0:
        raise ValueError("trialSpecList must contain at least one trial.")

    tuningSummaryRowList: list[dict[str, Any]] = []

    bestResult: dict[str, Any] | None = None
    bestTrialSpec: ResidualCnnTrialSpec | None = None
    bestValMacroF1 = float("-inf")

    for trialIndex, trialSpec in enumerate(trialSpecList, start=1):
        trainingConfig = buildTrainingConfigFromTrialSpec(trialSpec)
        cnnConfig = buildModelConfigFromTrialSpec(projectConfig, trialSpec)

        if verbose:
            print(
                f"[search] trialStart | index={trialIndex}/{len(trialSpecList)} | "
                f"name={trialSpec.trialName}"
            )

        trialResult = runResidualCnnExperiment(
            projectConfig=projectConfig,
            cnnConfig=cnnConfig,
            trainingConfig=trainingConfig,
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
                "imageSize": int(trialSpec.imageSize),
                "batchSize": int(trialSpec.batchSize),
                "learningRate": float(trialSpec.learningRate),
                "weightDecay": float(trialSpec.weightDecay),
                "numEpochs": int(trialSpec.numEpochs),
                "earlyStoppingPatience": int(trialSpec.earlyStoppingPatience),
                "labelSmoothing": float(trialSpec.labelSmoothing),
                "useClassWeights": bool(trialSpec.useClassWeights),
                "baseChannels": int(trialSpec.baseChannels),
                "stageBlockCounts": str(trialSpec.stageBlockCounts),
                "dropoutRate": float(trialSpec.dropoutRate),
                "useBatchNorm": bool(trialSpec.useBatchNorm),
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
        raise RuntimeError("We failed to select a best residual CNN trial.")

    tuningSummaryDf = pd.DataFrame(tuningSummaryRowList)
    tuningSummaryDf = tuningSummaryDf.sort_values(
        by=["valMacroF1", "testMacroF1", "valAccuracy"],
        ascending=False,
    ).reset_index(drop=True)

    artifactPathDict = saveResidualCnnArtifacts(
        projectConfig=projectConfig,
        model=bestResult["model"],
        cnnConfig=bestResult["cnnConfig"],
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