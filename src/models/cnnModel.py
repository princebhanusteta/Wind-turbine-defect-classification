from __future__ import annotations

"""We define the compact residual CNN used for crop-based defect classification."""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from src.config.projectConfig import ProjectConfig


@dataclass(frozen=True)
class ResidualCnnConfig:
    """We store the architecture settings for the compact residual CNN classifier."""

    inputChannels: int = 3
    numClasses: int = 6
    baseChannels: int = 32
    stageBlockCounts: Tuple[int, ...] = (2, 2, 2, 2)
    dropoutRate: float = 0.20
    useBatchNorm: bool = True

    @property
    def stageChannels(self) -> Tuple[int, ...]:
        """We return the output channel width of each residual stage."""
        return tuple(
            self.baseChannels * (2 ** stageIndex)
            for stageIndex in range(len(self.stageBlockCounts))
        )


def buildDefaultResidualCnnConfig(projectConfig: ProjectConfig) -> ResidualCnnConfig:
    """We build the default compact residual CNN configuration from the shared project config."""
    return ResidualCnnConfig(
        inputChannels=3,
        numClasses=len(projectConfig.classNames),
        baseChannels=32,
        stageBlockCounts=(2, 2, 2, 2),
        dropoutRate=0.20,
        useBatchNorm=True,
    )


class ResidualBlock(nn.Module):
    """We implement one basic residual block with two 3x3 convolutions."""

    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        stride: int = 1,
        useBatchNorm: bool = True,
    ) -> None:
        """We build the residual block and its optional projection shortcut."""
        super().__init__()

        self.useBatchNorm = useBatchNorm

        self.conv1 = nn.Conv2d(
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not useBatchNorm,
        )
        self.bn1 = nn.BatchNorm2d(outChannels) if useBatchNorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=outChannels,
            out_channels=outChannels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not useBatchNorm,
        )
        self.bn2 = nn.BatchNorm2d(outChannels) if useBatchNorm else nn.Identity()

        if stride != 1 or inChannels != outChannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=inChannels,
                    out_channels=outChannels,
                    kernel_size=1,
                    stride=stride,
                    bias=not useBatchNorm,
                ),
                nn.BatchNorm2d(outChannels) if useBatchNorm else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputTensor: torch.Tensor) -> torch.Tensor:
        """We run the residual block and return the transformed feature map."""
        residualTensor = self.shortcut(inputTensor)

        outputTensor = self.conv1(inputTensor)
        outputTensor = self.bn1(outputTensor)
        outputTensor = self.relu(outputTensor)

        outputTensor = self.conv2(outputTensor)
        outputTensor = self.bn2(outputTensor)

        outputTensor = outputTensor + residualTensor
        outputTensor = self.relu(outputTensor)

        return outputTensor


class ResidualCnnClassifier(nn.Module):
    """We implement a compact residual CNN classifier for raw crop images."""

    def __init__(self, cnnConfig: ResidualCnnConfig) -> None:
        """We store the config and build the stem, residual stages, and classifier head."""
        super().__init__()
        self.cnnConfig = cnnConfig

        self.stem = self._buildStem()
        self.residualStages = self._buildResidualStages()
        self.globalPool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=self.cnnConfig.dropoutRate)
        self.classifier = nn.Linear(
            in_features=self.cnnConfig.stageChannels[-1],
            out_features=self.cnnConfig.numClasses,
        )

    def _buildStem(self) -> nn.Sequential:
        """We build a lightweight stem that preserves early defect detail."""
        stemLayerList: list[nn.Module] = [
            nn.Conv2d(
                in_channels=self.cnnConfig.inputChannels,
                out_channels=self.cnnConfig.baseChannels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.cnnConfig.useBatchNorm,
            ),
        ]

        if self.cnnConfig.useBatchNorm:
            stemLayerList.append(nn.BatchNorm2d(self.cnnConfig.baseChannels))

        stemLayerList.append(nn.ReLU(inplace=True))

        return nn.Sequential(*stemLayerList)

    def _buildResidualStages(self) -> nn.Sequential:
        """We build the stack of residual stages with progressive channel growth."""
        stageList: list[nn.Module] = []
        currentChannels = self.cnnConfig.baseChannels

        for stageIndex, (stageChannels, blockCount) in enumerate(
            zip(self.cnnConfig.stageChannels, self.cnnConfig.stageBlockCounts)
        ):
            stride = 1 if stageIndex == 0 else 2
            stageModule = self._buildOneStage(
                inChannels=currentChannels,
                outChannels=stageChannels,
                blockCount=blockCount,
                firstStride=stride,
            )
            stageList.append(stageModule)
            currentChannels = stageChannels

        return nn.Sequential(*stageList)

    def _buildOneStage(
        self,
        inChannels: int,
        outChannels: int,
        blockCount: int,
        firstStride: int,
    ) -> nn.Sequential:
        """We build one residual stage with one transition block and remaining same-width blocks."""
        if blockCount <= 0:
            raise ValueError(f"blockCount must be positive, got: {blockCount}")

        blockList: list[nn.Module] = [
            ResidualBlock(
                inChannels=inChannels,
                outChannels=outChannels,
                stride=firstStride,
                useBatchNorm=self.cnnConfig.useBatchNorm,
            )
        ]

        for _ in range(blockCount - 1):
            blockList.append(
                ResidualBlock(
                    inChannels=outChannels,
                    outChannels=outChannels,
                    stride=1,
                    useBatchNorm=self.cnnConfig.useBatchNorm,
                )
            )

        return nn.Sequential(*blockList)

    def forwardFeatures(self, imageTensor: torch.Tensor) -> torch.Tensor:
        """We map a batch of images to the pooled CNN feature representation."""
        featureTensor = self.stem(imageTensor)
        featureTensor = self.residualStages(featureTensor)
        featureTensor = self.globalPool(featureTensor)
        featureTensor = torch.flatten(featureTensor, start_dim=1)
        return featureTensor

    def forward(self, imageTensor: torch.Tensor) -> torch.Tensor:
        """We map a batch of images to multiclass logits."""
        featureTensor = self.forwardFeatures(imageTensor)
        featureTensor = self.dropout(featureTensor)
        logits = self.classifier(featureTensor)
        return logits