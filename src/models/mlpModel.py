from __future__ import annotations

"""We define the shared multiclass MLP used on top of HOG feature vectors."""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from src.config.projectConfig import ProjectConfig


@dataclass(frozen=True)
class MlpConfig:
    """We store the architecture settings for one shared HOG-MLP classifier."""

    inputFeatureDim: int
    numClasses: int
    hiddenDims: Tuple[int, ...] = (512, 256)
    dropoutRate: float = 0.30
    useBatchNorm: bool = True


def buildDefaultMlpConfig(
    projectConfig: ProjectConfig,
    inputFeatureDim: int,
) -> MlpConfig:
    """We build one strong default HOG-MLP architecture from the project config."""
    return MlpConfig(
        inputFeatureDim=int(inputFeatureDim),
        numClasses=len(projectConfig.classNames),
        hiddenDims=(512, 256),
        dropoutRate=0.30,
        useBatchNorm=True,
    )


class MlpClassifier(nn.Module):
    """We implement one shared multiclass MLP for standardized HOG feature vectors."""

    def __init__(self, mlpConfig: MlpConfig) -> None:
        """We store the config and build the sequential network."""
        super().__init__()
        self.mlpConfig = mlpConfig
        self.network = self._buildNetwork()

    def _buildNetwork(self) -> nn.Sequential:
        """We build the dense network blocks and the final multiclass output layer."""
        layerList: list[nn.Module] = []
        inFeatures = self.mlpConfig.inputFeatureDim

        for hiddenDim in self.mlpConfig.hiddenDims:
            layerList.append(nn.Linear(inFeatures, hiddenDim))

            if self.mlpConfig.useBatchNorm:
                layerList.append(nn.BatchNorm1d(hiddenDim))

            layerList.append(nn.ReLU())
            layerList.append(nn.Dropout(self.mlpConfig.dropoutRate))
            inFeatures = hiddenDim

        layerList.append(nn.Linear(inFeatures, self.mlpConfig.numClasses))

        return nn.Sequential(*layerList)

    def forward(self, featureTensor: torch.Tensor) -> torch.Tensor:
        """We map one batch of HOG feature vectors to multiclass logits."""
        logits = self.network(featureTensor)
        return logits