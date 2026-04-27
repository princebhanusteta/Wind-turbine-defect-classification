from dataclasses import dataclass, field
from pathlib import Path

import torch


def detectDevice() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 27

    projectRoot: Path = Path(__file__).resolve().parents[2]

    dataRawDir: Path = field(init=False)
    wtDatasetDir: Path = field(init=False)
    imageDir: Path = field(init=False)
    annotationDir: Path = field(init=False)
    secondAnnotatorDir: Path = field(init=False)
    splitFilePath: Path = field(init=False)
    classFilePath: Path = field(init=False)

    dataProcessedDir: Path = field(init=False)
    auditDir: Path = field(init=False)
    cropsDir: Path = field(init=False)
    manifestsDir: Path = field(init=False)

    outputsDir: Path = field(init=False)
    figuresDir: Path = field(init=False)
    metricsDir: Path = field(init=False)
    samplePredictionsDir: Path = field(init=False)
    tablesDir: Path = field(init=False)

    reportAssetsDir: Path = field(init=False)
    reportFiguresDir: Path = field(init=False)
    reportTablesDir: Path = field(init=False)

    notebooksDir: Path = field(init=False)

    classNames: list[str] = field(default_factory=lambda: [
        "craze",
        "corrosion",
        "surface_injure",
        "thunderstrike",
        "crack",
        "hide_craze",
    ])

    imageSize: int = 224
    batchSize: int = 32
    numWorkers: int = 0
    learningRate: float = 1e-3
    weightDecay: float = 1e-4
    numEpochs: int = 15
    device: str = field(default_factory=detectDevice)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataRawDir", self.projectRoot / "dataRaw")
        object.__setattr__(self, "wtDatasetDir", self.dataRawDir / "wtDataset")
        object.__setattr__(self, "imageDir", self.wtDatasetDir / "JPEGImages")
        object.__setattr__(self, "annotationDir", self.wtDatasetDir / "Annotations")
        object.__setattr__(self, "secondAnnotatorDir", self.wtDatasetDir / "annotation_second_person")
        object.__setattr__(self, "splitFilePath", self.wtDatasetDir / "train_val_test_split.txt")
        object.__setattr__(self, "classFilePath", self.wtDatasetDir / "class_definitions.txt")

        object.__setattr__(self, "dataProcessedDir", self.projectRoot / "dataProcessed")
        object.__setattr__(self, "auditDir", self.dataProcessedDir / "audit")
        object.__setattr__(self, "cropsDir", self.dataProcessedDir / "crops")
        object.__setattr__(self, "manifestsDir", self.dataProcessedDir / "manifests")

        object.__setattr__(self, "outputsDir", self.projectRoot / "outputs")
        object.__setattr__(self, "figuresDir", self.outputsDir / "figures")
        object.__setattr__(self, "metricsDir", self.outputsDir / "metrics")
        object.__setattr__(self, "samplePredictionsDir", self.outputsDir / "samplePredictions")
        object.__setattr__(self, "tablesDir", self.outputsDir / "tables")

        object.__setattr__(self, "reportAssetsDir", self.projectRoot / "reportAssets")
        object.__setattr__(self, "reportFiguresDir", self.reportAssetsDir / "figures")
        object.__setattr__(self, "reportTablesDir", self.reportAssetsDir / "tables")

        object.__setattr__(self, "notebooksDir", self.projectRoot / "notebooks")


projectConfig = ProjectConfig()