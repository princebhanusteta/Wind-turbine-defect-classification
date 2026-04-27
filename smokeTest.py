from pathlib import Path

from src.config import projectConfig
from src.utils import ensureProjectDirs, printProjectSummary, setGlobalSeed


def countFiles(targetDir: Path) -> int:
    return sum(1 for currentPath in targetDir.iterdir() if currentPath.is_file())


def getFirstFilePath(targetDir: Path) -> Path:
    filePaths = sorted([currentPath for currentPath in targetDir.iterdir() if currentPath.is_file()])
    if not filePaths:
        raise FileNotFoundError(f"No files found in: {targetDir}")
    return filePaths[0]


def main() -> None:
    setGlobalSeed(projectConfig.seed)
    ensureProjectDirs(projectConfig)

    print("Running smoke test")
    printProjectSummary(projectConfig)

    requiredPaths = {
        "imageDir": projectConfig.imageDir,
        "annotationDir": projectConfig.annotationDir,
        "secondAnnotatorDir": projectConfig.secondAnnotatorDir,
        "splitFilePath": projectConfig.splitFilePath,
        "classFilePath": projectConfig.classFilePath,
    }

    print("\nChecking required paths")
    for pathName, currentPath in requiredPaths.items():
        print(f"{pathName}Exists:", currentPath.exists())

    imageCount = countFiles(projectConfig.imageDir)
    annotationCount = countFiles(projectConfig.annotationDir)
    secondAnnotatorCount = countFiles(projectConfig.secondAnnotatorDir)

    print("\nCounting raw files")
    print("imageCount:", imageCount)
    print("annotationCount:", annotationCount)
    print("secondAnnotatorCount:", secondAnnotatorCount)

    firstImagePath = getFirstFilePath(projectConfig.imageDir)
    firstAnnotationPath = getFirstFilePath(projectConfig.annotationDir)
    firstSecondAnnotatorPath = getFirstFilePath(projectConfig.secondAnnotatorDir)

    print("\nInspecting first sample paths")
    print("firstImagePath:", firstImagePath.name)
    print("firstAnnotationPath:", firstAnnotationPath.name)
    print("firstSecondAnnotatorPath:", firstSecondAnnotatorPath.name)

    smokeTestReportPath = projectConfig.metricsDir / "smokeTestReport.txt"
    with open(smokeTestReportPath, "w", encoding="utf-8") as textFile:
        textFile.write("Smoke test completed successfully\n")
        textFile.write(f"seed: {projectConfig.seed}\n")
        textFile.write(f"device: {projectConfig.device}\n")
        textFile.write(f"imageCount: {imageCount}\n")
        textFile.write(f"annotationCount: {annotationCount}\n")
        textFile.write(f"secondAnnotatorCount: {secondAnnotatorCount}\n")
        textFile.write(f"firstImagePath: {firstImagePath.name}\n")
        textFile.write(f"firstAnnotationPath: {firstAnnotationPath.name}\n")
        textFile.write(f"firstSecondAnnotatorPath: {firstSecondAnnotatorPath.name}\n")

    print("\nSaved report:", smokeTestReportPath)
    print("Smoke test completed successfully")


if __name__ == "__main__":
    main()