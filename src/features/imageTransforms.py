from __future__ import annotations

"""We define the shared image preprocessing and augmentation used in the modeling pipeline."""

from typing import Final

from PIL import Image, ImageOps
from torchvision import transforms


IMAGENET_MEAN: Final[list[float]] = [0.485, 0.456, 0.406]
IMAGENET_STD: Final[list[float]] = [0.229, 0.224, 0.225]
MLP_MEAN: Final[list[float]] = [0.5, 0.5, 0.5]
MLP_STD: Final[list[float]] = [0.5, 0.5, 0.5]

if hasattr(Image, "Resampling"):
    BILINEAR_RESAMPLE = Image.Resampling.BILINEAR
else:
    BILINEAR_RESAMPLE = Image.BILINEAR


class ResizePadToSquare:
    """We resize an image while preserving aspect ratio and then pad it to a square canvas."""

    def __init__(
        self,
        targetSize: int,
        fillColor: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """We store the target square size and the padding color."""
        if targetSize <= 0:
            raise ValueError(f"targetSize must be positive, got: {targetSize}")

        self.targetSize = int(targetSize)
        self.fillColor = fillColor

    def __call__(self, imageObject: Image.Image) -> Image.Image:
        """We resize the image to fit inside a square and pad the remaining area."""
        imageObject = imageObject.convert("RGB")

        originalWidth, originalHeight = imageObject.size

        if originalWidth <= 0 or originalHeight <= 0:
            raise ValueError(
                f"Image must have positive size, got: {(originalWidth, originalHeight)}"
            )

        scaleFactor = min(
            self.targetSize / originalWidth,
            self.targetSize / originalHeight,
        )

        resizedWidth = max(1, int(round(originalWidth * scaleFactor)))
        resizedHeight = max(1, int(round(originalHeight * scaleFactor)))

        resizedImage = imageObject.resize(
            (resizedWidth, resizedHeight),
            resample=BILINEAR_RESAMPLE,
        )

        horizontalPadding = self.targetSize - resizedWidth
        verticalPadding = self.targetSize - resizedHeight

        padLeft = horizontalPadding // 2
        padRight = horizontalPadding - padLeft
        padTop = verticalPadding // 2
        padBottom = verticalPadding - padTop

        paddedImage = ImageOps.expand(
            resizedImage,
            border=(padLeft, padTop, padRight, padBottom),
            fill=self.fillColor,
        )

        return paddedImage


def getBaselineTransform(
    imageSize: int = 128,
    convertToGrayscale: bool = True,
) -> transforms.Compose:
    """We define the baseline preprocessing pipeline used before classical feature extraction."""
    transformList = [
        ResizePadToSquare(targetSize=imageSize, fillColor=(0, 0, 0)),
    ]

    if convertToGrayscale:
        transformList.append(transforms.Grayscale(num_output_channels=1))

    transformObject = transforms.Compose(transformList)
    return transformObject


def getMlpTrainTransform(imageSize: int = 224) -> transforms.Compose:
    """We define the MLP training transform with deterministic RGB tensor normalization."""
    transformObject = transforms.Compose(
        [
            ResizePadToSquare(targetSize=imageSize, fillColor=(0, 0, 0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MLP_MEAN, std=MLP_STD),
        ]
    )

    return transformObject


def getMlpEvalTransform(imageSize: int = 224) -> transforms.Compose:
    """We define the deterministic MLP evaluation transform."""
    transformObject = transforms.Compose(
        [
            ResizePadToSquare(targetSize=imageSize, fillColor=(0, 0, 0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MLP_MEAN, std=MLP_STD),
        ]
    )

    return transformObject


def getCnnTrainTransform(imageSize: int = 224) -> transforms.Compose:
    """We define the CNN training transform with light spatial and color augmentation."""
    transformObject = transforms.Compose(
        [
            ResizePadToSquare(targetSize=imageSize, fillColor=(0, 0, 0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=8,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.05,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.15,
                scale=(0.02, 0.08),
                ratio=(0.4, 2.5),
                value="random",
            ),
        ]
    )

    return transformObject


def getCnnEvalTransform(imageSize: int = 224) -> transforms.Compose:
    """We define the deterministic CNN evaluation transform without random augmentation."""
    transformObject = transforms.Compose(
        [
            ResizePadToSquare(targetSize=imageSize, fillColor=(0, 0, 0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return transformObject