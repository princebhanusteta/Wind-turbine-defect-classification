# WTDAnalysis

## Project Title
Crop-Based Multi-Class Classification of Wind Turbine Blade Defects from UAV Images

## Project Goal
We build a clean end-to-end machine learning project for classifying wind turbine blade defects from cropped UAV image regions. We use the provided full images, Pascal VOC annotations, and official train-validation-test split. Our final goal is to produce a neat, reproducible pipeline that supports both scientific reporting and presentation.

## Task Scope
This project focuses on crop-based 6-class defect classification, not full-object detection.

The six classes are:
- craze
- corrosion
- surface_injure
- thunderstrike
- crack
- hide_craze

## Data Source
The raw dataset is stored in:

- `dataRaw/wtDataset/JPEGImages`
- `dataRaw/wtDataset/Annotations`
- `dataRaw/wtDataset/annotation_second_person`
- `dataRaw/wtDataset/train_val_test_split.txt`
- `dataRaw/wtDataset/class_definitions.txt`

We treat the raw dataset folder as read-only.

## Planned Project Structure
- `dataRaw/` stores original raw data
- `dataProcessed/` stores generated crops, audits, and manifests
- `src/` stores project logic
- `notebooks/` stores lightweight visualization and result notebooks
- `outputs/` stores generated figures, metrics, and tables
- `reportAssets/` stores final report-ready figures and tables

## Planned Workflow
1. We verify project setup and raw data integrity.
2. We audit the dataset and annotation quality.
3. We generate crop-level data from full-image annotations.
4. We build a clean crop manifest with labels and split membership.
5. We train a simple baseline model.
6. We train a compact CNN model.
7. We evaluate results using clear visual and quantitative analysis.
8. We prepare clean outputs for the report and presentation.

## Reproducibility Rules
- We use seed `27`.
- We keep most logic inside `src/`.
- We keep notebooks lightweight.
- We preserve the provided split and avoid leakage.
- We keep the project clean, minimal, and reproducible.

## Environment
We use a dedicated conda environment for this project and keep dependencies listed in:
- `requirements.txt`
- `requirements-lock.txt`

