# Colposcopy Digital Atlas - Processed Datasets

This repository contains **two processed colposcopy datasets** together with the
Python preprocessing code used to prepare them. The original classification
dataset has been **extended with a segmentation dataset and segmentation
preprocessing workflow**.

## Included datasets

### 1. Patient-level classification dataset

Directory: `colposcopy-digital-atlas-dataset-pac/`

This dataset is based on the IARC Colposcopy Digital Atlas data collected with
the original scraper available at:

https://github.com/naghim/colposcopy-digital-atlas-dataset

The images are organized into patient-wise training, validation, and test
subsets. Keeping every image from the same patient in a single subset reduces
the risk of data leakage.

The processing includes:

- discovery of patient case directories;
- patient-wise train/validation/test splitting;
- optional center-square cropping;
- resizing to a fixed resolution;
- RGB conversion;
- extraction of patient-level labels;
- generation of `train_labels.csv`, `val_labels.csv`, and `test_labels.csv`.

The binary labels are encoded as:

- `0`: low-grade cervical lesion;
- `1`: high-grade cervical lesion.

The dataset was prepared with `preprocess_dataset.py`.

### 2. Acetowhite-area segmentation dataset

Directory: `processed_annocerv_384_acetowhite/`

This repository is extended with a processed AnnoCerv segmentation dataset for
the pixel-level segmentation of acetowhite regions. It contains 297 colposcopy
images from 100 cases, resized to 384 x 384 pixels, with corresponding binary
acetowhite masks.

The segmentation data are split patient-wise into:

- 213 training images;
- 43 validation images;
- 41 test images.

Each subset contains:

- resized colposcopy images in `images/`;
- binary acetowhite masks in `masks/`;
- a `metadata.csv` file describing the samples.

The root-level `metadata.csv` combines all subsets, while `manifest.json`
records the processing configuration and dataset statistics.

The segmentation workflow is implemented in
`preprocess_dataset_segmentation.py`. In addition to dataset discovery and
patient-wise splitting, it provides mask processing, image-mask augmentation,
PyTorch dataset and data-loader utilities, and summary functions.

## Repository structure

```text
colposcopy-digital-atlas-processed/
|-- colposcopy-digital-atlas-dataset-pac/
|   |-- train/
|   |-- val/
|   |-- test/
|   |-- train_labels.csv
|   |-- val_labels.csv
|   `-- test_labels.csv
|-- processed_annocerv_384_acetowhite/
|   |-- train/
|   |   |-- images/
|   |   |-- masks/
|   |   `-- metadata.csv
|   |-- validation/
|   |   |-- images/
|   |   |-- masks/
|   |   `-- metadata.csv
|   |-- test/
|   |   |-- images/
|   |   |-- masks/
|   |   `-- metadata.csv
|   |-- metadata.csv
|   `-- manifest.json
|-- preprocess_dataset.py
|-- preprocess_dataset_segmentation.py
`-- README.md
```

## Classification preprocessing

Example:

```bash
python preprocess_dataset.py \
    --source path/to/original_dataset \
    --dest colposcopy-digital-atlas-dataset-pac \
    --image-size 224 \
    --crop-mode center \
    --train 0.70 \
    --val 0.15 \
    --seed 42
```

The remaining 15% of cases form the test subset.

## Segmentation preprocessing

The reusable segmentation functions can be imported from:

```python
from preprocess_dataset_segmentation import (
    AnnoCervSegmentationDataset,
    SplitConfig,
    export_processed_dataset,
    load_processed_splits,
)
```

The default split configuration is 70% training, 15% validation, and 15% test,
with random seed 42. Splitting is performed at case level.

## Requirements

The classification preprocessing requires Python 3.8 or newer and Pillow.
The segmentation workflow additionally uses NumPy, pandas, and optionally
PyTorch for dataset and data-loader functionality.

```bash
pip install Pillow numpy pandas torch
```

## Reproducibility

Both workflows use patient/case-wise splitting and a fixed random seed by
default. This makes experiments reproducible and prevents images from the same
patient or case from being distributed across training, validation, and test
subsets.
