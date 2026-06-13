# Colposcopy Digital Atlas Dataset – Processed Version

This repository contains a processed version of the **IARC Colposcopy Atlas Dataset**, prepared for machine learning experiments using **patient-wise train/validation/test splitting** to minimize data leakage between dataset subsets.

## Overview

The original dataset was collected from the IARC Screening Group Atlas of Colposcopy using the original scraper implementation available at:

https://github.com/naghim/colposcopy-digital-atlas-dataset

This repository provides a standardized dataset structure suitable for deep learning workflows.

The preprocessing pipeline performs:

* Automatic discovery of patient cases (`case_*` folders)
* Patient-wise train/validation/test splitting
* Optional center square cropping of images
* Image resizing to a fixed resolution
* RGB conversion of all images
* Label extraction from the original dataset structure
* Generation of split-specific label files (`train_labels.csv`, `val_labels.csv`, `test_labels.csv`)
* Organization of processed images into a standardized directory structure

The default preprocessing configuration uses:

* Training split: 70%
* Validation split: 15%
* Test split: 15%
* Random seed: 42
* Image size: 224 × 224 pixels
* Center square cropping

## Dataset Structure

```text
paciens-data/
└── colposcopy-digital-atlas-dataset-pac/
    ├── train/
    │   ├── case_AAAN/
    │   ├── case_AABM/
    │   └── ...
    ├── val/
    │   ├── case_AABG/
    │   └── ...
    ├── test/
    │   ├── case_ABCD/
    │   └── ...
    ├── train_labels.csv
    ├── val_labels.csv
    └── test_labels.csv
```

Each `case_*` directory contains all images associated with a single patient case.

The CSV files contain patient-level labels corresponding to each dataset split.

## Label Files

The repository provides split-specific label files:

* `train_labels.csv`
* `val_labels.csv`
* `test_labels.csv`

Each CSV file contains one row per patient case:

```csv
case_id,label
case_AAEM,0
case_AACE,1
case_AACC,1
...
```

### Columns

| Column    | Description                    |
| --------- | ------------------------------ |
| `case_id` | Unique patient case identifier |
| `label`   | Binary classification label    |

### Label Encoding

* `0` → Low-grade cervical lesion
* `1` → High-grade cervical lesion

Labels are assigned at the **patient level**, meaning that all images belonging to the same case share the same diagnostic label.

## Processing Pipeline

The dataset was generated using the preprocessing script:

```bash
python scripts/process_dataset.py \
    --source path/to/original_dataset \
    --dest path/to/output_dataset
```

### Available Arguments

| Argument       | Description                                | Default  |
| -------------- | ------------------------------------------ | -------- |
| `--source`     | Path to the original dataset               | Required |
| `--dest`       | Output directory for the processed dataset | Required |
| `--image-size` | Output image resolution                    | `224`    |
| `--crop-mode`  | Cropping strategy (`center`, `none`)       | `center` |
| `--train`      | Training split ratio                       | `0.70`   |
| `--val`        | Validation split ratio                     | `0.15`   |
| `--seed`       | Random seed for reproducibility            | `42`     |

The preprocessing pipeline performs:

* Parsing of the original dataset structure
* Patient-level data splitting to avoid data leakage
* Center cropping and image resizing
* RGB conversion of all images
* Label extraction and CSV generation
* Organization of images into training, validation, and test subsets

## Requirements

Python 3.8+

Required packages:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
Pillow>=9.0.0
```

## Reproducibility

The dataset splits are generated using a fixed random seed (`seed=42` by default) to ensure reproducibility of experiments. Patient-wise splitting is used to prevent information leakage between training, validation, and test sets.
