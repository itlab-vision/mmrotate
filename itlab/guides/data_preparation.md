# DOTA Dataset Preparation Guide

This document describes how to download, split (crop), and prepare DOTA dataset versions for MMRotate model training and offline evaluation.

______________________________________________________________________

## 1. Dataset Downloading

To simplify dataset retrieval from Google Drive, use the custom automation script `./tools/data/dota/download_dota.py`.

In addition to downloading and extracting raw dataset archives into `./data/DOTA_<version>`, the script automatically generates an image listing text file (`<split>_set.txt`, e.g., `val_set.txt`, `train_set.txt`) in the root of the dataset version directory (`data/DOTA_1_0/val_set.txt`). This file contains the sorted list of all original `.png` image IDs (without file extension) for the target split and is required for offline evaluation via `DOTA_devkit`.

### Key Arguments

- `--dota-version`: DOTA version(s) to download (`1.0`, `1.5`, `2.0`, or `all`). Default: `1.0`.
- `--split`: Dataset split(s) to download (`train`, `val`, `test`, or `all`). Default: `val`.
- `--out-dir`: Directory where dataset files will be saved. Default: `./data`.
- `--overwrite`: Force re-download and overwrite existing data. Default: `False`.

### Usage Examples

Download default DOTA v1.0 validation set:

```bash
python ./tools/data/dota/download_dota.py
```

Download specific version and split:

```bash
python ./tools/data/dota/download_dota.py --dota-version 1.0 --split val
python ./tools/data/dota/download_dota.py --dota-version 1.5 --split test
```

Download all versions and splits:

```bash
python ./tools/data/dota/download_dota.py --dota-version all --split all
```

Overwrite existing downloads:

```bash
python ./tools/data/dota/download_dota.py --dota-version all --split all --overwrite
```

### Specifics for Downloading DOTA-v2.0

Automated downloading for DOTA-v2.0 requires custom Google Drive IDs. These IDs must be configured in a `.env` file using the provided [`.env.example`](../../tools/data/dota/.env.template) template.

If custom links are unavailable, the dataset must be downloaded manually and organized according to the following directory structure:

```
./data
└── DOTA_2_0
    ├── train
    │   ├── images
    │   └── labelTxt
    └── val
        ├── images
        └── labelTxt
```

**Official Resource:** [DOTA Dataset Official Website](https://captain-whu.github.io/DOTA/dataset.html)

**Alternative Mirrors:**

- **Original Images:** [Dataset Ninja](https://datasetninja.com/dota#introduction).
- **Original Labels (OBB format):** [Ultralytics Assets Release](https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv2.zip).

______________________________________________________________________

## 2. Image Splitting (Cropping Patches)

Large aerial images in DOTA must be cropped into smaller sub-images (e.g., 1024x1024) prior to training and testing.

### Standard Splitting Script (`img_split.py`)

Low-level splitting can be executed directly using a specific JSON configuration:

```bash
python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_train.json
```

***Note:** Polygon intersection calculations during `img_split.py` require the `shapely` library (`pip install shapely`).*

______________________________________________________________________

### Automated Splitting Script (`run_dota_split.py`)

Use `tools/data/dota/run_dota_split.py` to dynamically generate split configurations and execute multi-process image cropping across different DOTA versions, splits, and scales.

#### Key Arguments

- `--dota-version`: DOTA version(s) to process (`1.0`, `1.5`, `2.0`). Default: `1.0`.
- `--data-split`: Dataset split(s) to process (`train`, `val`, `test`, `trainval`). Default: `val`.
- `--scale`: Scaling mode (`ss`, `ms`, `ms-cfa`, `ss-roi-test`, `ms-roi-test`, `ms-roi-train`). Default: `ss ms`.
- `--nproc`: Number of parallel worker processes. Default: `6`.
- `--overwrite`: Overwrite target directories if they already exist. Default: `False`.

#### Usage Examples

Run default splitting (DOTA v1.0, validation split, single-scale and multi-scale with 6 processes):

```bash
python tools/data/dota/run_dota_split.py
```

Overwrite existing split directories:

```bash
python tools/data/dota/run_dota_split.py --overwrite
```

Process DOTA v1.5 with 12 worker processes:

```bash
python tools/data/dota/run_dota_split.py --nproc 12 --dota-version 1.5
```

Process trainval split / roi-scales for DOTA 1.0 and DOTA 1.5:

```bash
python tools/data/dota/run_dota_split.py --data-split trainval --scale s-roi-test ms-roi-test ms-roi-train --dota-version 1.0 1.5
```
