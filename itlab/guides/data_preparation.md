# DOTA Dataset Preparation Guide

This document describes how to download, split (crop), and prepare DOTA dataset versions for MMRotate model training and offline evaluation.

---

## 1. Dataset Downloading

To simplify dataset retrieval from Google Drive, use the custom automation script `./tools/data/dota/download_dota.py`.

### Prerequisites

Ensure `gdown` is installed:

```bash
pip install gdown
```

### Usage Examples

Download default DOTA v1.0 validation set:
```bash
python ./tools/data/dota/download_dota.py
```

Download specific version and split:
```bash
python ./tools/data/dota/download_dota.py --version 1.0 --split val
python ./tools/data/dota/download_dota.py --version 1.5 --split test
```

Download all versions and splits:
```bash
python ./tools/data/dota/download_dota.py --version all --split all
```

Overwrite existing downloads:
```bash
python ./tools/data/dota/download_dota.py --version all --split all --overwrite
```

---

## 2. Image Splitting (Cropping Patches)

Large aerial images in DOTA must be cropped into smaller sub-images (e.g., 1024x1024) prior to training and testing.

### Standard Splitting Script (`img_split.py`)

Low-level splitting can be executed directly using a specific JSON configuration:

```bash
python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_train.json
```

***Note:** Polygon intersection calculations during `img_split.py` require the `shapely` library (`pip install shapely`).*

---

### Automated Splitting Script (`run_dota_split.py`)

- **Location**: `tools/data/dota/run_dota_split.py`
- **Description**: A wrapper script that dynamically generates split configurations and executes multi-process image cropping (`img_split.py`) across different DOTA versions, splits, and scales.

#### Key Arguments

- `--dota-version`: DOTA version(s) to process (`1.0`, `1.5`, `2.0`, or `all`). Default: `1.0`.
- `--data-split`: Dataset split(s) to process (`train`, `val`, `test`, or `all`). Default: `val`.
- `--scale`: Scaling mode (`ss` for single-scale, `ms` for multi-scale, or `all`). Default: `ss ms`.
- `--nproc`: Number of parallel worker processes. Default: `10`.
- `--overwrite`: Overwrite target directories if they already exist. Default: `False`.

#### Usage Examples

Run default splitting (DOTA v1.0, validation split, single-scale and multi-scale with 10 processes):
```bash
python tools/data/dota/run_dota_split.py
```

Process DOTA v1.5 with 6 worker processes:
```bash
python tools/data/dota/run_dota_split.py --nproc 6 --dota-version 1.5
```

Process all splits (train, val, test) and scales for DOTA 1.0 with 8 processes:
```bash
python tools/data/dota/run_dota_split.py --dota-version 1.0 --data-split all --scale all --nproc 8
```

Overwrite existing split directories:
```bash
python tools/data/dota/run_dota_split.py --dota-version 1.0 --data-split val --overwrite
```
