# Model Training Guide

This guide details the complete workflow for training MMRotate models on DOTA datasets within the ITLab GPU infrastructure, from raw dataset downloading and patch splitting to direct interactive training and Slurm batch job submission.

______________________________________________________________________

## 1. Download Train & Validation Datasets

Download the official DOTA training and validation dataset archives using `download_dota.py`:

```bash
python ./tools/data/dota/download_dota.py --dota-version 1.0 1.5 --split train val
```

______________________________________________________________________

## 2. Prepare `trainval` Datasets (Patch Splitting)

High-resolution aerial images in DOTA must be cropped into overlapping patches for model training. Prepare combined `trainval` splits with patch splitting using `run_dota_split.py`:

```bash
python tools/data/dota/run_dota_split.py --nproc 12 --dota-version 1.0 1.5 --data-split trainval --scale all
```

______________________________________________________________________

## 3. Model Training

### Direct Execution (Interactive Node)

To launch model training directly in an interactive GPU session or node:

```bash
python tools/train.py configs_gaucho/gaucho_anchorless_dotav1/gaussian_fcos_r50_fpn_gaucho_probiou_1x_dota_le90.py
```

### Slurm Job Submission

To submit model training as a non-interactive background batch job on the ITLab Slurm cluster:

```bash
sbatch tools/train_model.slurm configs_gaucho/gaucho_anchorless_dotav1/gaussian_fcos_r50_fpn_gaucho_probiou_1x_dota_le90.py
```

______________________________________________________________________

## 4. Batch Training Job Generation

To submit multiple training jobs across an entire directory of model configurations automatically:

First, ensure execution permissions for the batch training script:

```bash
chmod +x tools/batch_train.sh
```

Then execute batch job submission specifying the config directory:

```bash
./tools/batch_train.sh configs/gaucho/dotav1
```

This script scans all `.py` configuration files within the specified folder and queues individual Slurm training jobs for each configuration.
