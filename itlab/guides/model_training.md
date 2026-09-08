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
python tools/train.py configs_gaucho/gaucho_anchorless_dotav1/gaussian_fcos_r50_fpn_gaucho_probiou_1x_dota_le90.py --auto-resume
```

The `--auto-resume` flag automatically resumes training from the latest completed epoch.
The script first checks for the `latest.pth` file; if not found, it parses all available `epoch_*.pth` files and loads the weights from the highest epoch number.
If no saved weights are found (e.g., during an initial run), training starts from the first epoch.

### Slurm Job Submission

To submit model training as a non-interactive background batch job on the Slurm cluster:

```bash
sbatch tools/train_model.slurm configs_gaucho/gaucho_anchorless_dotav1/gaussian_fcos_r50_fpn_gaucho_probiou_1x_dota_le90.py
```

***Note:** The `--auto-resume` flag is applied by default within this script.*
