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
python tools/train.py configs/gaucho/dotav1/gaucho_anchorless_dotav1/gaussian_fcos_r50_fpn_gaucho_probiou_1x_dota_le90.py --auto-resume
```

The `--auto-resume` flag automatically resumes training from the latest completed epoch.
The script first checks for the `latest.pth` file; if not found, it parses all available `epoch_*.pth` files and loads the weights from the highest epoch number.
If no saved weights are found (e.g., during an initial run), training starts from the first epoch.

### Slurm Job Submission

To submit model training as a non-interactive background batch job on the Slurm cluster, use the `train_model.slurm` script. This script supports both single-GPU and multi-GPU distributed training with automatic port conflict resolution.

The `--auto-resume` flag is applied by default within this script. The script also prints the configuration file path at the very beginning of the `train_%j.out` log file, allowing you to easily find the log for a specific experiment later by running `grep "CONFIG:" *.out` in your directory.

**Single-GPU Training (Default):**

```bash
sbatch tools/train_model.slurm configs/gaucho/dotav1/gaucho_anchorless_dotav1/gaussian_fcos_r50_fpn_gaucho_probiou_1x_dota_le90.py
```

**Multi-GPU Distributed Training (e.g., 4 GPUs):**

```bash
sbatch --gres=gpu:4 tools/train_model.slurm configs/gaucho/dotav1/gaucho_anchorless_dotav1/gaussian_fcos_r50_fpn_gaucho_probiou_1x_dota_le90.py 4
```

When training on multiple GPUs, you must ensure that your configuration file supports dynamic multi-GPU scaling. This includes automatically scaling the learning rate (e.g., `lr = base_lr * gpu_number`) and switching normalization layers to `SyncBN` when more than one GPU is detected. You can use `configs/faa/oriented_rcnn_r50_fpn_1x_dota15_rr_le90_faa.py` as a reference example for a properly configured file.
