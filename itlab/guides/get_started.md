# Getting Started Guide

This document outlines the recommended step-by-step onboarding roadmap for developers joining the ITLab MMRotate workflow. Follow these guides in sequence to set up your environment, prepare datasets, and run model evaluations.

______________________________________________________________________

## Onboarding Roadmap

### Step 1: Environment Setup

Before running code, configure your GPU server environment:

- Read **[environment_setup.md](environment_setup.md)**.
- Create the `openmmlab` Conda environment (Python 3.8, PyTorch 1.8.0, CUDA 11.1).
- Install OpenMMLab packages (`mmcv-full`, `mmdet`, `mmcls`) and install MMRotate in editable mode.
- Apply the server-specific `opencv-python-headless` fix and verify installation via PyTorch and demo scripts.

______________________________________________________________________

### Step 2: Data Preparation

Once the environment is active, prepare dataset files for training or testing:

- Read **[data_preparation.md](data_preparation.md)**.
- Download required DOTA dataset archives using `tools/data/dota/download_dota.py`.
- Crop large aerial images into sub-patches using the automated multi-process script `tools/data/dota/run_dota_split.py`.

______________________________________________________________________

### Step 3: Automated Multi-Model Evaluation and Benchmarking

Execute automated batch evaluations across pre-trained model checkpoints and generate summary reports:

- Read **[evaluation_and_benchmarking.md](evaluation_and_benchmarking.md)**.
- Download pre-trained model checkpoints using `tools/analysis_tools/download_dota_weights.py`.
- Execute automated multi-model batch evaluations (mAP metrics and FPS benchmarks) using `tools/analysis_tools/evaluate_models.py`.
- Generate consolidated markdown summary tables from JSON logs using `tools/analysis_tools/build_summary_table.py`.

______________________________________________________________________

### Step 4: Model Training

Train custom or baseline models on DOTA datasets:

- Read **[model_training.md](model_training.md)**.
- Prepare `trainval` patch splits using `tools/data/dota/run_dota_split.py`.
- Launch training directly (`tools/train.py`) or submit Slurm cluster jobs (`tools/train_model.slurm` & `tools/batch_train.sh`).
