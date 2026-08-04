# Environment Setup and Deployment Guide

This guide describes the exact, tested step-by-step process for deploying MMRotate (ITLab fork) on NVIDIA GPU servers.

---

## 1. Verified Prerequisites

- **Python**: 3.8
- **CUDA**: 11.1
- **Package Manager**: Conda

---

## 2. Step-by-Step Installation

### Step 1: Create and Activate Conda Environment

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

***Note:** If package caching issues occur during installation, run `conda clean --all -y`.*

---

### Step 2: Install PyTorch and Torchvision

Install PyTorch 1.8.0 with CUDA 11.1 support:

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia
```

---

### Step 3: Install OpenMIM, MMCV, and MMDetection

Install OpenMMLab dependencies via MIM:

```bash
pip install -U openmim
mim install mmcv-full
mim install mmdet\<3.0.0
```

---

### Step 4: Install MMRotate (ITLab Fork) and Additional Dependencies

Clone the repository and install MMRotate in editable mode along with required dependencies:

```bash
git clone https://github.com/itlab-vision/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
pip install shapely gdown mmcls
```

---

### Step 5: Server OpenCV Fix

Headless GPU servers require `opencv-python-headless` to avoid GUI/display dependencies:

```bash
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-python-headless
```

---

## 3. Verification

### Verification Step 1: Check PyTorch CUDA Setup

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output format:
```text
PyTorch: 1.8.0
CUDA available: True
GPU: NVIDIA A100-PCIE-40GB
```

### Verification Step 2: Run Demo Inference

```bash
mim download mmrotate --config oriented_rcnn_r50_fpn_1x_dota_le90 --dest demo/tmp/

python demo/image_demo.py demo/demo.jpg demo/tmp/oriented_rcnn_r50_fpn_1x_dota_le90.py demo/tmp/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth --out-file demo/tmp/result.jpg
```

Expected result: Generates `result.jpg` containing plotted rotated bounding boxes over detected objects.
