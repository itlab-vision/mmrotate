# ONNX Model Structure Export Guide

This document describes how to export MMRotate model architectures (backbones, necks, and detection heads) into decoupled ONNX computational graphs.

______________________________________________________________________

## 1. Export Overview

Oriented object detection models contain specialized operators (such as rotated RoI align and rotated NMS) that prevent monolithic end-to-end export. The analysis script [`tools/analysis_tools/dump_model_structure.py`](../../tools/analysis_tools/dump_model_structure.py) breaks models down into decoupled ONNX subsystems with aligned tensor interfaces:

- `backbone_<name>.onnx`: Pure backbone feature extractor ($C_2..C_5$).
- `neck.onnx`: Pure neck / feature pyramid network ($C_2..C_5 \\to P_2..P_6$ or $P_3..P_7$).
- `backbone_neck.onnx`: End-to-end feature extraction from raw input images to pyramid levels.
- `dense_head.onnx` / `roi_head.onnx`: Dense or RoI decision head converging into a unified `RotatedNMS` operator.

______________________________________________________________________

## 2. Command Line Usage

Run the export script using Python:

```bash
python tools/analysis_tools/dump_model_structure.py \
    <CONFIG_PATH> \
    [--out-dir <OUTPUT_DIRECTORY>] \
    [--shape <HEIGHT> <WIDTH>] \
    [--cfg-options <KEY=VALUE>]
```

### Script Arguments

- `config`: Path to the model configuration file (e.g., `configs/cfa/cfa_r50_fpn_40e_dota_oc.py`).
- `--out-dir`: Destination folder for generated `.onnx` files (default: `onnx_models`).
- `--shape`: Input image dimensions `H W` (default: `1024 1024`).
- `--cfg-options`: Override configuration settings at runtime in `key=value` format.

### Example Command

```bash
python tools/analysis_tools/dump_model_structure.py \
    configs/cfa/cfa_r50_fpn_40e_dota_oc.py \
    --out-dir model_structure_analysis/cfa_r50_fpn_40e_dota_oc \
    --shape 1024 1024
```

Running this command exports the decoupled `.onnx` files (`backbone_resnet.onnx`, `neck.onnx`, `backbone_neck.onnx`, `dense_head.onnx`) directly to the target directory.
