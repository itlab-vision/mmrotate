# Evaluation, Benchmarking, and Visualization Guide

This document describes how to download model weights, run local offline evaluation (mAP), compute FPS benchmarks, and execute automated batch evaluations across multiple models using MMRotate tools.

______________________________________________________________________

## 1. Downloading Pre-trained Model Checkpoints

To download pre-trained MMRotate checkpoints required for benchmarks:

```bash
python tools/analysis_tools/download_dota_weights.py
```

______________________________________________________________________

## 2. Offline Evaluation (`tools/test.py`)

Offline evaluation computes mean Average Precision (mAP) on local validation or test sets, formats predictions, generates visualization images, and exports detection pickle files for confusion matrix analysis.

***Note:** By default, dataset directory paths are defined in the `configs/_base_/datasets/dotav1.py` file or in the corresponding configuration file (which takes precedence over `_base_`). These paths can be overridden via command-line arguments at runtime (see example in [Overriding Dataset Paths at Runtime](#overriding-dataset-paths-at-runtime)).*

### Key Arguments

- `config`: Path to the model configuration Python file.
- `checkpoint`: Path to the pre-trained model checkpoint file (`.pth`).
- `--eval`: Evaluation metric(s) to calculate (e.g., `mAP`).
- `--format-only`: Format prediction results into text files for server submission without computing evaluation metrics.
- `--eval-options`: Custom evaluation options dictionary (e.g., `submission_dir=work_dirs/Task1_results`).
- `--show-dir`: Output directory path to save images with drawn rotated bounding box predictions.
- `--out`: Output path to dump raw prediction results into a `.pkl` file.
- `--cfg-options`: Override specific configuration options at runtime (e.g., `data.test.ann_file=...`).

### Basic Local Evaluation Command

```bash
python -W ignore ./tools/test.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  checkpoints/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth \
  --eval mAP
```

### Overriding Dataset Paths at Runtime

You can override dataset paths directly via command line arguments without altering config files:

```bash
python -W ignore ./tools/test.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  checkpoints/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth \
  --eval mAP \
  --cfg-options data.test_dataloader.workers_per_gpu=2 \
                data.test_dataloader.samples_per_gpu=2 \
                data.test.ann_file=data/split_ss_dota_1_0/val/annfiles \
                data.test.img_prefix=data/split_ss_dota_1_0/val/images
```

### Formatting Predictions for Online Server Submission

To generate submission text files for the official DOTA evaluation server:

```bash
python -W ignore ./tools/test.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  checkpoints/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth \
  --format-only \
  --eval-options submission_dir=work_dirs/Task1_results
```

### Visualization of Bounding Boxes

To save images with plotted rotated bounding box predictions:

```bash
python -W ignore ./tools/test.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  checkpoints/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth \
  --show-dir work_dirs/vis
```

### Confusion Matrix Analysis

Generate prediction pickle files first using `tools/test.py --out`, then plot the confusion matrix:

#### Step 1: Export Pickle File

```bash
python -W ignore ./tools/test.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  checkpoints/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth \
  --out work_dirs/detection_res.pkl
```

#### Step 2: Generate Matrix Plot

```bash
python tools/analysis_tools/confusion_matrix.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  work_dirs/detection_res.pkl \
  work_dirs/ \
  --show
```

______________________________________________________________________

## 3. FPS Benchmark Utility (`benchmark.py`)

MMRotate provides a distributed benchmark tool to measure single-model inference speed (forward pass + post-processing).

### Running Benchmark

```bash
PYTHONWARNINGS="ignore" python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29500 \
  tools/analysis_tools/benchmark.py \
  configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
  checkpoints/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth \
  --launcher pytorch --log-interval 5 \
  --cfg-options data.test_dataloader.workers_per_gpu=0 \
                data.test_dataloader.samples_per_gpu=2 \
                data.test.ann_file=data/split_ss_dota_1_0/val/annfiles \
                data.test.img_prefix=data/split_ss_dota_1_0/val/images
```

Example Output:
`Overall fps: 16.5 img / s, times per image: 60.7 ms / img`

### Batch Size Support Note

Standard MMRotate benchmark code evaluates with `batch_size = 1` regardless of configuration. Minor modifications were introduced in this fork's `tools/analysis_tools/benchmark.py` to correctly evaluate custom `batch_size` settings during inference.

______________________________________________________________________

## 4. Automated Multi-Model Batch Evaluation (`evaluate_models.py`)

Use `tools/analysis_tools/evaluate_models.py` to automate multi-model evaluation (mAP and FPS benchmarking) across dataset splits and export JSON log files.

### Key Arguments

- `--dota-version`: DOTA dataset version (`1.0`, `1.5`, or `2.0`). Default: `1.0`.
- `--data-split`: Dataset split to evaluate on (`val` or `test`). Default: `val`.
- `--tasks`: Evaluation tasks to execute (`map`, `benchmark`, or `map+benchmark`). Default: `map+benchmark`.
- `--metafiles`: Specific metafile paths to process. Scans the entire configs/ directory if empty.
- `--models`: List of specific model names to evaluate. Evaluates all matching models if empty.
- `--map-samples-per-gpu`: Batch size per GPU for mAP evaluation. Default: `2`.
- `--map-workers-per-gpu`: Dataloader worker count for mAP evaluation. Default: `2`.
- `--benchmark-samples-per-gpu`: Batch size per GPU for benchmark FPS calculation. Default: `1`.
- `--benchmark-workers-per-gpu`: Dataloader worker count for benchmark FPS calculation. Default: `0`.
- `--work-dir`: Output directory path where JSON reports and execution logs are saved. Default: `work_dirs`.

### Usage Examples

Run default batch evaluation (DOTA v1.0, validation split):

```bash
python tools/analysis_tools/evaluate_models.py
```

Evaluate models from specific metafiles only:

```bash
python tools/analysis_tools/evaluate_models.py \
  --metafiles configs/gaucho/metafile.yml configs/faa/metafile.yml
```

Evaluate on DOTA v1.5 test split:

```bash
python tools/analysis_tools/evaluate_models.py --dota-version 1.5 --data-split test
```

Evaluate specific list of models on a sample dataset split:

```bash
python tools/analysis_tools/evaluate_models.py --data-split sample \
  --models rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90 \
           rotated_atss_hbb_r50_fpn_1x_dota_oc \
           rotated_retinanet_obb_r50_fpn_1x_dota_le90
```

### Converting JSON Results into Summary Tables

To convert evaluation JSON output files into consolidated summary tables, use `tools/analysis_tools/build_summary_table.py`:

```bash
python tools/analysis_tools/build_summary_table.py \
  --input '/path/to/models_stats_dota1_0_val_map+benchmark_20260729_120008.json'
```
