# CFA Model Structure

This document details the internal architecture, configuration parameters, and code components for the CFA (Convex-hull Feature Adaptation) model.

The structure and parameters are analyzed based on the target configuration [`configs/cfa/cfa_r50_fpn_40e_dota_oc.py`](../../configs/cfa/cfa_r50_fpn_40e_dota_oc.py).

______________________________________________________________________

## 1. ONNX Model Export

### ONNX Generation Command

```bash
python tools/analysis_tools/dump_model_structure.py \
    configs/cfa/cfa_r50_fpn_40e_dota_oc.py \
    --out-dir model_structure/cfa_r50_fpn_40e_dota_oc \
    --shape 1024 1024
```

This command generates the following decoupled `.onnx` files:

- `backbone_resnet.onnx`
- `neck.onnx`
- `backbone_neck.onnx`
- `dense_head.onnx`

______________________________________________________________________

## 2. Configuration Parameters Breakdown

### 2.1. Backbone (`ResNet-50`)

Configuration parameters in [`configs/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py` (lines 15–25)](../../configs/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py#L15-L25):

```python
    backbone=dict(
        type='ResNet',
        depth=50,       # Network depth (from {18, 34, 50, 101, 152})
        num_stages=4,   # Stages of the network (layer1-layer4)
        out_indices=(0, 1, 2, 3),  # Stages from which feature maps are returned
        frozen_stages=1,           # Stages to be frozen (stop grad and set eval mode)
        zero_init_residual=False,  # Whether to zero-init last norm layer in resblocks to act as identity
        norm_cfg=dict(type=norm_type, requires_grad=True),  # Config dict for norm layers
        norm_eval=True,            # Freeze running stats (mean and var) in norm layers
        style='pytorch',           # Stride-two layer is the 3x3 conv layer
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  # Initialization config dict
```

The backbone generates 4 multi-scale feature maps (`out_indices=(0, 1, 2, 3)`):

- **C2:** stride 4, 256 channels (H/4 × W/4, resolution 256 × 256 for 1024 × 1024 input);
- **C3:** stride 8, 512 channels (H/8 × W/8, resolution 128 × 128);
- **C4:** stride 16, 1024 channels (H/16 × W/16, resolution 64 × 64);
- **C5:** stride 32, 2048 channels (H/32 × W/32, resolution 32 × 32).

### 2.2. Neck (`FPN`)

Configuration parameters in [`configs/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py` (lines 26–33)](../../configs/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py#L26-L33):

```python
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # Number of input channels per scale
        out_channels=256,                    # Number of output channels (used at each scale)
        start_level=1,                       # Index of start input backbone level used to build feature pyramid
        add_extra_convs='on_input',          # Add conv layers on top of the last feature map of neck inputs
        num_outs=5,                          # Number of output scales
        norm_cfg=norm_cfg),                  # Config dict for normalization layer
```

The FPN constructs 5 multi-scale feature pyramid levels (`num_outs=5`, starting from C3 via `start_level=1`):

- **P3 (from C3):** stride 8, 256 channels (128 × 128);
- **P4 (from C4):** stride 16, 256 channels (64 × 64);
- **P5 (from C5):** stride 32, 256 channels (32 × 32);
- **P6 (extra stride-2 conv over C5):** stride 64, 256 channels (16 × 16);
- **P7 (extra stride-2 conv over P6):** stride 128, 256 channels (8 × 8).

### 2.3. Head / Detector (`RotatedRepPoints` & `RotatedRepPointsHead`)

Configuration parameters in [`configs/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py` (lines 14, 34–57)](../../configs/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py#L34-L57):

```python
model = dict(
    type='RotatedRepPoints',  # Implementation of Rotated RepPoints
    ...
    bbox_head=dict(
        type='RotatedRepPointsHead',  # Rotated RepPoints head
        num_classes=15,               # Number of classes
        in_channels=256,              # Number of input channels
        feat_channels=256,            # Number of feature channels
        point_feat_channels=256,      # Number of channels of points features
        stacked_convs=3,              # Number of stacked convolutions
        num_points=9,                 # Number of points in points set
        gradient_mul=0.3,             # Multiplier to gradients from points refinement and recognition
        point_strides=[8, 16, 32, 64, 128],  # Points strides
        point_base_scale=2,           # Bbox scale for assigning labels
        norm_cfg=norm_cfg,            # Config dict for normalization
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),         # Config of classification loss
        loss_bbox_init=dict(type='ConvexGIoULoss', loss_weight=0.375),  # Config of initial points loss
        loss_bbox_refine=dict(type='ConvexGIoULoss', loss_weight=1.0),  # Config of points loss in refinement
        transform_method='rotrect',   # Method to transform RepPoints to bbox
        use_reassign=False,           # Whether to reassign samples
        topk=6,                       # Number of the highest topk points
        anti_factor=0.75),            # Feature anti-aliasing coefficient
```

CFA override in [`configs/cfa/cfa_r50_fpn_1x_dota_oc.py` (lines 3–9)](../../configs/cfa/cfa_r50_fpn_1x_dota_oc.py#L3-L9):

```python
model = dict(
    # Enable CFA (Convex-hull Feature Adaptation)
    bbox_head=dict(
        use_reassign=True),           # Whether to reassign samples (enabled for CFA)
    train_cfg=dict(
        refine=dict(
            # Lower IoU threshold to collect a candidate pool for reassign
            assigner=dict(
                pos_iou_thr=0.1,      # IoU threshold for positive bboxes
                neg_iou_thr=0.1))))   # IoU threshold for negative bboxes
```

______________________________________________________________________

## 3. Code Components

### 3.1. Backbone (`ResNet-50`)

| Component Name       | Role & Implementation                                                      | Code Link                                                                                                                                                |
| :------------------- | :------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backbone**         | `ResNet-50`: 4-stage residual backbone (imported from `mmdet`).            | [mmdet.models.backbones.ResNet](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py)                                  |
| **Backbone Builder** | `build_backbone`: Instantiates ResNet-50 via `ROTATED_BACKBONES` registry. | [builder.py: L15-L17](../../mmrotate/models/builder.py#L15-L17) (invoked at [single_stage.py: L30](../../mmrotate/models/detectors/single_stage.py#L30)) |

### 3.2. Neck (`FPN`)

| Component Name   | Role & Implementation                                                                  | Code Link                                                                                                                                                |
| :--------------- | :------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Neck (FPN)**   | `FPN`: Feature Pyramid Network for multi-scale feature fusion (imported from `mmdet`). | [mmdet.models.necks.FPN](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/fpn.py)                                                |
| **Neck Builder** | `build_neck`: Instantiates FPN via `ROTATED_NECKS` registry.                           | [builder.py: L20-L22](../../mmrotate/models/builder.py#L20-L22) (invoked at [single_stage.py: L32](../../mmrotate/models/detectors/single_stage.py#L32)) |

### 3.3. Head / Detector (`RotatedRepPoints` & `RotatedRepPointsHead`)

| Component Name               | Role & Implementation                                                                                                                  | Code Link                                                                                                       |
| :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **Detector**                 | `RotatedRepPoints`: Top-level single-stage detector wrapper handling model setup.                                                      | [rotated_reppoints.py](../../mmrotate/models/detectors/rotated_reppoints.py)                                    |
| **Base Detector**            | `RotatedSingleStageDetector`: Base class managing feature extraction, training, and testing loops.                                     | [single_stage.py](../../mmrotate/models/detectors/single_stage.py)                                              |
| **Dense Head**               | `RotatedRepPointsHead`: Anchor-free head predicting non-rigid point sets, class scores, and oriented boxes.                            | [rotated_reppoints_head.py](../../mmrotate/models/dense_heads/rotated_reppoints_head.py)                        |
| **Stage 1 (Point Init)**     | `reppoints_pts_init_*`: Predicts initial 9-point spatial offsets around each feature point.                                            | [rotated_reppoints_head.py: L185-L189](../../mmrotate/models/dense_heads/rotated_reppoints_head.py#L185-L189)   |
| **Stage 2 (DCN Adaptation)** | `reppoints_cls_*`, `reppoints_pts_refine_*`: `DeformConv2d` sampling features at 9 predicted points for classification and refinement. | [rotated_reppoints_head.py: L191-L203](../../mmrotate/models/dense_heads/rotated_reppoints_head.py#L191-L203)   |
| **CFA Reassignment**         | `reassign`: Dynamically selects candidate point pools, splits by gradient consistency, and computes anti-aliasing.                     | [rotated_reppoints_head.py: L864-L1027](../../mmrotate/models/dense_heads/rotated_reppoints_head.py#L864-L1027) |
| **Convex Assigner**          | `MaxConvexIoUAssigner`: Assigns ground-truth boxes to candidate convex hulls via Convex-IoU metric.                                    | [max_convex_iou_assigner.py](../../mmrotate/core/bbox/assigners/max_convex_iou_assigner.py)                     |
| **Convex Loss**              | `ConvexGIoULoss`: Continuous differentiable convex-hull GIoU loss for initial and refined point sets.                                  | [convex_giou_loss.py](../../mmrotate/models/losses/convex_giou_loss.py)                                         |
| **Points to OBB**            | `points2rotrect`: Converts 9 predicted points to minimum area bounding rectangles via Jarvis march algorithm.                          | [rotated_reppoints_head.py: L205-L217](../../mmrotate/models/dense_heads/rotated_reppoints_head.py#L205-L217)   |
| **Rotated NMS**              | `multiclass_nms_rotated`: Suppresses overlapping rotated bounding boxes across pyramid levels (test-only).                             | [bbox_nms_rotated.py](../../mmrotate/core/post_processing/bbox_nms_rotated.py)                                  |

______________________________________________________________________

## 4. ResNet-50 Backbone Breakdown

For the detailed configuration parameters of the backbone (`depth=50`, `frozen_stages=1`, `out_indices`, etc.), see [Section 2.1](#21-backbone-resnet-50).

### Layer and Parameter Specifications

| Stage       | Name           | Input Resolution | Output Channels | Stride | Forwarded to Neck  | Structure                            | Parameters               |
| :---------- | :------------- | :--------------- | :-------------- | :----- | :----------------- | :----------------------------------- | :----------------------- |
| **Stem**    | `conv1`, `bn1` | 1024 × 1024      | 64              | 4      | —                  | 7 × 7 Conv, MaxPool                  | ~9.5K (9,536)            |
| **Stage 1** | `layer1`       | 256 × 256        | 256             | 4      | **C2** (256 × 256) | 3 Bottleneck blocks (64, 64, 256)    | ~0.22M (215,808)         |
| **Stage 2** | `layer2`       | 128 × 128        | 512             | 8      | **C3** (128 × 128) | 4 Bottleneck blocks (128, 128, 512)  | ~1.22M (1,219,584)       |
| **Stage 3** | `layer3`       | 64 × 64          | 1024            | 16     | **C4** (64 × 64)   | 6 Bottleneck blocks (256, 256, 1024) | ~7.10M (7,098,368)       |
| **Stage 4** | `layer4`       | 32 × 32          | 2048            | 32     | **C5** (32 × 32)   | 3 Bottleneck blocks (512, 512, 2048) | ~14.96M (14,964,736)     |
| **Total**   |                |                  |                 |        | **4 feature maps** | **50 Conv layers**                   | **23,508,032 (~23.51M)** |

The 4 feature maps (C2, C3, C4, C5) produced by `layer1`–`layer4` (`out_indices=(0, 1, 2, 3)`) are forwarded directly to the FPN neck (`in_channels=[256, 512, 1024, 2048]`).
