# GauCho Model Structure

This document details the internal architecture, configuration parameters, and code components for the GauCho (Gaussian Distributions with Cholesky Decomposition) model applied to oriented object detection.

The structure and parameters are analyzed based on the target configuration [`configs/gaucho/dotav1/SOTA/roi_transformer_r101_fpn_gaucho_probiou_1x_dota_ms_rr_le90.py`](../../configs/gaucho/dotav1/SOTA/roi_transformer_r101_fpn_gaucho_probiou_1x_dota_ms_rr_le90.py).

______________________________________________________________________

## 1. ONNX Model Export

### ONNX Generation Command

```bash
python tools/analysis_tools/dump_model_structure.py \
    configs/gaucho/dotav1/SOTA/roi_transformer_r101_fpn_gaucho_probiou_1x_dota_ms_rr_le90.py \
    --out-dir model_structure/roi_transformer_r101_fpn_gaucho_probiou_1x_dota_ms_rr_le90 \
    --shape 1024 1024
```

This command generates the following decoupled `.onnx` files:

- `backbone_resnet.onnx`
- `neck.onnx`
- `backbone_neck.onnx`
- `roi_head.onnx`

______________________________________________________________________

## 2. Configuration Parameters Breakdown

### 2.1. Backbone (`ResNet-101`)

Configuration parameters in [`configs/gaucho/dotav1/gaucho_two_stage_dotav1/roi_transformer_r50_fpn_gaucho_probiou_1x_dota_le90.py` (lines 56–65)](../../configs/gaucho/dotav1/gaucho_two_stage_dotav1/roi_transformer_r50_fpn_gaucho_probiou_1x_dota_le90.py#L56-L65):

```python
    backbone=dict(
        type='ResNet',
        depth=101,      # Network depth: overridden to 101 in target config
        num_stages=4,   # Stages of the network (layer1-layer4)
        out_indices=(0, 1, 2, 3),  # Stages from which feature maps are returned
        frozen_stages=1,           # Stages to be frozen (stop grad and set eval mode)
        zero_init_residual=False,  # Whether to zero-init last norm layer in resblocks to act as identity
        norm_cfg=dict(type='BN', requires_grad=True),  # Config dict for norm layers
        norm_eval=True,            # Freeze running stats (mean and var) in norm layers
        style='pytorch',           # Stride-two layer is the 3x3 conv layer
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),  # Pretrained checkpoint: overridden to resnet101 in target config
```

The backbone generates 4 multi-scale feature maps (`out_indices=(0, 1, 2, 3)`):

- **C2:** stride 4, 256 channels (H/4 × W/4, resolution 256 × 256 for 1024 × 1024 input);
- **C3:** stride 8, 512 channels (H/8 × W/8, resolution 128 × 128);
- **C4:** stride 16, 1024 channels (H/16 × W/16, resolution 64 × 64);
- **C5:** stride 32, 2048 channels (H/32 × W/32, resolution 32 × 32).

### 2.2. Neck (`FPN`)

Configuration parameters in [`configs/gaucho/dotav1/gaucho_two_stage_dotav1/roi_transformer_r50_fpn_gaucho_probiou_1x_dota_le90.py` (lines 66–70)](../../configs/gaucho/dotav1/gaucho_two_stage_dotav1/roi_transformer_r50_fpn_gaucho_probiou_1x_dota_le90.py#L66-L70):

```python
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # Number of input channels per scale
        out_channels=256,                    # Number of output channels (used at each scale)
        num_outs=5),                         # Number of output scales
```

The FPN constructs 5 multi-scale feature pyramid levels (`num_outs=5`):

- **P2 (from C2):** stride 4, 256 channels (256 × 256);
- **P3 (from C3):** stride 8, 256 channels (128 × 128);
- **P4 (from C4):** stride 16, 256 channels (64 × 64);
- **P5 (from C5):** stride 32, 256 channels (32 × 32);
- **P6 (max pool over P5):** stride 64, 256 channels (16 × 16).

### 2.3. RPN Head (`RotatedRPNHead`)

Configuration parameters in [`configs/gaucho/dotav1/gaucho_two_stage_dotav1/roi_transformer_r50_fpn_gaucho_probiou_1x_dota_le90.py` (lines 71–87)](../../configs/gaucho/dotav1/gaucho_two_stage_dotav1/roi_transformer_r50_fpn_gaucho_probiou_1x_dota_le90.py#L71-L87):

```python
    rpn_head=dict(
        type='RotatedRPNHead',
        in_channels=256,              # Number of channels in the input feature map
        feat_channels=256,            # Number of hidden channels
        version=angle_version,        # Angle representation ('le90')
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],               # Anchor scale
            ratios=[0.5, 1.0, 2.0],   # Anchor aspect ratios
            strides=[4, 8, 16, 32, 64]),  # Feature map strides corresponding to P2-P6
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),  # Config of classification loss
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),  # Config of regression loss
```

The RPN generates horizontal proposals from anchor boxes on pyramid levels P2–P6.

### 2.4. RoI Head (`RoITransRoIHead`)

Configuration parameters in [`configs/gaucho/dotav1/gaucho_two_stage_dotav1/roi_transformer_r50_fpn_gaucho_probiou_1x_dota_le90.py` (lines 88–155)](../../configs/gaucho/dotav1/gaucho_two_stage_dotav1/roi_transformer_r50_fpn_gaucho_probiou_1x_dota_le90.py#L88-L155):

```python
    roi_head=dict(
        type='RoITransRoIHead',
        version=angle_version,        # Angle representation ('le90')
        num_stages=2,                 # Number of cascade stages
        stage_loss_weights=[1, 1],    # Loss weights of cascade stages
        bbox_roi_extractor=[
            # Stage 1: Standard horizontal RoIAlign from RPN proposals
            dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),  # Spatial resolution 7x7
                out_channels=256,                                       # Output feature channels
                featmap_strides=[4, 8, 16, 32]),                        # Strides of feature maps (P2-P5)
            # Stage 2: Rotated RoIAlignRotated from oriented proposals
            dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=7,       # Spatial resolution 7x7
                    sample_num=2,     # Number of sampling points
                    clockwise=True),  # Clockwise angle orientation
                out_channels=256,     # Output feature channels
                featmap_strides=[4, 8, 16, 32]),                        # Strides of feature maps (P2-P5)
        ],
        bbox_head=[
            # Stage 1: Predicts class-agnostic oriented proposals (RRoIs) via GauCho
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,              # Number of input channels
                fc_out_channels=1024,         # Number of hidden channels in FC layers
                roi_feat_size=7,              # Spatial size of RoI features
                num_classes=num_classes,      # Number of classes (15)
                bbox_coder=dict(
                    type=coder,               # Bbox coder ('GauchoAnchorOBBDecoder')
                    horizontal=True,          # Whether base anchors are horizontal
                    angle_range=angle_version,  # Angle representation ('le90')
                    norm_factor=2,            # Regularization factor of angle
                    edge_swap=True,           # Whether to swap edges if w < h
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
                reg_class_agnostic=True,      # Shared bbox regressor across classes
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),         # Config of classification loss
                gaucho_encoding=use_gaucho,   # Enable GauCho Gaussian Cholesky parameterization
                reg_decoded_bbox=reg_decoded_bbox,  # Loss operates on decoded bounding boxes
                loss_bbox=gaussian_loss),     # Config of regression loss (GDLoss_v1)
            # Stage 2: Predicts class-specific refined oriented boxes via GauCho
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,              # Number of input channels
                fc_out_channels=1024,         # Number of hidden channels in FC layers
                roi_feat_size=7,              # Spatial size of RoI features
                num_classes=num_classes,      # Number of classes (15)
                bbox_coder=dict(
                    type=coder,               # Bbox coder ('GauchoAnchorOBBDecoder')
                    angle_range=angle_version,  # Angle representation ('le90')
                    norm_factor=None,         # Regularization factor of angle
                    edge_swap=True,           # Whether to swap edges if w < h
                    proj_xy=True,             # Whether to project center offsets according to angle
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1, 0.05]),
                reg_class_agnostic=False,     # Class-specific bbox regressor
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),         # Config of classification loss
                gaucho_encoding=use_gaucho,   # Enable GauCho Gaussian Cholesky parameterization
                reg_decoded_bbox=reg_decoded_bbox,  # Loss operates on decoded bounding boxes
                loss_bbox=gaussian_loss)      # Config of regression loss (GDLoss_v1)
        ]),
```

### 2.5. GauCho & ProbIoU Hyperparameters

Configuration parameters in [`configs/gaucho/dotav1/SOTA/roi_transformer_r50_fpn_gaucho_probiou_1x_dota_ms_rr_le90.py` (lines 13–26)](../../configs/gaucho/dotav1/SOTA/roi_transformer_r50_fpn_gaucho_probiou_1x_dota_ms_rr_le90.py#L13-L26):

```python
use_gaucho = True            # Enable GauCho Gaussian Cholesky parameterization in bbox head
coder = 'GauchoAnchorOBBDecoder'  # Bbox coder decoding Cholesky offsets into covariance elements and OBBs
reg_decoded_bbox = True      # Whether regression loss operates directly on decoded bounding boxes
gaussian_loss = dict(
    type='GDLoss_v1',        # Gaussian-based regression loss module
    gaussian_prediction=True,  # Treat input as 2D covariance matrix elements (a, b, c)
    loss_type='probiou',     # Distance metric type: ProbIoU (Probabilistic IoU)
    fun='log1p',             # Transformation function applied to distance ('log1p', 'sqrt', or '')
    tau=1.0,                 # Scale parameter for loss transformation
    loss_weight=1.0)         # Weight of regression loss
```

______________________________________________________________________

## 3. Code Components

### 3.1. Backbone (`ResNet-101`)

| Component Name       | Role & Implementation                                                       | Code Link                                                                                                                                          |
| :------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backbone**         | `ResNet-101`: 101-layer residual backbone (imported from `mmdet`).          | [mmdet.models.backbones.ResNet](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py)                            |
| **Backbone Builder** | `build_backbone`: Instantiates ResNet-101 via `ROTATED_BACKBONES` registry. | [builder.py: L15-L17](../../mmrotate/models/builder.py#L15-L17) (invoked at [two_stage.py: L32](../../mmrotate/models/detectors/two_stage.py#L32)) |

### 3.2. Neck (`FPN`)

| Component Name   | Role & Implementation                                                                  | Code Link                                                                                                                                          |
| :--------------- | :------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Neck (FPN)**   | `FPN`: Feature Pyramid Network for multi-scale feature fusion (imported from `mmdet`). | [mmdet.models.necks.FPN](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/fpn.py)                                          |
| **Neck Builder** | `build_neck`: Instantiates FPN via `ROTATED_NECKS` registry.                           | [builder.py: L20-L22](../../mmrotate/models/builder.py#L20-L22) (invoked at [two_stage.py: L35](../../mmrotate/models/detectors/two_stage.py#L35)) |

### 3.3. RPN & RoI Head / Detector (`RoITransformer`, `RotatedRPNHead` & `RoITransRoIHead`)

| Component Name              | Role & Implementation                                                                                                                 | Code Link                                                                                                                                                                                                            |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Detector**                | `RoITransformer`: Top-level two-stage rotated detector with cascaded RoI transformation.                                              | [roi_transformer.py: L7-L31](../../mmrotate/models/detectors/roi_transformer.py#L7-L31)                                                                                                                              |
| **Base Detector**           | `RotatedTwoStageDetector`: Base class managing feature extraction, proposal generation, and cascaded refinement loops.                | [two_stage.py: L11-L196](../../mmrotate/models/detectors/two_stage.py#L11-L196)                                                                                                                                      |
| **RPN Head**                | `RotatedRPNHead`: Dense head generating horizontal proposals from multi-scale features P2–P6.                                         | [rotated_rpn_head.py: L18-L534](../../mmrotate/models/dense_heads/rotated_rpn_head.py#L18-L534)                                                                                                                      |
| **RPN Builder**             | `build_head`: Instantiates RotatedRPNHead via `ROTATED_HEADS` registry.                                                               | [builder.py: L35-L37](../../mmrotate/models/builder.py#L35-L37) (invoked at [two_stage.py: L41](../../mmrotate/models/detectors/two_stage.py#L41))                                                                   |
| **RoI Head**                | `RoITransRoIHead`: Two-stage cascade RoI head managing RoI feature extraction and box prediction heads.                               | [roi_trans_roi_head.py: L14-L357](../../mmrotate/models/roi_heads/roi_trans_roi_head.py#L14-L357)                                                                                                                    |
| **RoI Head Builder**        | `build_head`: Instantiates RoITransRoIHead via `ROTATED_HEADS` registry.                                                              | [builder.py: L35-L37](../../mmrotate/models/builder.py#L35-L37) (invoked at [two_stage.py: L50](../../mmrotate/models/detectors/two_stage.py#L50))                                                                   |
| **Stage 1 RoI Extractor**   | `SingleRoIExtractor`: Extracts 7 × 7 feature patches from horizontal RPN proposals via `RoIAlign`.                                    | [single_level_roi_extractor.py](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py)                                                           |
| **Stage 2 RoI Extractor**   | `RotatedSingleRoIExtractor`: Extracts 7 × 7 feature patches aligned with oriented proposals via `RoIAlignRotated`.                    | [rotate_single_level_roi_extractor.py: L14-L168](../../mmrotate/models/roi_heads/roi_extractors/rotate_single_level_roi_extractor.py#L14-L168)                                                                       |
| **Bounding Box Head**       | `RotatedShared2FCBBoxHead`: 2-layer MLP head predicting class logits and 5 regression deltas per stage.                               | [convfc_rbbox_head.py: L209-L224](../../mmrotate/models/roi_heads/bbox_heads/convfc_rbbox_head.py#L209-L224) (GauCho loss at [L278-L285](../../mmrotate/models/roi_heads/bbox_heads/convfc_rbbox_head.py#L278-L285)) |
| **Base BBox Head**          | `RotatedBBoxHead`: Base head implementing `gaucho_encoding` handling and decoded loss routing.                                        | [rotated_bbox_head.py: L77-L81](../../mmrotate/models/roi_heads/bbox_heads/rotated_bbox_head.py#L77-L81) (loss call at [L354-L359](../../mmrotate/models/roi_heads/bbox_heads/rotated_bbox_head.py#L354-L359))       |
| **GauCho BBox Coder**       | `GauchoAnchorOBBDecoder`: Converts network outputs `(dx, dy, dα, dβ, dγ)` to Gaussian covariance parameters and OBBs.                 | [gaucho_anchor_coder.py: L13-L112](../../mmrotate/core/bbox/coder/gaucho_anchor_coder.py#L13-L112)                                                                                                                   |
| **GauCho Decode Transform** | `delta2bbox`: Performs Cholesky reconstruction `(ga, gb, gc)` and closed-form eigendecomposition to `(gx, gy, gw, gh, gt)`.           | [gaucho_anchor_coder.py: L187-L340](../../mmrotate/core/bbox/coder/gaucho_anchor_coder.py#L187-L340)                                                                                                                 |
| **Gaussian ProbIoU Loss**   | `GDLoss_v1` / `probiou_loss`: Computes Hellinger/Bhattacharyya distance between predicted and ground-truth 2D Gaussian distributions. | [gaussian_dist_loss_v1.py: L174-L210](../../mmrotate/models/losses/gaussian_dist_loss_v1.py#L174-L210) (class at [L214-L304](../../mmrotate/models/losses/gaussian_dist_loss_v1.py#L214-L304))                       |
| **Rotated NMS**             | `multiclass_nms_rotated`: Suppresses overlapping rotated bounding boxes during inference.                                             | [bbox_nms_rotated.py: L6-L114](../../mmrotate/core/post_processing/bbox_nms_rotated.py#L6-L114)                                                                                                                      |

______________________________________________________________________

## 4. ResNet-101 Backbone Breakdown

For the detailed configuration parameters of the backbone (`depth=101`, `frozen_stages=1`, `out_indices`, etc.), see [Section 2.1](#21-backbone-resnet-101).

### Layer and Parameter Specifications

| Stage       | Name           | Input Resolution | Output Channels | Stride | Forwarded to Neck  | Structure                             | Parameters               |
| :---------- | :------------- | :--------------- | :-------------- | :----- | :----------------- | :------------------------------------ | :----------------------- |
| **Stem**    | `conv1`, `bn1` | 1024 × 1024      | 64              | 4      | —                  | 7 × 7 Conv, MaxPool                   | ~9.5K (9,536)            |
| **Stage 1** | `layer1`       | 256 × 256        | 256             | 4      | **C2** (256 × 256) | 3 Bottleneck blocks (64, 64, 256)     | ~0.22M (215,808)         |
| **Stage 2** | `layer2`       | 128 × 128        | 512             | 8      | **C3** (128 × 128) | 4 Bottleneck blocks (128, 128, 512)   | ~1.22M (1,219,584)       |
| **Stage 3** | `layer3`       | 64 × 64          | 1024            | 16     | **C4** (64 × 64)   | 23 Bottleneck blocks (256, 256, 1024) | ~26.09M (26,090,496)     |
| **Stage 4** | `layer4`       | 32 × 32          | 2048            | 32     | **C5** (32 × 32)   | 3 Bottleneck blocks (512, 512, 2048)  | ~14.96M (14,964,736)     |
| **Total**   |                |                  |                 |        | **4 feature maps** | **101 Conv layers (33 blocks)**       | **42,500,160 (~42.50M)** |

The 4 feature maps (C2, C3, C4, C5) produced by `layer1`–`layer4` (`out_indices=(0, 1, 2, 3)`) are forwarded directly to the FPN neck (`in_channels=[256, 512, 1024, 2048]`).

______________________________________________________________________

## 5. Quantum Layer Integration

### 5.1. Overview

In GauCho, geometric box representations and Cholesky transformation parameters are compact 5-dimensional vectors (`[x, y, w, h, a]` and `[dx, dy, dα, dβ, dγ]`). GauCho integrates the hybrid [`QuantumLayer`](../../mmrotate/models/utils/quantum_layer.py#L69) module to refine these 5D vectors by learning joint correlations across all parameters simultaneously.

> For full details on module architecture, circuit design, and parameters, see the [Quantum Layer Module Guide](../guides/quantum_layer.md).

### 5.2. Integration Points

| Integration Point              | Code Location                                                                                                            | Input Data                                    |
| :----------------------------- | :----------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------- |
| **RoI Geometry Refinement**    | [`GauchoAnchorOBBDecoder.decode` (L123-L124)](../../mmrotate/core/bbox/coder/gaucho_anchor_coder.py#L123-L124)           | Proposal bboxes: `[x, y, w, h, a]` (5D)       |
| **Cholesky Delta Refinement**  | [`GauchoAnchorOBBDecoder.decode` (L125-L126)](../../mmrotate/core/bbox/coder/gaucho_anchor_coder.py#L125-L126)           | Predicted deltas: `[dx, dy, dα, dβ, dγ]` (5D) |
| **Regression Head Refinement** | [`RotatedConvFCBBoxHead.forward` (L206-L207)](../../mmrotate/models/roi_heads/bbox_heads/convfc_rbbox_head.py#L206-L207) | Network output: `bbox_pred` (5D)              |

#### Limitations on Target Delta Modulation (`encode`)

Theoretically, one could consider applying a quantum layer during target delta encoding ([`GauchoAnchorOBBDecoder.encode`](../../mmrotate/core/bbox/coder/gaucho_anchor_coder.py#L85)). However, in GauCho models this integration is not feasible:

1. **Bypassed during training:** In GauCho SOTA, the regression loss (ProbIoU) is calculated directly on decoded bounding boxes via `reg_decoded_bbox=True`. In [`RotatedBBoxHead._get_target_single` (L213–L221)](../../mmrotate/models/roi_heads/bbox_heads/rotated_bbox_head.py#L213-L221), when `reg_decoded_bbox=True`, targets are assigned directly from ground-truth boxes, completely bypassing `self.bbox_coder.encode(...)`.
2. **Distributed training failure:** Because `encode` is never invoked, any quantum layer instantiated there never executes, its parameters do not participate in producing the loss, and PyTorch DistributedDataParallel (DDP) raises a `RuntimeError` regarding unused parameters.
3. **Non-differentiable targets:** Ground-truth targets are constants for loss calculation rather than network outputs, meaning trainable parameters in `encode` would not receive optimization gradients even if called.

Consequently, quantum layer integration into `encode` is not supported in GauCho.

### 5.3. Configuration Usage

Quantum layers are configured via `quantum_cfg` in experiment configuration files located in [`configs/gaucho/dotav1/SOTA-quantum/`](../../configs/gaucho/dotav1/SOTA-quantum/):

- [`roi_transformer_r101_fpn_gaucho_probiou_quantum_rois_1x_dota_ms_rr_le90.py`](../../configs/gaucho/dotav1/SOTA-quantum/roi_transformer_r101_fpn_gaucho_probiou_quantum_rois_1x_dota_ms_rr_le90.py) — Refinement on RoI proposals in decoder.
- [`roi_transformer_r101_fpn_gaucho_probiou_quantum_deltas_1x_dota_ms_rr_le90.py`](../../configs/gaucho/dotav1/SOTA-quantum/roi_transformer_r101_fpn_gaucho_probiou_quantum_deltas_1x_dota_ms_rr_le90.py) — Refinement on predicted regression deltas in decoder.
- [`roi_transformer_r101_fpn_gaucho_probiou_quantum_head_1x_dota_ms_rr_le90.py`](../../configs/gaucho/dotav1/SOTA-quantum/roi_transformer_r101_fpn_gaucho_probiou_quantum_head_1x_dota_ms_rr_le90.py) — Refinement on classical regression head output.

#### Example: Decoder Integration (RoIs and Deltas)

```python
bbox_coder=dict(
    type='GauchoAnchorOBBDecoder',
    angle_range='le90',
    norm_factor=None,
    edge_swap=True,
    proj_xy=True,
    target_means=[0., 0., 0., 0., 0.],
    target_stds=[0.05, 0.05, 0.1, 0.1, 0.05],
    quantum_cfg=dict(
        # 1. Quantum refinement on RoI proposals
        rois=dict(
            type='QuantumLayer',
            input_dim=5,
            output_dim=5,
            num_qubits=5,
            reps=2,
            residual=True,
            scale_input=True,
            chunk_size=128),
        # 2. Quantum refinement on Cholesky deltas
        deltas=dict(
            type='QuantumLayer',
            input_dim=5,
            output_dim=5,
            num_qubits=5,
            reps=2,
            residual=True,
            scale_input=True,
            chunk_size=128)))
```

#### Example: Regression Head Integration

```python
dict(
    type='RotatedShared2FCBBoxHead',
    in_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    num_classes=num_classes,
    reg_class_agnostic=True,
    # Quantum refinement on regression output
    quantum_cfg=dict(
        type='QuantumLayer',
        input_dim=5,
        output_dim=5,
        num_qubits=5,
        reps=2,
        residual=True,
        scale_input=True))
```
