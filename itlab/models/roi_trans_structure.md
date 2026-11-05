# RoI Transformer Model Structure

This document details the internal architecture, configuration parameters, and code components for the RoI Transformer model applied to oriented object detection.

The structure and parameters are analyzed based on the target configuration [`configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py`](../../configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py).

______________________________________________________________________

## 1. ONNX Model Export

### ONNX Generation Command

```bash
python tools/analysis_tools/dump_model_structure.py \
    configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py \
    --out-dir model_structure/roi_trans_swin_tiny_fpn_1x_dota_le90 \
    --shape 1024 1024
```

This command generates the following decoupled `.onnx` files:

- `backbone_swintransformer.onnx`
- `neck.onnx`
- `backbone_neck.onnx`
- `roi_head.onnx`

______________________________________________________________________

## 2. Configuration Parameters Breakdown

### 2.1. Backbone (`Swin-Tiny`)

Configuration parameters in [`configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py` (lines 6–23)](../../configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py#L6-L23) (overriding ResNet-50 from [`roi_trans_r50_fpn_1x_dota_le90.py`](../../configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py#L9-L18)):

```python
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,         # Patch embedding dimension
        depths=[2, 2, 6, 2],   # Depths of each Swin Transformer stage
        num_heads=[3, 6, 12, 24],  # Number of attention heads in each stage
        window_size=7,         # Window size for window-based local self-attention
        mlp_ratio=4,           # Ratio of MLP hidden dimension to embedding dimension
        qkv_bias=True,         # If True, add a learnable bias to query, key, value
        qk_scale=None,         # Override default qk scale of head_dim ** -0.5 if set
        drop_rate=0.,          # Dropout rate
        attn_drop_rate=0.,     # Attention dropout rate
        drop_path_rate=0.2,    # Stochastic depth rate
        patch_norm=True,       # If True, add normalization after patch embedding
        out_indices=(0, 1, 2, 3),  # Output feature maps from all 4 stages
        with_cp=False,         # Use checkpointing to save memory
        convert_weights=True,  # Convert official Swin pretrained weights
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),  # Official pre-trained checkpoint
```

The backbone generates 4 multi-scale feature maps (`out_indices=(0, 1, 2, 3)`):

- **C2:** stride 4, 96 channels (H/4 × W/4, resolution 256 × 256 for 1024 × 1024 input);
- **C3:** stride 8, 192 channels (H/8 × W/8, resolution 128 × 128);
- **C4:** stride 16, 384 channels (H/16 × W/16, resolution 64 × 64);
- **C5:** stride 32, 768 channels (H/32 × W/32, resolution 32 × 32).

### 2.2. Neck (`FPN`)

Configuration parameters in [`configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py` (lines 24–29)](../../configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py#L24-L29) (overriding ResNet input channels from [`roi_trans_r50_fpn_1x_dota_le90.py` (lines 19–23)](../../configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py#L19-L23)):

```python
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[96, 192, 384, 768],  # Number of input channels per scale (matching Swin-Tiny outputs)
        out_channels=256,                 # Number of output channels (used at each scale)
        num_outs=5),                      # Number of output scales
```

The FPN constructs 5 multi-scale feature pyramid levels (`num_outs=5`):

- **P2 (from C2):** stride 4, 256 channels (256 × 256);
- **P3 (from C3):** stride 8, 256 channels (128 × 128);
- **P4 (from C4):** stride 16, 256 channels (64 × 64);
- **P5 (from C5):** stride 32, 256 channels (32 × 32);
- **P6 (max pool over P5):** stride 64, 256 channels (16 × 16).

### 2.3. RPN Head (`RotatedRPNHead`)

Configuration parameters in [`configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py` (lines 24–40)](../../configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py#L24-L40):

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

Configuration parameters in [`configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py` (lines 41–104)](../../configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py#L41-L104):

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
            # Stage 1 (RoI Learner): Predicts class-agnostic oriented proposals (RRoIs) from horizontal RoIs
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,              # Number of input channels
                fc_out_channels=1024,         # Number of hidden channels in FC layers
                roi_feat_size=7,              # Spatial size of RoI features
                num_classes=15,               # Number of classes in DOTA
                bbox_coder=dict(
                    type='DeltaXYWHAHBBoxCoder',  # Converts horizontal RoIs (HBB) to Rotated RoIs (RRoI)
                    angle_range=angle_version,    # Angle representation ('le90')
                    norm_factor=2,                # Regularization factor of angle
                    edge_swap=True,               # Whether to swap edges if w < h
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 1]),
                reg_class_agnostic=True,      # Shared bbox regressor across classes
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),         # Config of classification loss
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),  # Config of regression loss
            # Stage 2 (RoI Refiner): Predicts class-specific refined oriented bounding boxes (OBBs)
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,              # Number of input channels
                fc_out_channels=1024,         # Number of hidden channels in FC layers
                roi_feat_size=7,              # Spatial size of RoI features
                num_classes=15,               # Number of classes in DOTA
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',  # Converts Rotated RoIs (RRoI) to refined OBBs
                    angle_range=angle_version,    # Angle representation ('le90')
                    norm_factor=None,             # Regularization factor of angle
                    edge_swap=True,               # Whether to swap edges if w < h
                    proj_xy=True,                 # Whether to project center offsets according to angle
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1, 0.5]),
                reg_class_agnostic=False,     # Class-specific bbox regressor (15 x 5 deltas)
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),         # Config of classification loss
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0))   # Config of regression loss
        ]),
```

### 2.5. Optimizer & Training Hyperparameters (`AdamW`)

Configuration parameters in [`configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py` (lines 35–46)](../../configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py#L35-L46):

```python
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,               # Base learning rate for Swin Transformer
    betas=(0.9, 0.999),       # AdamW beta coefficients
    weight_decay=0.05,        # Weight decay rate
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),        # Zero decay for absolute position embeddings
            'relative_position_bias_table': dict(decay_mult=0.),  # Zero decay for relative position bias
            'norm': dict(decay_mult=0.)                       # Zero decay for normalization layers
        }))
```

______________________________________________________________________

## 3. Code Components

### 3.1. Backbone (`Swin-Tiny`)

| Component Name       | Role & Implementation                                                                            | Code Link                                                                                                                                          |
| :------------------- | :----------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backbone**         | `SwinTransformer`: Hierarchical Vision Transformer with shifted windows (imported from `mmdet`). | [mmdet.models.backbones.SwinTransformer](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/swin.py)                     |
| **Backbone Builder** | `build_backbone`: Instantiates SwinTransformer via `ROTATED_BACKBONES` registry.                 | [builder.py: L15-L17](../../mmrotate/models/builder.py#L15-L17) (invoked at [two_stage.py: L32](../../mmrotate/models/detectors/two_stage.py#L32)) |

### 3.2. Neck (`FPN`)

| Component Name   | Role & Implementation                                                                  | Code Link                                                                                                                                          |
| :--------------- | :------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Neck (FPN)**   | `FPN`: Feature Pyramid Network for multi-scale feature fusion (imported from `mmdet`). | [mmdet.models.necks.FPN](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/fpn.py)                                          |
| **Neck Builder** | `build_neck`: Instantiates FPN via `ROTATED_NECKS` registry.                           | [builder.py: L20-L22](../../mmrotate/models/builder.py#L20-L22) (invoked at [two_stage.py: L35](../../mmrotate/models/detectors/two_stage.py#L35)) |

### 3.3. RPN & RoI Head / Detector (`RoITransformer`, `RotatedRPNHead` & `RoITransRoIHead`)

| Component Name            | Role & Implementation                                                                                                     | Code Link                                                                                                                                                                                                |
| :------------------------ | :------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Detector**              | `RoITransformer`: Top-level two-stage rotated detector with cascaded RoI transformation.                                  | [roi_transformer.py: L7-L31](../../mmrotate/models/detectors/roi_transformer.py#L7-L31)                                                                                                                  |
| **Base Detector**         | `RotatedTwoStageDetector`: Base class managing feature extraction, proposal generation, and cascaded refinement loops.    | [two_stage.py: L11-L196](../../mmrotate/models/detectors/two_stage.py#L11-L196)                                                                                                                          |
| **RPN Head**              | `RotatedRPNHead`: Dense head generating horizontal proposals from multi-scale features P2–P6.                             | [rotated_rpn_head.py: L18-L534](../../mmrotate/models/dense_heads/rotated_rpn_head.py#L18-L534)                                                                                                          |
| **RPN Builder**           | `build_head`: Instantiates RotatedRPNHead via `ROTATED_HEADS` registry.                                                   | [builder.py: L35-L37](../../mmrotate/models/builder.py#L35-L37) (invoked at [two_stage.py: L41](../../mmrotate/models/detectors/two_stage.py#L41))                                                       |
| **RoI Head**              | `RoITransRoIHead`: Two-stage cascade RoI head managing RoI feature extraction and box prediction heads.                   | [roi_trans_roi_head.py: L14-L357](../../mmrotate/models/roi_heads/roi_trans_roi_head.py#L14-L357)                                                                                                        |
| **RoI Head Builder**      | `build_head`: Instantiates RoITransRoIHead via `ROTATED_HEADS` registry.                                                  | [builder.py: L35-L37](../../mmrotate/models/builder.py#L35-L37) (invoked at [two_stage.py: L50](../../mmrotate/models/detectors/two_stage.py#L50))                                                       |
| **Stage 1 RoI Extractor** | `SingleRoIExtractor`: Extracts 7 × 7 feature patches from horizontal RPN proposals via `RoIAlign`.                        | [single_level_roi_extractor.py](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py)                                               |
| **Stage 2 RoI Extractor** | `RotatedSingleRoIExtractor`: Extracts 7 × 7 feature patches aligned with oriented proposals via `RoIAlignRotated`.        | [rotate_single_level_roi_extractor.py: L14-L168](../../mmrotate/models/roi_heads/roi_extractors/rotate_single_level_roi_extractor.py#L14-L168)                                                           |
| **Bounding Box Head**     | `RotatedShared2FCBBoxHead`: 2-layer MLP head predicting class logits and 5 regression deltas per stage.                   | [convfc_rbbox_head.py: L209-L224](../../mmrotate/models/roi_heads/bbox_heads/convfc_rbbox_head.py#L209-L224)                                                                                             |
| **Base BBox Head**        | `RotatedBBoxHead`: Base head implementing loss computation and regression targets.                                        | [rotated_bbox_head.py: L18-L534](../../mmrotate/models/roi_heads/bbox_heads/rotated_bbox_head.py#L18-L534)                                                                                               |
| **Stage 1 BBox Coder**    | `DeltaXYWHAHBBoxCoder`: Encodes horizontal boxes into rotated deltas and decodes horizontal RoIs to Rotated RoIs (RRoIs). | [delta_xywha_hbbox_coder.py: L12-L113](../../mmrotate/core/bbox/coder/delta_xywha_hbbox_coder.py#L12-L113) (decoder at [L183-L273](../../mmrotate/core/bbox/coder/delta_xywha_hbbox_coder.py#L183-L273)) |
| **Stage 2 BBox Coder**    | `DeltaXYWHAOBBoxCoder`: Encodes and decodes rotated bounding boxes (OBBs) with angle regularization and projection.       | [delta_xywha_rbbox_coder.py: L12-L108](../../mmrotate/core/bbox/coder/delta_xywha_rbbox_coder.py#L12-L108) (decoder at [L185-L288](../../mmrotate/core/bbox/coder/delta_xywha_rbbox_coder.py#L185-L288)) |
| **Classification Loss**   | `CrossEntropyLoss`: Multi-class cross-entropy loss for classification in RPN and both RoI stages (imported from `mmdet`). | [mmdet.models.losses.CrossEntropyLoss](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/cross_entropy_loss.py)                                                                  |
| **Regression Loss**       | `SmoothL1Loss`: Smooth L1 loss for bounding box regression in RPN and both RoI stages (imported from `mmdet`).            | [mmdet.models.losses.SmoothL1Loss](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py)                                                                          |
| **Rotated NMS**           | `multiclass_nms_rotated`: Suppresses overlapping rotated bounding boxes during inference.                                 | [bbox_nms_rotated.py: L6-L94](../../mmrotate/core/post_processing/bbox_nms_rotated.py#L6-L94)                                                                                                            |

______________________________________________________________________

## 4. Swin-Tiny Backbone Breakdown

For the detailed configuration parameters of the backbone (`embed_dims=96`, `depths=[2, 2, 6, 2]`, `num_heads=[3, 6, 12, 24]`, `window_size=7`), see [Section 2.1](#21-backbone-swin-tiny).

### Layer and Parameter Specifications

| Stage       | Sub-modules                                    | Input Resolution | Output Channels | Stride | Forwarded to Neck  | Structure                                                | Parameters               |
| :---------- | :--------------------------------------------- | :--------------- | :-------------- | :----- | :----------------- | :------------------------------------------------------- | :----------------------- |
| **Stage 1** | `patch_embed`, `stages[0]` (2 blocks), `norm0` | 1024 × 1024      | 96              | 4      | **C2** (256 × 256) | PatchEmbed (4 × 4) + 2 Swin Blocks (dim 96, 3 heads)     | ~0.23M (229,782)         |
| **Stage 2** | `stages[1].downsample`, `stages[1]` (2 blocks) | 256 × 256        | 192             | 8      | **C3** (128 × 128) | PatchMerging (2 × 2) + 2 Swin Blocks (dim 192, 6 heads)  | ~0.97M (966,636)         |
| **Stage 3** | `stages[2].downsample`, `stages[2]` (6 blocks) | 128 × 128        | 384             | 16     | **C4** (64 × 64)   | PatchMerging (2 × 2) + 6 Swin Blocks (dim 384, 12 heads) | ~10.96M (10,956,168)     |
| **Stage 4** | `stages[3].downsample`, `stages[3]` (2 blocks) | 64 × 64          | 768             | 32     | **C5** (32 × 32)   | PatchMerging (2 × 2) + 2 Swin Blocks (dim 768, 24 heads) | ~15.37M (15,368,112)     |
| **Total**   |                                                |                  |                 |        | **4 feature maps** | **12 Swin Blocks (4 stages)**                            | **27,520,698 (~27.52M)** |

The 4 feature maps (C2, C3, C4, C5) produced by stages 1–4 (`out_indices=(0, 1, 2, 3)`) are forwarded directly to the FPN neck (`in_channels=[96, 192, 384, 768]`).
