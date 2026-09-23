_base_ = [
    '../gaucho_two_stage_dotav1/'
    'roi_transformer_r50_fpn_gaucho_probiou_1x_dota_le90.py'
]

################################################
################################################

angle_version = 'le90'

num_classes = 15

use_gaucho = True

coder = 'GauchoAnchorOBBDecoder'

reg_decoded_bbox = True

gaussian_loss = dict(
    type='GDLoss_v1',
    gaussian_prediction=True,
    loss_type='probiou',
    fun='log1p',
    tau=1.0,
    loss_weight=1.0)

dataset_type = 'DOTADataset'
data_root = 'data/split_ms_dota_1_0/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9, 11],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=train_pipeline,
        version=angle_version),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        version=angle_version),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/',
        version=angle_version))

################################################
################################################

model = dict(
    roi_head=dict(bbox_head=[
        dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHAHBBoxCoder',
                angle_range=angle_version,
                norm_factor=2,
                edge_swap=True,
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type=coder,
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1, 0.05]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            gaucho_encoding=use_gaucho,
            reg_decoded_bbox=reg_decoded_bbox,
            loss_bbox=gaussian_loss)
    ]), )
