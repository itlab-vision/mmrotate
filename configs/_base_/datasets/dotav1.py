# dataset settings
dataset_type = 'DOTADataset'
# dataset_type = 'DOTAv15Dataset'

# Uncomment the required option
data_root = 'data/split_ss_dota_1_0/'
# data_root = 'data/split_ss_dota_1_5/'
# data_root = 'data/split_ss_dota_2_0/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,  # only works for train_dataloader
    workers_per_gpu=2,  # works for any _dataloader
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/images/',   # replace with actual testing set
        img_prefix=data_root + 'test/images/',   # replace with actual testing set
        pipeline=test_pipeline),
    # train_dataloader=dict(samples_per_gpu=2, workers_per_gpu=2),
    # val_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=1),
)
