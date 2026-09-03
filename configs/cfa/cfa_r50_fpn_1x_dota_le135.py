_base_ = ['../rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_le135.py']

model = dict(
    bbox_head=dict(use_reassign=True),
    train_cfg=dict(
        refine=dict(assigner=dict(pos_iou_thr=0.1, neg_iou_thr=0.1))))

dataset_type = 'DOTADataset'
data_root = 'data/split_ss-cfa_dota_1_0/'
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/'))
