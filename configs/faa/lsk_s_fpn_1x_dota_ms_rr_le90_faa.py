_base_ = ['./lsk_s_fpn_1x_dota_rr_le90_faa.py']

# TOP-2 DOTA v1.0 model
dataset_type = 'DOTADataset'
data_root = 'data/split_ms_dota_1_0/'
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
