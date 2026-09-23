_base_ = ['./lsk_s_fpn_1x_dota15_rr_le90_faa.py']

dataset_type = 'DOTAv15Dataset'
data_root = 'data/split_ms_dota_1_5/'
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
