_base_ = ['./roi_trans_swin_tiny_fpn_1x_dota_le90.py']

dataset_type = 'DOTADataset'
data_root_train = 'data/split_ms-roi-train_dota_1_0/'
data_root_test = 'data/split_ms-roi-test_dota_1_0/'

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root_train + 'trainval/annfiles/',
        img_prefix=data_root_train + 'trainval/images/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root_train + 'trainval/annfiles/',
        img_prefix=data_root_train + 'trainval/images/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root_test + 'test/images/',
        img_prefix=data_root_test + 'test/images/'))
