_base_ = ['./roi_transformer_r50_fpn_gaucho_probiou_1x_dota_ms_rr_le90.py']

# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')), )
