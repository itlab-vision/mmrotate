_base_ = ['./cfa_r50_fpn_40e_dota_oc.py']

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')), )
