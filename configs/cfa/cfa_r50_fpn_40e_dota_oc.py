_base_ = ['./cfa_r50_fpn_1x_dota_oc.py']

# evaluation
evaluation = dict(interval=5, metric='mAP')
# learning policy
lr_config = dict(step=[24, 32, 38])
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=5)
