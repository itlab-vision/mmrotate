_base_ = ['../rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py']

model = dict(
    # Enable CFA (Convex-hull Feature Adaptation)
    bbox_head=dict(use_reassign=True),
    train_cfg=dict(
        refine=dict(
            # Lower IoU threshold to collect a candidate pool for reassign
            assigner=dict(pos_iou_thr=0.1, neg_iou_thr=0.1))))
