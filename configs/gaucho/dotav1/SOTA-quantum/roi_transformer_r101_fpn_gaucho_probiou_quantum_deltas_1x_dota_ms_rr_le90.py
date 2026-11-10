_base_ = [
    '../SOTA/roi_transformer_r101_fpn_gaucho_probiou_1x_dota_ms_rr_le90.py'
]

num_classes = 15

quantum_layer_cfg = dict(
    type='QuantumLayer',
    input_dim=5,
    output_dim=5,
    num_qubits=5,
    reps=2,
    residual=True,
    scale_input=True,
    chunk_size=128)

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
                angle_range='le90',
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
                type='GauchoAnchorOBBDecoder',
                angle_range='le90',
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1, 0.05],
                quantum_cfg=dict(deltas=quantum_layer_cfg)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            gaucho_encoding=True,
            reg_decoded_bbox=True,
            loss_bbox=dict(
                type='GDLoss_v1',
                gaussian_prediction=True,
                loss_type='probiou',
                fun='log1p',
                tau=1.0,
                loss_weight=1.0))
    ]))
