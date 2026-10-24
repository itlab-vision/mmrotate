# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint

from mmrotate.models import build_detector

logger = logging.getLogger('structure_exporter')


class RotatedNMSOp(torch.autograd.Function):
    """Custom ONNX symbolic operator representing Rotated NMS."""

    @staticmethod
    def forward(ctx,
                bboxes,
                scores,
                score_thr=0.05,
                iou_thr=0.1,
                max_per_img=2000):
        batch_size = bboxes.shape[0]
        return torch.zeros(batch_size, max_per_img, 6, device=bboxes.device)

    @staticmethod
    def symbolic(g,
                 bboxes,
                 scores,
                 score_thr=0.05,
                 iou_thr=0.1,
                 max_per_img=2000):
        return g.op(
            'mmrotate::RotatedNMS',
            bboxes,
            scores,
            score_thr_f=score_thr,
            iou_thr_f=iou_thr,
            max_per_img_i=max_per_img)


class ModelStructureExporter:
    """Exports decoupled model subsystems (backbone, neck, heads) into ONNX."""

    def __init__(self,
                 model: nn.Module,
                 out_dir: str,
                 input_shape: tuple = (1024, 1024)):
        self.model = model
        self.out_dir = out_dir
        self.h, self.w = input_shape
        self.model.eval()

    @contextmanager
    def _patch_faafusion(self, module: nn.Module):
        """Temporarily bypasses non-exportable 2D FFT and unfold in FAAFusion
        while keeping all learned weights and convolution pathways."""
        patched_modules = []
        for m in module.modules():
            if m.__class__.__name__ == 'FAAFusion':
                orig_fwd = m.forward

                def make_vis_fwd(mod):

                    def vis_fwd(x_high, x_low):
                        _, _, H_l, W_l = x_low.shape
                        _, _, H_h, W_h = x_high.shape
                        if (H_h, W_h) != (H_l, W_l):
                            x_high_up = nn.functional.interpolate(
                                x_high,
                                size=(H_l, W_l),
                                mode='bilinear',
                                align_corners=False)
                        else:
                            x_high_up = x_high
                        xl_proj = mod.proj_low(x_low)
                        xh_proj = mod.proj_high(x_high_up)
                        xh_aligned = xh_proj + xl_proj
                        xh_recon = mod.recon(xh_aligned)
                        x_high_mod = mod.layer_scale * xh_recon + x_high_up
                        return x_low + x_high_mod

                    return vis_fwd

                m.forward = make_vis_fwd(m)
                patched_modules.append((m, orig_fwd))
        try:
            yield
        finally:
            for m, orig_fn in patched_modules:
                m.forward = orig_fn

    def _export_backbone(self) -> str:
        """Exports pure backbone to ONNX."""
        bb_name = self.model.backbone.__class__.__name__.lower()
        bb_path = os.path.join(self.out_dir, f'backbone_{bb_name}.onnx')
        dummy_input = torch.randn(1, 3, self.h, self.w)

        out_indices = getattr(self.model.backbone, 'out_indices', (0, 1, 2, 3))
        output_names = [f'c{idx + 2}_out' for idx in out_indices]

        try:
            torch.onnx.export(
                self.model.backbone,
                dummy_input,
                bb_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input_image'],
                output_names=output_names)
            logger.info(f'Exported backbone: {bb_path}')
            return bb_path
        except Exception as e:
            logger.error(f'Failed to export backbone: {e}')
            return None

    def _export_neck(self) -> str:
        """Exports pure neck to ONNX."""
        if not hasattr(self.model, 'neck') or self.model.neck is None:
            return None

        neck = self.model.neck
        in_channels = getattr(neck, 'in_channels', [256, 512, 1024, 2048])
        input_names = [f'c{i + 2}_in' for i in range(len(in_channels))]
        dummy_inputs = tuple(
            torch.randn(1, ch, self.h // (4 * (2**i)), self.w // (4 * (2**i)))
            for i, ch in enumerate(in_channels))

        class NeckWrapper(nn.Module):

            def __init__(self, neck_module):
                super().__init__()
                self.neck = neck_module

            def forward(self, *feats):
                return self.neck(feats)

        neck_model = NeckWrapper(neck)
        onnx_path = os.path.join(self.out_dir, 'neck.onnx')

        start_level = getattr(neck, 'start_level', 0)
        num_outs = getattr(neck, 'num_outs', 5)
        start_p = 2 + start_level
        out_names = [f'p{start_p + i}_out' for i in range(num_outs)]

        try:
            with self._patch_faafusion(neck):
                torch.onnx.export(
                    neck_model,
                    dummy_inputs,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=out_names)
            logger.info(f'Exported neck: {onnx_path}')
            return onnx_path
        except Exception as e:
            logger.error(f'Failed to export neck: {e}')
            return None

    def _export_feature_extractor(self) -> str:
        """Exports combined Backbone + Neck feature extractor to ONNX."""

        class FeatureExtractorWrapper(nn.Module):

            def __init__(self, detector):
                super().__init__()
                self.backbone = detector.backbone
                self.neck = detector.neck if hasattr(
                    detector, 'neck') and detector.neck is not None else None

            def forward(self, x):
                feats = self.backbone(x)
                if self.neck is not None:
                    feats = self.neck(feats)
                return feats

        feature_model = FeatureExtractorWrapper(self.model)
        onnx_path = os.path.join(self.out_dir, 'backbone_neck.onnx')
        dummy_input = torch.randn(1, 3, self.h, self.w)

        start_level = getattr(self.model.neck, 'start_level', 0) if hasattr(
            self.model, 'neck') and self.model.neck is not None else 0
        num_outs = getattr(self.model.neck, 'num_outs', 5) if hasattr(
            self.model, 'neck') and self.model.neck is not None else 5
        start_p = 2 + start_level
        out_names = [f'p{start_p + i}_out' for i in range(num_outs)]

        try:
            with self._patch_faafusion(self.model):
                torch.onnx.export(
                    feature_model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input_image'],
                    output_names=out_names,
                    dynamic_axes={
                        'input_image': {
                            0: 'batch_size',
                            2: 'height',
                            3: 'width'
                        }
                    })
            logger.info(f'Exported feature extractor: {onnx_path}')
            return onnx_path
        except Exception as e:
            logger.error(f'Failed to export feature extractor: {e}')
            return None

    def _export_roi_head(self) -> str:
        """Exports two-stage RoI head with RotatedNMS to ONNX."""
        if not hasattr(self.model, 'roi_head') or not hasattr(
                self.model.roi_head, 'bbox_head'):
            return None

        bbox_head = self.model.roi_head.bbox_head
        bbox_head.eval()

        class RoIHeadWrapper(nn.Module):

            def __init__(self, head):
                super().__init__()
                self.head = head

            def forward(self, x):
                x = x.flatten(1)
                if hasattr(self.head, 'shared_fcs'):
                    for fc in self.head.shared_fcs:
                        x = self.head.relu(fc(x))
                cls_score = self.head.fc_cls(x) if hasattr(
                    self.head, 'fc_cls') else x
                bbox_pred = self.head.fc_reg(x) if hasattr(
                    self.head, 'fc_reg') else x

                return RotatedNMSOp.apply(
                    bbox_pred.unsqueeze(0), cls_score.unsqueeze(0), 0.05, 0.1,
                    1000)

        roi_model = RoIHeadWrapper(bbox_head)
        onnx_path = os.path.join(self.out_dir, 'roi_head.onnx')
        dummy_roi = torch.randn(10, 256, 7, 7)

        try:
            torch.onnx.export(
                roi_model,
                dummy_roi,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['roi_pooled_features_7x7'],
                output_names=['final_detections'],
                dynamic_axes={'roi_pooled_features_7x7': {
                    0: 'num_rois'
                }})
            logger.info(f'Exported RoI head: {onnx_path}')
            return onnx_path
        except Exception as e:
            logger.error(f'Failed to export RoI head: {e}')
            return None

    def _export_dense_head(self) -> str:
        """Exports single-stage dense head with feature unification and
        RotatedNMS to ONNX."""
        if hasattr(self.model,
                   'roi_head') or not hasattr(self.model, 'bbox_head'):
            return None

        bbox_head = self.model.bbox_head
        bbox_head.eval()

        class DenseHeadWrapper(nn.Module):

            def __init__(self, head):
                super().__init__()
                self.head = head

            def forward(self, *feats):
                outs = self.head(feats)
                if len(outs) == 3:
                    cls_scores, pts_init, bbox_preds = outs
                elif len(outs) == 2:
                    cls_scores, bbox_preds = outs
                else:
                    cls_scores = outs[0]
                    bbox_preds = outs[-1]

                cls_flat = [
                    c.permute(0, 2, 3, 1).reshape(c.shape[0], -1, c.shape[1])
                    for c in cls_scores
                ]
                all_cls_scores = torch.cat(cls_flat, dim=1)

                bbox_flat = [
                    b.permute(0, 2, 3, 1).reshape(b.shape[0], -1, b.shape[1])
                    for b in bbox_preds
                ]
                all_bbox_preds = torch.cat(bbox_flat, dim=1)

                return RotatedNMSOp.apply(all_bbox_preds, all_cls_scores, 0.05,
                                          0.1, 2000)

        dense_model = DenseHeadWrapper(bbox_head)
        onnx_path = os.path.join(self.out_dir, 'dense_head.onnx')

        start_level = getattr(self.model.neck, 'start_level', 0) if hasattr(
            self.model, 'neck') and self.model.neck is not None else 0
        num_outs = getattr(self.model.neck, 'num_outs', 5) if hasattr(
            self.model, 'neck') and self.model.neck is not None else 5
        feat_channels = getattr(bbox_head, 'in_channels', 256)
        start_p = 2 + start_level
        input_names = [f'p{start_p + i}_in' for i in range(num_outs)]

        dummy_feats = tuple(
            torch.randn(1, feat_channels, self.h //
                        (2**(start_p + i)), self.w // (2**(start_p + i)))
            for i in range(num_outs))

        try:
            torch.onnx.export(
                dense_model,
                dummy_feats,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=['final_detections'])
            logger.info(f'Exported dense head: {onnx_path}')
            return onnx_path
        except Exception as e:
            logger.error(f'Failed to export dense head: {e}')
            return None

    def export_all(self):
        """Exports all detected model subsystems."""
        exported = []
        fn_bb = self._export_backbone()
        if fn_bb:
            exported.append(fn_bb)

        fn_neck = self._export_neck()
        if fn_neck:
            exported.append(fn_neck)

        fn_feat = self._export_feature_extractor()
        if fn_feat:
            exported.append(fn_feat)

        fn_roi = self._export_roi_head()
        if fn_roi:
            exported.append(fn_roi)

        fn_dense = self._export_dense_head()
        if fn_dense:
            exported.append(fn_dense)

        logger.info('\nExported ONNX models:')
        for item in exported:
            logger.info(f'  - {item}')
        return exported


def init_logger():
    """Initializes standard logger for console output."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)


def parse_args():
    desc = 'Export MMRotate model architecture into modular ONNX files.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        'config', help='Path to model configuration file (.py)')
    parser.add_argument(
        '--checkpoint', default=None, help='Path to checkpoint file (.pth)')
    parser.add_argument(
        '--out-dir',
        default='onnx_models',
        help='Directory to save exported ONNX models')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1024, 1024],
        help='Input image dimensions (H, W)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options in key=value format')
    return parser.parse_args()


def main():
    init_logger()
    args = parse_args()

    logger.info(f'Loading config: {args.config}')
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    os.makedirs(args.out_dir, exist_ok=True)

    logger.info('Building detector...')
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    if args.checkpoint:
        logger.info(f'Loading checkpoint: {args.checkpoint}')
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.eval()

    h, w = args.shape if len(args.shape) == 2 else (args.shape[0],
                                                    args.shape[0])
    exporter = ModelStructureExporter(model, args.out_dir, input_shape=(h, w))
    exporter.export_all()


if __name__ == '__main__':
    main()
