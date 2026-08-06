# FAA

## Overview

This configs uses FAA with pretrained backbone weights. The models depends on external pretrained backbones and does not include them in the repository.

## Required pretrained backbones

1. **LSKNet-S backbone**
   - Source: ImageNet 300-epoch pretrained weights
   - Download: https://huggingface.co/GreatBird/LSKNet/resolve/main/lsk_s_backbone.pth.tar?download=true
   - Save as: `data/pretrained/lsk_s_backbone.pth.tar`

2. **Strip R-CNN-S backbone**
   - Source: ImageNet 300-epoch pretrained weights
   - Download: https://drive.google.com/uc?export=download&id=1_c2aXANKHl0cIBb370LNIkCyDmQpA3_o
   - Save as: `data/pretrained/stripnet_s.pth`

## References

- Original FAA repository: https://github.com/gcy0423/Fourier-Angle-Alignment
- Pretrained backbones from: https://github.com/zcablii/LSKNet

