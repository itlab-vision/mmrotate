## Model Zoo

***Note:** All pre-trained weights for the models evaluated in this benchmark were obtained from the official OpenMMLab resource (`download.openmmlab.com`). These models were originally trained on the DOTA v1.0 dataset.*

- [Rotated RetinaNet-OBB/HBB](https://github.com/open-mmlab/mmrotate/tree/main/configs/rotated_retinanet/README.md) (ICCV'2017)
- [Rotated FasterRCNN-OBB](https://github.com/open-mmlab/mmrotate/tree/main/configs/rotated_faster_rcnn/README.md) (TPAMI'2017)
- [Rotated RepPoints-OBB](https://github.com/open-mmlab/mmrotate/tree/main/configs/rotated_reppoints/README.md) (ICCV'2019)
- [Rotated FCOS](https://github.com/open-mmlab/mmrotate/tree/main/configs/rotated_fcos/README.md) (ICCV'2019)
- [RoI Transformer](https://github.com/open-mmlab/mmrotate/tree/main/configs/roi_trans/README.md) (CVPR'2019)
- [Gliding Vertex](https://github.com/open-mmlab/mmrotate/tree/main/configs/gliding_vertex/README.md) (TPAMI'2020)
- [Rotated ATSS-OBB](https://github.com/open-mmlab/mmrotate/tree/main/configs/rotated_atss/README.md) (CVPR'2020)
- [CSL](https://github.com/open-mmlab/mmrotate/tree/main/configs/csl/README.md) (ECCV'2020)
- [R<sup>3</sup>Det](https://github.com/open-mmlab/mmrotate/tree/main/configs/r3det/README.md) (AAAI'2021)
- [S<sup>2</sup>A-Net](https://github.com/open-mmlab/mmrotate/tree/main/configs/s2anet/README.md) (TGRS'2021)
- [ReDet](https://github.com/open-mmlab/mmrotate/tree/main/configs/redet/README.md) (CVPR'2021)
- [Beyond Bounding-Box](https://github.com/open-mmlab/mmrotate/tree/main/configs/cfa/README.md) (CVPR'2021)
- [Oriented R-CNN](https://github.com/open-mmlab/mmrotate/tree/main/configs/oriented_rcnn/README.md) (ICCV'2021)
- [GWD](https://github.com/open-mmlab/mmrotate/tree/main/configs/gwd/README.md) (ICML'2021)
- [KLD](https://github.com/open-mmlab/mmrotate/tree/main/configs/kld/README.md) (NeurIPS'2021)
- [SASM](configs/sasm_reppoints/README.md) (AAAI'2022)
- [Oriented RepPoints](https://github.com/open-mmlab/mmrotate/tree/main/configs/oriented_reppoints/README.md) (CVPR'2022)
- [KFIoU](https://github.com/open-mmlab/mmrotate/tree/main/configs/kfiou/README.md) (arXiv)
- [G-Rep](https://github.com/open-mmlab/mmrotate/tree/main/configs/g_reppoints/README.md) (stay tuned)

## Benchmark

### 1. Environment & Dependencies

All experiments were conducted on a high-performance computing cluster running CentOS Linux 8, managed by the Slurm Workload Manager. 

#### Hardware Configuration
The evaluation was executed on a high-performance compute node with the following overall specifications:

| Component | Specification |
| :--- | :--- |
| **OS** | CentOS Linux 8 |
| **GPU** | 8x NVIDIA A100-PCIE-40GB |
| **CPU** | AMD EPYC 7742 64-Core Processor |
| **RAM** | 512 GB |
| **Scheduler** | Slurm Workload Manager 24.11.5 |

**Actual Resource Allocation:**
While the compute node possesses the capacity listed above, the specific resources requested and allocated via Slurm for benchmarking jobs were strictly limited to:
- **1x GPU**
- **6x CPU cores** 
- **Unrestricted memory** (no specific RAM limit was requested, meaning the jobs operated without strict memory constraints)


#### Software & Libraries
All models were launched within an isolated Miniconda3 environment to ensure reproducibility. The core dependencies and their respective versions are listed below:

| Software / Package | Version |
| :--- | :--- |
| **OS** | CentOS Linux 8 |
| **CUDA Toolkit** | 11.1 |
| **Python** | Python 3.8.20 |
| **PyTorch** | 1.8.0 |
| **TorchVision** | 0.9.0 |
| **MMCV / MMCV-full** | 1.7.2|
| **MMDetection** | 2.28.2 |
| **MMRotate** | 0.3.4 |

### 2. Datasets & Preprocessing

The experiments are based on the **DOTA (Dataset for Object Detection in Aerial Images)**. Aerial images are typically characterized by massive resolutions (e.g., 4000×4000 pixels or more) and objects of highly varied scales and orientations. 

#### Dataset Versions
- **DOTA v1.0:** Contains 2,806 large-scale aerial images with 15 object categories.
- **DOTA v1.5:** Uses the same images as v1.0 but features updated and refined annotations. It introduces a new class (*container crane*, making it 16 classes in total) and includes annotations for extremely small object instances (less than 10 pixels).

#### Data Split & Evaluation Policy
The DOTA dataset is officially split into `training`, `validation`, and `testing` sets. Because the ground truth annotations for the `testing` set are closed-source and require submission to the official DOTA evaluation server, all local evaluations and FPS benchmarks in this document were strictly performed on the **`validation`** split.

#### Image Cropping Strategies (SS vs. MS)
Due to the massive resolution of aerial imagery, images cannot be fed directly into standard CNNs. Before training and evaluation, the original images are processed into smaller patches (e.g., 1024×1024) with a specific overlap. Two main strategies are used:
- **Single-Scale (SS):** The original images are cropped into patches at their original scale (no resizing before cropping). 
- **Multi-Scale (MS):** The original images are first resized to multiple scaling factors (e.g., 0.5, 1.0, 1.5) and then cropped into patches. This provides the model with rich scale invariance during training.

***Note:** The model's evaluation is strictly performed using the same cropping strategy (SS or MS) that was utilized during its training phase. This is denoted in the `Scale` parameter of the benchmark table.*

#### Downloads
The datasets are available for download below. 

| Dataset | Download Link |
| :--- | :--- |
| **DOTA v1.0** | [Google Drive Link](https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224?usp=sharing) |
| **DOTA v1.5** | [Google Drive Link](https://drive.google.com/drive/folders/1n5w45suVOyaqY84hltJhIZdtVFD9B224?usp=sharing) |

Alternative download methods and official mirrors can be found on the [Official DOTA Website](https://captain-whu.github.io/DOTA/dataset.html).

***Note:** The datasets are provided in their original base format. The image cropping for Single-Scale (SS) and Multi-Scale (MS) evaluation is performed locally using the `tools/data/dota/split/img_split.py` script prior to training and testing.*

### 3. Model Configurations & Evaluation Metrics

To clarify the exact setups and metrics used for the models in the benchmark, the following parameters define the core strategies and measurements applied:

- **Speed (FPS):** Denotes the model's inference speed measured with `batch_size=1`. The FPS calculation encompasses the full inference pipeline, including both the network's forward pass and post-processing steps. Note that several warm-up iterations are performed initially and are not included in the final timing.
- **Scale (Cropping Scale):** Indicates the dataset preprocessing applied (as described above). 
  - `-` denotes **Single-Scale (SS)**.
  - `MS` denotes **Multi-Scale**.
- **Rotation (Augmentation):** Refers to the data augmentation applied dynamically during the training pipeline. 
  - `-` denotes **no rotation augmentation**.
  - `RR` denotes **Random Rotation**. When enabled, both the images and their corresponding bounding box coordinates are randomly rotated by arbitrary angles (e.g., uniformly between $[0, 360)$ degrees) on the fly before being fed into the network. This vastly improves the model's rotation invariance.
- **Angle (Bounding Box Representation):** Oriented bounding boxes (OBB) can be mathematically parameterized in several ways. For a detailed explanation of the angle definitions (e.g., `oc`, `le90`, `le135`) expected by the specific model's detection head, please refer to the [Definition of Rotated Box](../docs/en/intro.md#definition-of-rotated-box) section in documentation.


### 4. Benchmark Results

#### Results on DOTA v1.0

***Note:** The `Ref mAP (%)` column displays official baseline metrics extracted from the `metafile.yml` files located in the `configs/` directory. These reference values were likely obtained on the test split by submitting predictions to the official DOTA evaluation server.*

| Family | Model Name | mAP (%) | Ref mAP (%) | FPS | Scale | Rotation | Angle | Config | Download |
|---|---|---|---|---|---|---|---|---|---|
| roi_trans | `roi_trans_swin_tiny_fpn_1x_dota_le90` | 86.49 | 77.51 | 24.6 | - | - | le90 | [config](../../configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90/roi_trans_swin_tiny_fpn_1x_dota_le90-ddeee9ae.pth) |
| roi_trans | `roi_trans_r50_fpn_fp16_1x_dota_le90` | 85.12 | 75.75 | 24.5 | - | - | le90 | [config](../../configs/roi_trans/roi_trans_r50_fpn_fp16_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_fp16_1x_dota_le90/roi_trans_r50_fpn_fp16_1x_dota_le90-62eb88b1.pth) |
| roi_trans | `roi_trans_r50_fpn_1x_dota_le90` | 84.61 | 76.08 | 24.6 | - | - | le90 | [config](../../configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_1x_dota_le90/roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth) |
| cfa | `cfa_r50_fpn_40e_dota_oc` | 84.01 | 73.45 | 24.7 | - | - | oc | [config](../../configs/cfa/cfa_r50_fpn_40e_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/cfa/cfa_r50_fpn_40e_dota_oc/cfa_r50_fpn_40e_dota_oc-2f387232.pth) |
| oriented_rcnn | `oriented_rcnn_r50_fpn_fp16_1x_dota_le90` | 83.52 | 75.63 | 25.6 | - | - | le90 | [config](../../configs/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90/oriented_rcnn_r50_fpn_fp16_1x_dota_le90-57c88621.pth) |
| oriented_rcnn | `oriented_rcnn_r50_fpn_1x_dota_le90` | 83.33 | 75.69 | 25.6 | - | - | le90 | [config](../../configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth) |
| redet | `redet_re50_refpn_1x_dota_le90` | 83.28 | 76.68 | 15.8 | - | - | le90 | [config](../../configs/redet/redet_re50_refpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/redet/redet_re50_fpn_1x_dota_le90/redet_re50_fpn_1x_dota_le90-724ab2da.pth) |
| redet | `redet_re50_refpn_fp16_1x_dota_le90` | 83.26 | 75.99 | 16.1 | - | - | le90 | [config](../../configs/redet/redet_re50_refpn_fp16_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/redet/redet_re50_refpn_fp16_1x_dota_le90/redet_re50_refpn_fp16_1x_dota_le90-1e34da2d.pth) |
| roi_trans | `roi_trans_r50_fpn_1x_dota_ms_le90` | 83.17 | 79.66 | 24.6 | MS | - | le90 | [config](../../configs/roi_trans/roi_trans_r50_fpn_1x_dota_ms_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_1x_dota_ms_rr_le90/roi_trans_r50_fpn_1x_dota_ms_rr_le90-fa99496f.pth) |
| redet | `redet_re50_refpn_1x_dota_ms_rr_le90` | 81.62 | 79.87 | 16.1 | MS | RR | le90 | [config](../../configs/redet/redet_re50_refpn_1x_dota_ms_rr_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/redet/redet_re50_fpn_1x_dota_ms_rr_le90/redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth) |
| ConvNeXt | `rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90` | 81.61 | 74.49 | 26.7 | - | - | le90 | [config](../../configs/convnext/rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/convnext/rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90-388184f6.pth) |
| rotated_faster_rcnn | `rotated_faster_rcnn_r50_fpn_1x_dota_le90` | 81.32 | 73.40 | 26.9 | - | - | le90 | [config](../../configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth) |
| gliding_vertex | `gliding_vertex_r50_fpn_1x_dota_le90` | 81.29 | 73.23 | 26.3 | - | - | le90 | [config](../../configs/gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90/gliding_vertex_r50_fpn_1x_dota_le90-12e7423c.pth) |
| s2anet | `s2anet_r50_fpn_1x_dota_le135` | 81.22 | 73.91 | 23.0 | - | - | le135 | [config](../../configs/s2anet/s2anet_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/s2anet/s2anet_r50_fpn_1x_dota_le135/s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth) |
| s2anet | `s2anet_r50_fpn_fp16_1x_dota_le135` | 80.82 | 74.19 | 22.9 | - | - | le135 | [config](../../configs/s2anet/s2anet_r50_fpn_fp16_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/s2anet/s2anet_r50_fpn_fp16_1x_dota_le135/s2anet_r50_fpn_fp16_1x_dota_le135-5cac515c.pth) |
| rotated_fcos | `rotated_fcos_kld_r50_fpn_1x_dota_le90` | 80.71 | 71.89 | 28.8 | - | - | le90 | [config](../../configs/rotated_fcos/rotated_fcos_kld_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_kld_r50_fpn_1x_dota_le90/rotated_fcos_kld_r50_fpn_1x_dota_le90-ecafdb2b.pth) |
| rotated_fcos | `rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90` | 79.80 | 71.76 | 28.4 | - | - | le90 | [config](../../configs/rotated_fcos/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90-4e044ad2.pth) |
| rotated_fcos | `rotated_fcos_sep_angle_r50_fpn_1x_dota_le90` | 79.06 | 70.70 | 28.9 | - | - | le90 | [config](../../configs/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth) |
| oriented_reppoints | `oriented_reppoints_r50_fpn_1x_dota_le135` | 78.93 | 71.94 | 24.6 | - | - | le135 | [config](../../configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/orientedreppoints/oriented_reppoints_r50_fpn_1x_dota_le135/oriented_reppoints_r50_fpn_1x_dota_le135-ef072de9.pth) |
| kfiou | `r3det_kfiou_ln_r50_fpn_1x_dota_oc` | 78.90 | 72.68 | 19.3 | - | - | oc | [config](../../configs/kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc/r3det_kfiou_ln_r50_fpn_1x_dota_oc-8e7f049d.pth) |
| rotated_fcos | `rotated_fcos_r50_fpn_1x_dota_le90` | 78.74 | 71.28 | 28.8 | - | - | le90 | [config](../../configs/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90/rotated_fcos_r50_fpn_1x_dota_le90-d87568ed.pth) |
| kld | `r3det_tiny_kld_r50_fpn_1x_dota_oc` | 78.67 | 72.76 | 20.9 | - | - | oc | [config](../../configs/kld/r3det_tiny_kld_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/r3det_tiny_kld_r50_fpn_1x_dota_oc/r3det_tiny_kld_r50_fpn_1x_dota_oc-589e142a.pth) |
| kld | `r3det_kld_stable_r50_fpn_1x_dota_oc` | 78.63 | 72.12 | 19.3 | - | - | oc | [config](../../configs/kld/r3det_kld_stable_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/r3det_kld_stable_r50_fpn_1x_dota_oc/r3det_kld_stable_r50_fpn_1x_dota_oc-e011059d.pth) |
| kld | `rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90` | 78.63 | 71.30 | 25.3 | - | - | le90 | [config](../../configs/kld/rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90-474d9955.pth) |
| kld | `r3det_kld_r50_fpn_1x_dota_oc` | 78.62 | 71.83 | 19.2 | - | - | oc | [config](../../configs/kld/r3det_kld_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/r3det_kld_r50_fpn_1x_dota_oc/r3det_kld_r50_fpn_1x_dota_oc-31866226.pth) |
| rotated_atss | `rotated_atss_obb_r50_fpn_1x_dota_le135` | 78.49 | 72.29 | 27.3 | - | - | le135 | [config](../../configs/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le135/rotated_atss_obb_r50_fpn_1x_dota_le135-eab7bc12.pth) |
| rotated_atss | `rotated_atss_obb_r50_fpn_1x_dota_le90` | 77.51 | 70.64 | 27.0 | - | - | le90 | [config](../../configs/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le90/rotated_atss_obb_r50_fpn_1x_dota_le90-e029ca06.pth) |
| cfa | `cfa_r50_fpn_1x_dota_le135` | 76.69 | 69.63 | 24.7 | - | - | le135 | [config](../../configs/cfa/cfa_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/cfa/cfa_r50_fpn_1x_dota_le135/cfa_r50_fpn_1x_dota_le135-aed1cbc6.pth) |
| kld | `rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90` | 76.54 | 70.22 | 24.9 | - | - | le90 | [config](../../configs/kld/rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90-31193e00.pth) |
| oriented_reppoints | `oriented_reppoints_r50_fpn_40e_dota_ms_le135` | 76.04 | 75.21 | 24.6 | MS | - | le135 | [config](../../configs/oriented_reppoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/orientedreppoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135/oriented_reppoints_r50_fpn_40e_dota_ms_le135-bb0323fd.pth) |
| r3det | `r3det_r50_fpn_1x_dota_oc` | 75.80 | 69.80 | 19.3 | - | - | oc | [config](../../configs/r3det/r3det_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_r50_fpn_1x_dota_oc/r3det_r50_fpn_1x_dota_oc-b1fb045c.pth) |
| rotated_retinanet | `rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90` | 75.69 | 76.50 | 25.2 | MS | RR | le90 | [config](../../configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90-1da1ec9c.pth) |
| r3det | `r3det_tiny_r50_fpn_1x_dota_oc` | 75.36 | 70.18 | 22.9 | - | - | oc | [config](../../configs/r3det/r3det_tiny_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_tiny_r50_fpn_1x_dota_oc/r3det_tiny_r50_fpn_1x_dota_oc-c98a616c.pth) |
| rotated_retinanet | `rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90` | 74.94 | 68.79 | 25.0 | - | - | le90 | [config](../../configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90-01de71b5.pth) |
| rotated_retinanet | `rotated_retinanet_obb_r50_fpn_1x_dota_le90` | 74.73 | 68.42 | 24.9 | - | - | le90 | [config](../../configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth) |
| rotated_retinanet | `rotated_retinanet_obb_r50_fpn_1x_dota_le135` | 74.41 | 69.79 | 25.4 | - | - | le135 | [config](../../configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135/rotated_retinanet_obb_r50_fpn_1x_dota_le135-e4131166.pth) |
| csl | `rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90` | 74.37 | 69.51 | 23.6 | - | - | le90 | [config](../../configs/csl/rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/csl/rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90-b4271aed.pth) |
| sasm | `sasm_reppoints_r50_fpn_1x_dota_oc` | 72.80 | 66.45 | 24.2 | - | - | oc | [config](../../configs/sasm_reppoints/sasm_reppoints_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/sasm/sasm_reppoints_r50_fpn_1x_dota_oc/sasm_reppoints_r50_fpn_1x_dota_oc-6d9edded.pth) |
| rotated_reppoints | `rotated_reppoints_r50_fpn_1x_dota_oc` | 66.83 | 59.44 | N/A | - | - | oc | [config](../../configs/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc/rotated_reppoints_r50_fpn_1x_dota_oc-d38ce217.pth) |
| kld | `rotated_retinanet_hbb_kld_stable_r50_fpn_1x_dota_oc` | 53.52 | 69.86 | 24.5 | - | - | oc | [config](../../configs/kld/rotated_retinanet_hbb_kld_stable_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_hbb_kld_stable_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_kld_stable_r50_fpn_1x_dota_oc-92a76443.pth) |
| gwd | `rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc` | 52.43 | 69.55 | 24.4 | - | - | oc | [config](../../configs/gwd/rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/gwd/rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc-41fd7805.pth) |
| rotated_atss | `rotated_atss_hbb_r50_fpn_1x_dota_oc` | 51.80 | 65.59 | 27.9 | - | - | oc | [config](../../configs/rotated_atss/rotated_atss_hbb_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_hbb_r50_fpn_1x_dota_oc/rotated_atss_hbb_r50_fpn_1x_dota_oc-eaa94033.pth) |
| kfiou | `rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90` | 51.06 | 69.60 | 23.8 | - | - | le90 | [config](../../configs/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90-03e02f75.pth) |
| kfiou | `rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc` | 50.83 | 69.76 | 24.6 | - | - | oc | [config](../../configs/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc-c00be030.pth) |
| kld | `rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc` | 50.23 | 69.94 | 24.4 | - | - | oc | [config](../../configs/kld/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc-49c1f937.pth) |
| rotated_retinanet | `rotated_retinanet_hbb_r50_fpn_1x_dota_oc` | 48.45 | 64.55 | 24.2 | - | - | oc | [config](../../configs/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc-e8a7c7df.pth) |
| kfiou | `rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135` | 46.86 | 69.77 | 24.3 | - | - | le135 | [config](../../configs/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135-0eaa4156.pth) |
| g_reppoints | `g_reppoints_r50_fpn_1x_dota_le135` | N/A | 69.49 | N/A | - | - | le135 | [config](../../configs/g_reppoints/g_reppoints_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/g_reppoints/g_reppoints_r50_fpn_1x_dota_le135/g_reppoints_r50_fpn_1x_dota_le135-b840eed7.pth) |

#### Results on DOTA v1.5
| Family | Model Name | mAP (%) | FPS | Scale | Rotation | Angle | Config | Download |
|---|---|---|---|---|---|---|---|---|
| roi_trans | `roi_trans_swin_tiny_fpn_1x_dota_le90` | 78.08 | 23.8 | - | - | le90 | [config](../../configs/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_swin_tiny_fpn_1x_dota_le90/roi_trans_swin_tiny_fpn_1x_dota_le90-ddeee9ae.pth) |
| roi_trans | `roi_trans_r50_fpn_fp16_1x_dota_le90` | 77.40 | 23.9 | - | - | le90 | [config](../../configs/roi_trans/roi_trans_r50_fpn_fp16_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_fp16_1x_dota_le90/roi_trans_r50_fpn_fp16_1x_dota_le90-62eb88b1.pth) |
| redet | `redet_re50_refpn_fp16_1x_dota_le90` | 76.58 | 15.9 | - | - | le90 | [config](../../configs/redet/redet_re50_refpn_fp16_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/redet/redet_re50_refpn_fp16_1x_dota_le90/redet_re50_refpn_fp16_1x_dota_le90-1e34da2d.pth) |
| oriented_rcnn | `oriented_rcnn_r50_fpn_fp16_1x_dota_le90` | 76.46 | 25.4 | - | - | le90 | [config](../../configs/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90/oriented_rcnn_r50_fpn_fp16_1x_dota_le90-57c88621.pth) |
| redet | `redet_re50_refpn_1x_dota_le90` | 76.25 | 15.8 | - | - | le90 | [config](../../configs/redet/redet_re50_refpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/redet/redet_re50_fpn_1x_dota_le90/redet_re50_fpn_1x_dota_le90-724ab2da.pth) |
| cfa | `cfa_r50_fpn_40e_dota_oc` | 76.19 | 23.4 | - | - | oc | [config](../../configs/cfa/cfa_r50_fpn_40e_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/cfa/cfa_r50_fpn_40e_dota_oc/cfa_r50_fpn_40e_dota_oc-2f387232.pth) |
| roi_trans | `roi_trans_r50_fpn_1x_dota_le90` | 75.81 | 24.2 | - | - | le90 | [config](../../configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_1x_dota_le90/roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth) |
| s2anet | `s2anet_r50_fpn_1x_dota_le135` | 75.37 | 22.6 | - | - | le135 | [config](../../configs/s2anet/s2anet_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/s2anet/s2anet_r50_fpn_1x_dota_le135/s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth) |
| gliding_vertex | `gliding_vertex_r50_fpn_1x_dota_le90` | 75.31 | 25.1 | - | - | le90 | [config](../../configs/gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90/gliding_vertex_r50_fpn_1x_dota_le90-12e7423c.pth) |
| roi_trans | `roi_trans_r50_fpn_1x_dota_ms_le90` | 75.22 | 24.6 | MS | - | le90 | [config](../../configs/roi_trans/roi_trans_r50_fpn_1x_dota_ms_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/roi_trans/roi_trans_r50_fpn_1x_dota_ms_rr_le90/roi_trans_r50_fpn_1x_dota_ms_rr_le90-fa99496f.pth) |
| rotated_faster_rcnn | `rotated_faster_rcnn_r50_fpn_1x_dota_le90` | 74.96 | 26.7 | - | - | le90 | [config](../../configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth) |
| oriented_rcnn | `oriented_rcnn_r50_fpn_1x_dota_le90` | 74.94 | 24.8 | - | - | le90 | [config](../../configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth) |
| redet | `redet_re50_refpn_1x_dota_ms_rr_le90` | 74.91 | 15.9 | MS | RR | le90 | [config](../../configs/redet/redet_re50_refpn_1x_dota_ms_rr_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/redet/redet_re50_fpn_1x_dota_ms_rr_le90/redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth) |
| rotated_fcos | `rotated_fcos_kld_r50_fpn_1x_dota_le90` | 74.75 | 23.9 | - | - | le90 | [config](../../configs/rotated_fcos/rotated_fcos_kld_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_kld_r50_fpn_1x_dota_le90/rotated_fcos_kld_r50_fpn_1x_dota_le90-ecafdb2b.pth) |
| ConvNeXt | `rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90` | 74.72 | 27.7 | - | - | le90 | [config](../../configs/convnext/rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/convnext/rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_convnext_adamw_fpn_1x_dota_le90-388184f6.pth) |
| s2anet | `s2anet_r50_fpn_fp16_1x_dota_le135` | 74.53 | 22.4 | - | - | le135 | [config](../../configs/s2anet/s2anet_r50_fpn_fp16_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/s2anet/s2anet_r50_fpn_fp16_1x_dota_le135/s2anet_r50_fpn_fp16_1x_dota_le135-5cac515c.pth) |
| rotated_atss | `rotated_atss_obb_r50_fpn_1x_dota_le135` | 73.56 | 27.0 | - | - | le135 | [config](../../configs/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le135/rotated_atss_obb_r50_fpn_1x_dota_le135-eab7bc12.pth) |
| rotated_fcos | `rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90` | 73.54 | 24.1 | - | - | le90 | [config](../../configs/rotated_fcos/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90-4e044ad2.pth) |
| rotated_fcos | `rotated_fcos_sep_angle_r50_fpn_1x_dota_le90` | 73.19 | 25.3 | - | - | le90 | [config](../../configs/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90/rotated_fcos_sep_angle_r50_fpn_1x_dota_le90-0be71a0c.pth) |
| kfiou | `r3det_kfiou_ln_r50_fpn_1x_dota_oc` | 73.13 | 19.2 | - | - | oc | [config](../../configs/kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc/r3det_kfiou_ln_r50_fpn_1x_dota_oc-8e7f049d.pth) |
| rotated_fcos | `rotated_fcos_r50_fpn_1x_dota_le90` | 73.01 | 24.1 | - | - | le90 | [config](../../configs/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_fcos/rotated_fcos_r50_fpn_1x_dota_le90/rotated_fcos_r50_fpn_1x_dota_le90-d87568ed.pth) |
| kld | `r3det_kld_r50_fpn_1x_dota_oc` | 72.38 | 19.2 | - | - | oc | [config](../../configs/kld/r3det_kld_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/r3det_kld_r50_fpn_1x_dota_oc/r3det_kld_r50_fpn_1x_dota_oc-31866226.pth) |
| kld | `r3det_tiny_kld_r50_fpn_1x_dota_oc` | 72.19 | 20.8 | - | - | oc | [config](../../configs/kld/r3det_tiny_kld_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/r3det_tiny_kld_r50_fpn_1x_dota_oc/r3det_tiny_kld_r50_fpn_1x_dota_oc-589e142a.pth) |
| oriented_reppoints | `oriented_reppoints_r50_fpn_1x_dota_le135` | 72.15 | 24.5 | - | - | le135 | [config](../../configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/orientedreppoints/oriented_reppoints_r50_fpn_1x_dota_le135/oriented_reppoints_r50_fpn_1x_dota_le135-ef072de9.pth) |
| kld | `r3det_kld_stable_r50_fpn_1x_dota_oc` | 71.92 | 19.2 | - | - | oc | [config](../../configs/kld/r3det_kld_stable_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/r3det_kld_stable_r50_fpn_1x_dota_oc/r3det_kld_stable_r50_fpn_1x_dota_oc-e011059d.pth) |
| rotated_atss | `rotated_atss_obb_r50_fpn_1x_dota_le90` | 71.89 | 26.1 | - | - | le90 | [config](../../configs/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_obb_r50_fpn_1x_dota_le90/rotated_atss_obb_r50_fpn_1x_dota_le90-e029ca06.pth) |
| kld | `rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90` | 71.71 | 25.1 | - | - | le90 | [config](../../configs/kld/rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_r50_adamw_fpn_1x_dota_le90-474d9955.pth) |
| cfa | `cfa_r50_fpn_1x_dota_le135` | 71.18 | 24.4 | - | - | le135 | [config](../../configs/cfa/cfa_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/cfa/cfa_r50_fpn_1x_dota_le135/cfa_r50_fpn_1x_dota_le135-aed1cbc6.pth) |
| kld | `rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90` | 70.29 | 24.7 | - | - | le90 | [config](../../configs/kld/rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90/rotated_retinanet_obb_kld_stable_r50_fpn_1x_dota_le90-31193e00.pth) |
| rotated_retinanet | `rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90` | 69.89 | 24.7 | MS | RR | le90 | [config](../../configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90/rotated_retinanet_obb_r50_fpn_1x_dota_ms_rr_le90-1da1ec9c.pth) |
| r3det | `r3det_r50_fpn_1x_dota_oc` | 69.84 | 19.2 | - | - | oc | [config](../../configs/r3det/r3det_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_r50_fpn_1x_dota_oc/r3det_r50_fpn_1x_dota_oc-b1fb045c.pth) |
| rotated_retinanet | `rotated_retinanet_obb_r50_fpn_1x_dota_le135` | 69.45 | 25.2 | - | - | le135 | [config](../../configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135/rotated_retinanet_obb_r50_fpn_1x_dota_le135-e4131166.pth) |
| r3det | `r3det_tiny_r50_fpn_1x_dota_oc` | 69.34 | 22.1 | - | - | oc | [config](../../configs/r3det/r3det_tiny_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/r3det/r3det_tiny_r50_fpn_1x_dota_oc/r3det_tiny_r50_fpn_1x_dota_oc-c98a616c.pth) |
| csl | `rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90` | 68.87 | 23.5 | - | - | le90 | [config](../../configs/csl/rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/csl/rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90-b4271aed.pth) |
| rotated_retinanet | `rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90` | 68.84 | 24.7 | - | - | le90 | [config](../../configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90/rotated_retinanet_obb_r50_fpn_fp16_1x_dota_le90-01de71b5.pth) |
| rotated_retinanet | `rotated_retinanet_obb_r50_fpn_1x_dota_le90` | 68.71 | 24.7 | - | - | le90 | [config](../../configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth) |
| oriented_reppoints | `oriented_reppoints_r50_fpn_40e_dota_ms_le135` | 68.39 | N/A | MS | - | le135 | [config](../../configs/oriented_reppoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/orientedreppoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135/oriented_reppoints_r50_fpn_40e_dota_ms_le135-bb0323fd.pth) |
| sasm | `sasm_reppoints_r50_fpn_1x_dota_oc` | 67.13 | 22.1 | - | - | oc | [config](../../configs/sasm_reppoints/sasm_reppoints_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/sasm/sasm_reppoints_r50_fpn_1x_dota_oc/sasm_reppoints_r50_fpn_1x_dota_oc-6d9edded.pth) |
| rotated_reppoints | `rotated_reppoints_r50_fpn_1x_dota_oc` | 61.34 | 24.0 | - | - | oc | [config](../../configs/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc/rotated_reppoints_r50_fpn_1x_dota_oc-d38ce217.pth) |
| kld | `rotated_retinanet_hbb_kld_stable_r50_fpn_1x_dota_oc` | 49.89 | 24.3 | - | - | oc | [config](../../configs/kld/rotated_retinanet_hbb_kld_stable_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_hbb_kld_stable_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_kld_stable_r50_fpn_1x_dota_oc-92a76443.pth) |
| gwd | `rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc` | 48.74 | 23.8 | - | - | oc | [config](../../configs/gwd/rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/gwd/rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc-41fd7805.pth) |
| rotated_atss | `rotated_atss_hbb_r50_fpn_1x_dota_oc` | 48.60 | 27.1 | - | - | oc | [config](../../configs/rotated_atss/rotated_atss_hbb_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_atss/rotated_atss_hbb_r50_fpn_1x_dota_oc/rotated_atss_hbb_r50_fpn_1x_dota_oc-eaa94033.pth) |
| kfiou | `rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90` | 48.11 | 23.8 | - | - | le90 | [config](../../configs/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le90-03e02f75.pth) |
| kfiou | `rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc` | 47.50 | 24.5 | - | - | oc | [config](../../configs/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_oc-c00be030.pth) |
| kld | `rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc` | 46.72 | 24.1 | - | - | oc | [config](../../configs/kld/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kld/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_kld_r50_fpn_1x_dota_oc-49c1f937.pth) |
| rotated_retinanet | `rotated_retinanet_hbb_r50_fpn_1x_dota_oc` | 44.84 | 24.1 | - | - | oc | [config](../../configs/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc/rotated_retinanet_hbb_r50_fpn_1x_dota_oc-e8a7c7df.pth) |
| kfiou | `rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135` | 44.51 | 24.2 | - | - | le135 | [config](../../configs/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135-0eaa4156.pth) |
| g_reppoints | `g_reppoints_r50_fpn_1x_dota_le135` | N/A | N/A | - | - | le135 | [config](../../configs/g_reppoints/g_reppoints_r50_fpn_1x_dota_le135.py) | [model](https://download.openmmlab.com/mmrotate/v0.1.0/g_reppoints/g_reppoints_r50_fpn_1x_dota_le135/g_reppoints_r50_fpn_1x_dota_le135-b840eed7.pth) |