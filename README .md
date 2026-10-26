# SGRD: Semantically Grounded Reverse Distillation for Unsupervised Anomaly Detection

This repository accompanies the paper **Semantically Grounded Reverse Distillation (SGRD)**.
SGRD integrates a frozen vision foundation model (DINOv2) as a fixed semantic reference into Reverse Distillation through two modules:
the **foundation-aware semantic bottleneck (FASB)** and the **foundation-semantic representation anchoring (FSRA)**.

> **Code release:** The complete source code (FASB, FSRA, DINOv2 feature extraction, and the training and evaluation scripts) will be released in this repository upon acceptance of the paper.

## 1. Environment

- Linux, Python 3.9.23
- PyTorch 2.0.0+cu118, torchvision 0.15.1+cu118

```bash
conda create -n sgrd python=3.9.23
conda activate sgrd
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.2 scipy==1.13.1 scikit-learn==1.6.1 scikit-image==0.24.0 \
            opencv-python==4.11.0.86 pandas==2.3.1 pillow==9.5.0 matplotlib==3.7.1 tqdm==4.67.1
```

## 2. Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| MVTec AD | 15 industrial categories (10 object, 5 texture) with pixel-level annotations | https://www.mvtec.com/research-teaching/datasets/mvtec-ad |
| VisA | 10,821 high-resolution images from 12 categories with diverse structural and surface defects | https://github.com/amazon-science/spot-diff , https://registry.opendata.aws/visa |
| MPDD | 6 categories of metal parts captured with varying pose, distance and illumination | https://github.com/stepanje/MPDD |
| BTAD | 3 real-world industrial products with body and surface defects | http://avires.dimi.uniud.it/papers/btad/btad.zip |

## 3. Pretrained Weights

- Teacher: ImageNet-pretrained WideResNet50 (torchvision).
- Foundation model: official DINOv2 ViT-B/14 (https://github.com/facebookresearch/dinov2), kept frozen.

## Acknowledgement

The backbone, decoder and data pipeline will be adapted from the official implementation of
Reverse Distillation (Deng and Li, CVPR 2022).
