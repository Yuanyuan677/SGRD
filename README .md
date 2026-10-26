# SGRD: Semantically Grounded Reverse Distillation for Unsupervised Anomaly Detection

Official implementation of **Semantically Grounded Reverse Distillation (SGRD)**.
SGRD integrates a frozen vision foundation model (DINOv2) as a fixed semantic reference into Reverse Distillation through two modules:
the **foundation-aware semantic bottleneck (FASB)** and the **foundation-semantic representation anchoring (FSRA)**.

## 1. Environment

- Linux, Python 3.9
- PyTorch 2.0.0 (CUDA 11.8), torchvision 0.15.1
- A single NVIDIA RTX 4090 GPU was used for all experiments

```bash
conda create -n sgrd python=3.9
conda activate sgrd
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 2. Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| MVTec AD | 15 industrial categories (10 object, 5 texture) with pixel-level annotations | https://www.mvtec.com/research-teaching/datasets/mvtec-ad |
| VisA | 10,821 high-resolution images from 12 categories with diverse structural and surface defects | https://github.com/amazon-science/spot-diff , https://registry.opendata.aws/visa |
| MPDD | 6 categories of metal parts captured with varying pose, distance and illumination | https://github.com/stepanjezek/MPDD |
| BTAD | 3 real-world industrial products with body and surface defects | http://avires.dimi.uniud.it/papers/btad/btad.zip |

Unzip the datasets and pass the dataset root through `--data_root`. Each category is organized as
`<category>/train/good`, `<category>/test/<defect_type>`, `<category>/ground_truth/<defect_type>`
(BTAD uses `ok` instead of `good`). For VisA, keep the original release structure together with `split_csv/1cls.csv`:

```
visa
 |-- candle
 |   |-- Data/Images/{Normal,Anomaly}
 |   |-- Data/Masks/Anomaly
 |-- ...
 |-- split_csv/1cls.csv
```

## 3. Pretrained Weights

- Teacher: ImageNet-pretrained WideResNet50 (downloaded automatically by torchvision).
- Foundation model: official DINOv2 ViT-B/14 (downloaded automatically through `torch.hub` from https://github.com/facebookresearch/dinov2 and kept frozen).

## 4. Train and Test

Training and evaluation are both implemented in `main.py`. A separate model is trained for each category.

```bash
python main.py
```

By default this trains and evaluates SGRD on all 15 MVTec AD categories. The default setting is
input size 256x256, batch size 16, 200 epochs, AdamW (learning rate and weight decay 1e-4), seed 42.
The paper reports MVTec AD results averaged over seeds 42, 222 and 1234 (`--seed`).

Other datasets or categories:

```bash
python main.py --dataset visa --data_root /path/to/visa
python main.py --dataset mpdd --data_root /path/to/mpdd
python main.py --dataset btad --data_root /path/to/btad
python main.py --class carpet wood --seed 222
```

Ablation switches: `--no_fasb` (RD + FSRA), `--no_fsra` (RD + FASB), or both (RD baseline).

## Acknowledgement

The backbone, decoder and data pipeline are adapted from the official implementation of
Reverse Distillation (Deng and Li, CVPR 2022).
