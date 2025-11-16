# MLLM-For3D: Adapting Multimodal Large Language Model for 3D Reasoning Segmentation (NeurIPS 2025)

![Overview](./assets/mllmfor3d_framework.png)

MLLM-For3D is a novel framework that **transfers multimodal reasoning ability from 2D large language models (LISA/LISA++) to 3D scenes**, enabling **label-free 3D reasoning segmentation**.  
It performs fine-grained segmentation of 3D objects based on **complex natural language queries**, bridging the gap between 2D reasoning segmentation and 3D spatial understanding.

[[Paper (NeurIPS 2025)]](https://arxiv.org/abs/2503.18135)  

MLLM-For3D unifies the training of **multi-view semantic distillation** and **3D reasoning segmentation**, using frozen 2D LISA models as teachers to generate pseudo masks for 3D points and enforcing **semantic and spatial consistency** during multi-view fusion.

---

## 🔍 Highlights
- **3D Reasoning Segmentation (3D-RS):** Interpret implicit human instructions in 3D scenes.
- **2D→3D Knowledge Transfer:** Distill reasoning masks from 2D LISA/LISA++ across multi-view renderings.
- **Multi-View Consistency:** Aligns pixel-point-text correspondences across views.
- **Label-Free Training:** No manual 3D annotations required.
- **Unified Evaluation Protocol:** Evaluated on the *ScanNet++ ReasonSeg3D* benchmark and *Reason3D* dataset.

---

## 📦 Installation

### Step 1. Install PyTorch
```bash
conda create -n mllmfor3d python=3.10 -y
conda activate mllmfor3d
conda install pytorch==2.0.1 torchvision==0.15.2 cudatoolkit=11.8 -c pytorch
```

### Step 2. Install Dependencies
```bash
pip install pytorch_lightning==2.1.2
pip install torchmetrics==0.11.4
pip install open3d==0.18.0
pip install transformers==4.36.2
pip install sentencepiece ftfy regex tqdm scikit-image
```

### Step 3. Install MinkowskiEngine and Torchsparse (Optional for sparse 3D backbone)
```bash
# MinkowskiEngine
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
pip install ninja
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

---

## 🗂 Data Preparation

### Step 1. Download Datasets
MLLM-For3D is trained and evaluated on **ScanNet++** and **Reason3D** datasets:
```bash
# ScanNet++
wget http://www.scan-net.org/download
# Reason3D Dataset (Huang et al., 2024)
git clone https://github.com/kuanchihhuang/Reason3D.git
```

### Step 2. Convert Data to 3D-RS Format
```bash
python tools/preprocess_scannetpp.py --data_root /path/to/scannetpp
python tools/convert_reason3d.py --input /path/to/reason3d --output /path/to/reason3d_3drs
```

### Step 3. Generate CLIP Text Embeddings
```bash
python utils/prompt_engineering.py --model ViT16 --class-set scannetpp_top100
# Output: scannetpp_top100_ViT16_clip_text.pth
```

---

## 🧠 Pre-training

MLLM-For3D consists of three submodules:
- **`model_images`**: Frozen 2D LISA/LISA++ reasoning model  
- **`model_points`**: Sparse 3D U-Net for point cloud encoding  
- **`model_fusion`**: Transformer fusion module for 2D–3D semantic alignment  

Run the multi-view pretraining:
```bash
python train/pretrain_multiview.py --cfg config/mllmfor3d_pretrain.yaml
# Models will be saved to output/mllmfor3d/{date}/model.pt
```

---

## 🧩 Annotation-Free Reasoning Segmentation

Evaluate the model in zero-shot label-free mode:
```bash
python eval/eval_zeroshot.py   --cfg config/mllmfor3d_eval.yaml   --pretrained output/mllmfor3d/{date}/model.pt
```

---

## 🎯 Fine-tuning on Reason3D or ReasonSeg3D

```bash
python train/finetune.py   --cfg config/mllmfor3d_reason3d.yaml   --pretrained output/mllmfor3d/{date}/model.pt
```

Results will be stored in:
```
output/finetune/reason3d/{date}/metrics.json
```

---

## 🧪 Evaluation Metrics

| Metric | Description |
|---------|--------------|
| **mIoU** | Mean Intersection over Union |
| **Acc@0.25 / Acc@0.50** | Mask accuracy under IoU thresholds |
| **cIoU / gIoU** | Cumulative and Generalized IoU (ReasonSeg3D) |

---

## 🧩 Supported Tasks

| Task | Description |
|------|--------------|
| **3D Reasoning Segmentation** | Segment object implied by a complex query |
| **3D Hierarchical Searching** | Locate and segment object within a region |
| **3D Referring Segmentation** | Segment object given explicit language reference |
| **3D QA with Mask Output** | Answer questions and highlight related region |

---

## 🧰 Code Structure

```
MLLM-For3D/
├── config/                 # YAML configs for training/eval
├── train/                  # Training scripts
├── eval/                   # Evaluation and visualization
├── models/                 # Core model definitions (LISA + 3D backbone)
├── datasets/               # ScanNet++, Reason3D dataset loaders
├── utils/                  # Tools for prompt, metrics, visualization
├── assets/                 # Figures and result visualizations
└── README.md
```

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{huang2025mllmfor3d,
  title={MLLM-For3D: Adapting Multimodal Large Language Model for 3D Reasoning Segmentation},
  author={Huang, Jiaxin and et al.},
  booktitle={NeurIPS},
  year={2025}
}
```

### Related Works
- **CLIP2Scene:** Chen et al., CVPR 2023  
- **LISA:** Lai et al., CVPR 2024 

---

## 🙏 Acknowledgements
Part of this repository builds upon:
- [CLIP2Scene (CVPR 2023)](https://github.com/cv3d/CLIP2Scene)
- [LISA (CUHK, 2024)](https://github.com/dvlab-research/LISA)

---

## 📧 Contact

For questions or collaborations, please contact  
**Jiaxin Huang** (hjx@mbzuai.ac.ae)