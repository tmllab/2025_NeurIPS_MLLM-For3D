# MLLM-For3D: Teaching 2D Vision-Language Models to Understand 3D Scenes — Without Any 3D Labels

*How we transferred multimodal reasoning from 2D to 3D and achieved state-of-the-art 3D reasoning segmentation at NeurIPS 2025*

---

Imagine walking into a room and asking an AI: **"It's hot, where can I get a cold drink?"** — and the system not only understands your intent but also locates and segments the refrigerator in a full 3D point cloud of the room. That's the problem we tackle in our NeurIPS 2025 paper, **MLLM-For3D**.

The key twist? Our model learns to do this **without ever seeing a single 3D annotation**.

> **Paper**: [arXiv:2503.18135](https://arxiv.org/abs/2503.18135)
> **Code**: [github.com/tmllab/2025_NeurIPS_MLLM-For3D](https://github.com/tmllab/2025_NeurIPS_MLLM-For3D)
> **Project Page**: [tmllab.github.io/2025_NeurIPS_MLLM-For3D](https://tmllab.github.io/2025_NeurIPS_MLLM-For3D/)

---

## The Problem: 3D Reasoning Segmentation

3D scene understanding has come a long way, but most existing methods require **expensive 3D annotations** — manually labeling point clouds with object masks and semantic categories. This is tedious, costly, and doesn't scale well.

Meanwhile, 2D vision-language models like [LISA](https://github.com/dvlab-research/LISA) have shown remarkable ability to interpret complex human queries and produce segmentation masks in images. The natural question is: **can we transfer this 2D reasoning ability to 3D?**

The naive approach — running a 2D model on each view independently and fusing the results — fails spectacularly. Why? Two main reasons:

1. **Hallucinations**: Without 3D context, the 2D model may "see" objects in some views that don't actually exist in the scene.
2. **Inconsistency**: The same object might be segmented differently (or not at all) across different viewpoints, leading to noisy and fragmented 3D masks.

---

## Our Solution: MLLM-For3D

We designed a four-stage pipeline that carefully addresses both challenges:

### Stage 1: Multi-view Pseudo-label Generation

We render multiple views of a 3D scene and feed each view, along with the text query, through a **frozen** LISA model (LISA-7B-v1). This gives us per-view segmentation masks and, critically, the internal `[SEG]` token embeddings that encode the model's "understanding" of what it's segmenting.

### Stage 2: Token Attention Filtering

Not all views are reliable. Some may contain hallucinated segmentation masks — the 2D model might segment a "refrigerator" in a view of a bedroom wall. We use the **attention scores** of the `[SEG]` token to quantify confidence for each view and filter out unreliable ones, keeping only the top-*r* most confident views.

### Stage 3: 3D Unprojection & Spatial Fusion

The reliable 2D masks are **unprojected into 3D space** using known camera parameters. For any 3D point visible in multiple views, an attention-based fusion module weighs each view's contribution based on both its reliability score and semantic similarity to a unified representation — enforcing **spatial consistency** in the 3D mask.

### Stage 4: Token-for-Query Alignment

Here's where it gets interesting. We derive a **unified query embedding** from the `[SEG]` tokens across all reliable views. This embedding becomes the "query" that the 3D model uses to identify target objects, ensuring **consistent identification of the same object** regardless of viewpoint. This is our Token-for-Query mechanism — it bridges the semantic gap between what the 2D model "thinks" and what the 3D model needs to "learn."

---

## Results: +55% mIoU Improvement

The results exceeded our expectations. On the ReasonSeg3D benchmark:

| Method | 3D Labels Needed | cIoU | gIoU |
|---|---|---|---|
| CLIP2Scene | No | 5.7 | 11.2 |
| OpenScene | No | 6.8 | 12.5 |
| Reason3D | **Yes** | 19.0 | 22.3 |
| **MLLM-For3D (Ours)** | **No** | **36.2** | **38.5** |

A few things stand out:

- MLLM-For3D **doesn't use any 3D annotations** yet outperforms Reason3D, which requires manually labeled 3D data.
- The improvement over prior label-free methods (CLIP2Scene, OpenScene) is massive — roughly **+55% in mIoU**.
- On the Visual Grounding without Object Names (VG-w/o-ON) benchmark — where queries describe spatial relationships without naming objects (e.g., "the thing to the left of the bed") — our method outperforms ScanRefer by nearly **30% absolute** in Acc@0.25.

---

## What Can It Do?

MLLM-For3D supports multiple 3D understanding tasks through natural language:

- **Reasoning Segmentation**: "Where can I wash my hands?" → segments the sink
- **Hierarchical Search**: "The book on the second shelf near the window" → finds the specific book
- **Referring Segmentation**: "The red chair next to the table" → segments that particular chair
- **QA with Mask**: "Where can I sit?" → highlights all chairs in the 3D scene

All from a single unified framework, and all without 3D annotations.

---

## Why Does This Matter?

This work has implications for robotics, AR/VR, and embodied AI. If a robot needs to understand the instruction "bring me the mug from the kitchen counter," it needs to:

1. **Understand intent** (what the user wants)
2. **Reason about spatial relationships** (kitchen counter, not the coffee table)
3. **Segment the target** in 3D space (for grasping or navigation)

MLLM-For3D can do all three — and because it doesn't require 3D annotations, it can potentially generalize to new environments without costly relabeling.

---

## A Companion Benchmark: Surprise3D

We also released **Surprise3D** (also at NeurIPS 2025), a large-scale dataset specifically designed to test spatial reasoning in 3D. It contains **200k+ vision-language pairs** across **900+ indoor scenes**, with **89k+ human-annotated spatial queries** that deliberately avoid naming objects — forcing models to truly reason about spatial relationships rather than rely on keyword shortcuts.

> Check it out: [arxiv.org/abs/2507.07781](https://arxiv.org/abs/2507.07781) | [GitHub](https://github.com/liziwennba/SURPRISE3D) | [HuggingFace](https://huggingface.co/datasets/hhllzz/surprise-3d)

---

## Try It Out

We've open-sourced the full codebase including pre-training, fine-tuning, and evaluation scripts. The code is built on PyTorch Lightning with MinkowskiEngine for sparse 3D processing and integrates with HuggingFace for the LISA model:

```bash
git clone https://github.com/tmllab/2025_NeurIPS_MLLM-For3D.git
cd 2025_NeurIPS_MLLM-For3D
pip install -r requirements_plus.txt
```

See the [README](https://github.com/tmllab/2025_NeurIPS_MLLM-For3D) for detailed setup instructions.

---

## Citation

If you find this work useful, please consider citing:

```
@inproceedings{huang2025mllmfor3d,
  title={MLLM-For3D: Adapting Multimodal Large Language Model for 3D Reasoning Segmentation},
  author={Huang, Jiaxin and Chen, Runnan and Li, Ziwen and Gao, Zhengqing and He, Xiao and Guo, Yandong and Liu, Tongliang and Gong, Mingming},
  booktitle={NeurIPS},
  year={2025}
}
```

---

*Jiaxin Huang is a researcher at MBZUAI working on 3D scene understanding and multimodal AI. Follow for more updates on 3D vision-language research.*

**Tags**: `#MachineLearning` `#ComputerVision` `#NeurIPS` `#3DSceneUnderstanding` `#MultimodalAI` `#DeepLearning` `#PointCloud`
