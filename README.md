# GC-DETR: Geometry-Conditioned Real-Time Object Detection

GC-DETR is a real-time object detector specifically designed for UAV-based open-water scenarios. It addresses the unique challenges of drastic scale variations, complex backgrounds, and misalignment between classification confidence and localisation quality.

## Architecture Overview

GC-DETR extends the DETR family of detectors with three novel components:

### 1. Sensor-Aware Query Module (SAQM)
Enhances initialized object queries with UAV pose-related priors (altitude, pitch, roll, yaw) using a lightweight gated projection. This establishes an explicit relationship between UAV sensor state and image features with negligible computational overhead.

### 2. Spectral Decomposition-based Decoupled Deformable Attention (SDDA)
A deformable cross-attention module with two key innovations:
- **Shape-conditioned sampling**: deformable offsets are scaled by predicted object width/height so the receptive field of each head adapts to the object geometry.
- **Spectral decoupling**: attention weights are computed by decomposing query features along orthogonal x/y spectral bases (DCT-style), allowing heads to specialise in different spatial frequency bands and improving feature alignment in complex backgrounds.

### 3. IoU Prediction Quality Head (IOUPQH)
Predicts a per-box IoU quality score alongside classification logits, then reweights classification confidence by the IoU score. This ensures the final ranking score reflects both categorical certainty and localisation accuracy, improving NMS quality.

## Results

On the SeaDronesSee dataset, GC-DETR achieves:
- **+1.85% AP₅₀:₉₅** compared to state-of-the-art detectors
- **+3.57% APₛₘₐₗₗ** for small object detection
- **−16.3%** reduction in parameter count

On a terrestrial aerial traffic dataset:
- **+2.7% AP₅₀:₉₅**

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
import torch
from gcdetr import build_gc_detr

# Build a GC-DETR model
model = build_gc_detr(
    num_classes=80,           # number of object categories
    backbone_name="resnet50", # backbone architecture
    pretrained_backbone=True, # use ImageNet pre-trained weights
    embed_dim=256,            # transformer embedding dimension
    num_queries=300,          # number of object queries
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    num_points=4,             # SDDA sampling points per head per level
    num_levels=3,             # number of feature-map scales
    aux_loss=True,
)

# Forward pass with optional UAV pose information
images = torch.randn(2, 3, 640, 640)
pose   = torch.tensor([[0.3, -0.1, 0.05, 1.2],   # [altitude, pitch, roll, yaw] (normalised)
                        [0.5,  0.0, 0.02, 0.8]])

outputs = model(images, pose=pose)

# outputs keys:
#   "pred_boxes"   – (B, N, 4) boxes in (cx, cy, w, h) normalised format
#   "pred_logits"  – (B, N, num_classes) raw classification logits
#   "pred_iou"     – (B, N, 1) IoU quality logits (apply sigmoid for [0,1] score)
#   "pred_scores"  – (B, N, num_classes) joint classification × IoU scores for NMS
#   "aux_outputs"  – list of per-layer dicts (when aux_loss=True)
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
gcdetr/
├── models/
│   ├── gc_detr.py       – Main model and build_gc_detr() factory
│   ├── saqm.py          – Sensor-Aware Query Module
│   ├── sdda.py          – Spectral Decomposition Deformable Attention
│   ├── ioupqh.py        – IoU Prediction Quality Head
│   ├── backbone.py      – ResNet feature extractor
│   ├── neck.py          – Channel-projection neck with positional encoding
│   └── transformer.py   – Encoder and decoder stacks
└── utils/
    ├── box_ops.py        – Bounding-box format conversions and IoU utilities
    └── misc.py           – NestedTensor and padding helpers
tests/
├── test_saqm.py
├── test_sdda.py
├── test_ioupqh.py
├── test_gc_detr.py
└── test_box_ops.py
```
