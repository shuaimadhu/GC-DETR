# GC-DETR: Geometry-Conditioned Real-Time Object Detector for UAV Open-Water Scenarios

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Overview
Detecting objects in open-water environments is crucial for improving unmanned aerial vehicles (UAVs) visual perception. However, most existing object detection methods are designed for conventional aerial imagery and struggle to address the unique challenges in UAV-based open-water images, including:
- Drastic scale variations of objects
- Complex and cluttered backgrounds
- Lack of consistency between classification confidence and localization quality

To address these challenges, we propose **GC-DETR** — a geometry-conditioned real-time object detector specifically designed for UAV-based open-water scenarios.

## 🔑 Core Innovations
### 1. Sensor-Aware Query Module (SAQM)
- Enhances pose-related priors in the initialized queries with **negligible computational overhead**
- Establishes an explicit relationship between UAV pose and image features

### 2. Spectral Decomposition-based Decoupled Deformable Attention (SDDA)
- Reconstructs the sampling range using object shape information
- Employs spectral decomposition-based geometric decoupling to diversify multi-head attention
- Enables more accurate feature alignment and stronger representation in complex backgrounds

### 3. IoU Prediction Quality Head (IOUPQH)
- Reorders classification confidence scores according to IoU-based localization quality
- Ensures consistency between classification logic and localization accuracy

## 📊 Experimental Results
### Open-Water Dataset (SeaDronesSee)
| Metric                | Improvement (vs. SOTA) |
|-----------------------|------------------------|
| AP<sub>50:95</sub>    | +1.85%                 |
| AP<sub>75</sub>       | +3.53%                 |
| AP<sub>Small</sub>    | +3.57%                 |
| Model Parameters      | -16.3% (reduction)     |


## Datasets

The experiments in this study were conducted on three publicly available aerial datasets.

### SeaDronesSee
Dataset access:  
https://seadronessee.cs.uni-tuebingen.de/

### AFO
Dataset access:  
https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects

### AU-AIR
Dataset access:  
https://www.modelscope.cn/datasets/OmniData/AU-AIR

The original datasets are not redistributed in this repository. Please download the datasets from the corresponding access links above.

The dataset split information, preprocessing settings, configuration files, training and inference settings, evaluation scripts, and other non-sensitive reproducibility materials used in our experiments will be provided in this repository.
