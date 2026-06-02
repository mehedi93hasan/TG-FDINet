# TG-FDINet: A Temporal-Graph Fusion Network for False Data Injection Detection in the Internet of Medical Things

<p align="center">
  <img src="figures/architecture.png" alt="TG-FDINet Architecture" width="800"/>
</p>

<p align="center">
  <a href="https://doi.org/10.xxxx/xxxxxx">
    <img src="https://img.shields.io/badge/IEEE-Published-blue?style=flat-square&logo=ieee" alt="IEEE Paper"/>
  </a>
  <a href="https://github.com/mehedi93hasan/TG-FDINet/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python" alt="Python"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.1%2B-orange?style=flat-square&logo=pytorch" alt="PyTorch"/>
  </a>
  <img src="https://img.shields.io/github/stars/mehedi93hasan/TG-FDINet?style=flat-square" alt="Stars"/>
 
</p>

---

## Abstract

The Internet of Medical Things (IoMT) exposes physiological monitoring systems to **False Data Injection (FDI)** attacks that directly threaten patient safety. Current deep learning detectors struggle to identify these threats because they lack physiological inductive biases and rely on single-scale temporal encoders that miss both rapid and stealthy attack signatures.

We propose **TG-FDINet**, a Temporal-Graph Fusion Network comprising three novel components:

1. **PCG-GAT** — a Physiological Causal Graph Attention encoder that propagates anomaly signals along fixed cardiovascular dependencies (HR→SpO₂, SpO₂→RR, ABP→HR, …);
2. **MS-TCN** — a dual-dilation Multi-Scale Temporal Convolutional Network that simultaneously captures beat-level (d=1) and trend-level (d=4) attack signatures;
3. **CMFG** — an adaptive Cross-Modal Fusion Gate that learns per-channel branch weighting without explicit supervision.

Evaluated on PhysioNet/CinC 2012, MIMIC-III Waveform, and WESAD, TG-FDINet achieves **+8.3 pp**, **+7.7 pp**, and **+10.2 pp** sensitivity gains over the strongest prior baseline (TSCAN), all significant at *p* < 0.01 under McNemar's test with Holm–Bonferroni correction.

---

## Table of Contents

- [Key Results](#key-results)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Key Results

### Main Comparison (Sensitivity, L1 severity, all FDI morphologies)

| Method | Params | PhysioNet-2012 Sens. | MIMIC-III Sens. | WESAD Sens. |
|:---|---:|:---:|:---:|:---:|
| Isolation Forest | — | 39.9 | 37.6 | 35.1 |
| OC-SVM | — | 37.2 | 36.4 | 34.8 |
| LSTM-AE | 1.2 M | 55.0 | 53.6 | 49.7 |
| TranAD | 3.8 M | 65.6 | 64.2 | 60.8 |
| TSCAN | 2.1 M | 73.3 | 72.5 | 68.2 |
| **TG-FDINet (ours)** | **1.8 M** | **81.6** | **80.2** | **78.4** |
| *Δ vs. TSCAN* | | *+8.3 pp* | *+7.7 pp* | *+10.2 pp* |

All gains are significant at *p* < 0.01 (McNemar's test, Holm–Bonferroni corrected).

### Ablation Study (PhysioNet-2012, mixed-type FDI)

| Configuration | Sens. (%) | F1 (%) |
|:---|:---:|:---:|
| TG-FDINet (Full) | **81.6** | **86.5** |
| No CMFG (fixed avg. fusion) | 79.8 | 84.4 |
| PCG-GAT only (no MS-TCN) | 77.1 | 82.3 |
| MS-TCN only (no PCG-GAT) | 75.8 | 80.4 |
| Single-scale TCN (d₁=1 only) | 75.8 | 80.4 |
| **Learned graph (no causal prior)** | **71.1** | **75.9** |

> The learned-graph ablation (−10.5 pp) confirms that the domain-knowledge physiological prior is **not recoverable from training data alone**.

### Computational Efficiency

| Method | Params | CPU (ms/win.) | GPU (ms/win.) |
|:---|---:|:---:|:---:|
| TranAD | 3.8 M | 2.1 | 0.34 |
| TSCAN | 2.1 M | 0.9 | 0.18 |
| **TG-FDINet** | **1.8 M** | **0.9** | **0.17** |

CPU latency measured per window (*L*=15, *C*=6) averaged over 1,000 inference calls on an Intel Xeon Gold 6248R.

---

## Architecture

```

<p align="center">
  
  <img width="2752" height="1404" alt="fig_1" src="https://github.com/user-attachments/assets/21cbc530-c659-4415-a9e1-085fb8d35b99" />

</p>


```

### Physiological Causal Graph (PCG)

The adjacency matrix **A** encodes ten directed cardiovascular dependencies as a fixed structural prior — providing distribution-shift-robust inductive bias that learned-graph methods cannot replicate:

```
HR ──→ SpO₂     SpO₂ ──→ RR      ABP ──→ HR
HR ──→ RR       SpO₂ ──→ HR      SysBP ──→ DiaBP
DiaBP ──→ HR    HR ──→ Temp      RR ──→ SpO₂
ABP ──→ RR
```

### CMFG Gate Interpretation

The CMFG learns physically meaningful branch allocation without supervision:

| FDI Morphology | Mean Gate ḡ | Dominant Branch |
|:---|:---:|:---|
| Instant (Spike) | **0.29** | Temporal (MS-TCN) |
| Constant (Stuck-at) | 0.51 | Joint |
| Gradual Drift | **0.76** | Graph (PCG-GAT) |
| Bias (Offset) | 0.68 | Graph (PCG-GAT) |

---

## Repository Structure

```
TG-FDINet/
├── models/
│   ├── __init__.py
│   └── tg_fdinet.py          # Full model: PCG-GAT, MS-TCN, CMFG, classifier
├── data/
│   ├── __init__.py
│   ├── datasets.py           # PhysioNet, MIMIC-III, WESAD PyTorch Dataset classes
│   └── fdi_injection.py      # Four FDI morphologies × four severity tiers
├── utils/
│   ├── __init__.py
│   └── metrics.py            # Sensitivity, F1, AUC, McNemar, Holm–Bonferroni
├── experiments/              # Experiment logs and results (auto-generated)
├── figures/                  # Architecture diagrams and result figures
├── configs/
│   ├── physionet.yaml        # Hyperparameters for PhysioNet-2012
│   ├── mimic3.yaml           # Hyperparameters for MIMIC-III Waveform
│   └── wesad.yaml            # Hyperparameters for WESAD
├── train.py                  # Training entry-point
├── evaluate.py               # Evaluation and inference entry-point
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements

- Python 3.9 or later
- PyTorch 2.1 or later
- CUDA 12.1 (optional, for GPU acceleration)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/mehedi93hasan/TG-FDINet.git
cd TG-FDINet

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

TG-FDINet is evaluated on three publicly available datasets. None contains native FDI labels; synthetic attacks are injected following the protocol in Section III-B of the paper.

### PhysioNet/CinC 2012

1. Download from [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/).
2. Extract the archives so that `.txt` patient records are under `set-a/`, `set-b/`, or `set-c/`.

```
physionet2012/
    set-a/
        000001.txt
        000002.txt
        ...
```

3. Update `configs/physionet.yaml` → `data_root: /path/to/physionet2012`.

### MIMIC-III Waveform

1. Request access and download from [PhysioNet MIMIC-III Waveform](https://physionet.org/content/mimic3wdb/1.0/).
   Access requires completion of the CITI "Data or Specimens Only Research" course.
2. Install `wfdb`: `pip install wfdb`.
3. Update `configs/mimic3.yaml` → `data_root: /path/to/mimic3-waveform`.

### WESAD

1. Download from [WESAD on PhysioNet](https://physionet.org/content/wesad/1.0.0/) or the [original UCI release](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29).
2. Ensure the directory contains per-subject folders `S2/` through `S17/`, each with `S{id}.pkl`.

```
WESAD/
    S2/S2.pkl
    S3/S3.pkl
    ...
```

3. Update `configs/wesad.yaml` → `data_root: /path/to/WESAD`.

---

## Quick Start

```python
import torch
from models.tg_fdinet import TGFDINet

# Instantiate model (PhysioNet-2012 channels)
model = TGFDINet(
    channel_names=['HR', 'SpO2', 'SysBP', 'DiaBP', 'RR', 'Temp'],
    seq_len=15,
    d_model=64,
    num_gat_layers=3,
)

print(f"Parameters: {model.count_parameters():,}")   # ~1.8 M

# Inference on a random batch
x = torch.randn(32, 6, 15)     # [Batch, Channels, Window_length]
logit, gate = model(x, return_gate=True)

print(f"Predictions : {logit.shape}")   # [32, 1]
print(f"Gate values : {gate.shape}")    # [32, 6, 64]
print(f"Mean gate   : {gate.mean():.3f}")
```

---

## Training

### Using the training script

```bash
# PhysioNet-2012
python train.py \
    --dataset physionet \
    --data_root /path/to/physionet2012 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --patience 10 \
    --seed 42

# MIMIC-III Waveform
python train.py \
    --dataset mimic3 \
    --data_root /path/to/mimic3-waveform \
    --epochs 100 --batch_size 64 --lr 1e-4 --seed 42

# WESAD
python train.py \
    --dataset wesad \
    --data_root /path/to/WESAD \
    --epochs 100 --batch_size 64 --lr 1e-4 --seed 42
```

Checkpoints are saved to `checkpoints/best_{dataset}.pt`. Training history (loss, F1, AUC per epoch) is logged to `checkpoints/history_{dataset}.json`.

### Key training details (from the paper)

| Hyperparameter | Value |
|:---|:---|
| Optimiser | Adam |
| Learning rate | 1×10⁻⁴ |
| Batch size | 64 |
| Positive-class weight w⁺ | N_neg / N_pos |
| Early stopping patience | 10 epochs (val F1) |
| Weight initialisation | Orthogonal |
| Runs per configuration | 5 (mean ± std reported) |

---

## Evaluation

```bash
python evaluate.py \
    --dataset physionet \
    --data_root /path/to/physionet2012 \
    --checkpoint checkpoints/best_physionet.pt \
    --severity L1 \
    --output results/physionet_test.json
```

Expected output (PhysioNet-2012, L1 severity):

```
============================================================
TG-FDINet  |  PHYSIONET  |  Severity L1
  Sensitivity (Recall) : 81.60%
  Precision            : 92.10%
  F1-Score             : 86.50%
  AUC-ROC              : 0.9600

  Mean CMFG gate (attack windows): g̅ = 0.621
  Expected: ~0.29 (Instant/Spike) → ~0.76 (Gradual Drift)
============================================================
```

---

## Reproducing Paper Results

To reproduce **Table I** of the paper exactly, run each dataset five times with seeds 42–46 and report mean ± std:

```bash
for SEED in 42 43 44 45 46; do
    python train.py --dataset physionet --data_root /path/to/physionet2012 \
                    --seed $SEED --save_dir checkpoints/seed_$SEED
    python evaluate.py --dataset physionet --data_root /path/to/physionet2012 \
                       --checkpoint checkpoints/seed_$SEED/best_physionet.pt \
                       --output results/physionet_seed_$SEED.json
done
```

### FDI Attack Injection (standalone)

```python
import numpy as np
from data.fdi_injection import inject_fdi

rng    = np.random.default_rng(42)
signal = np.random.randn(6, 500)         # [C, T] clean signal
std    = signal.std(axis=1)

# Inject L1 Gradual Drift on channels 0 and 2
x_adv, labels = inject_fdi(
    signal       = signal,
    attack_type  = 'gradual_drift',
    t_start      = 100,
    t_end        = 250,
    channels     = [0, 2],
    channel_std  = std,
    severity     = 'L1',
    rng          = rng,
)

print(f"Attack windows: {labels.sum()} / {len(labels)}")
```

---

## Model Zoo

Pre-trained checkpoints will be made available upon paper acceptance. Links will be updated here.

| Dataset | Sensitivity | F1 | AUC | Download |
|:---|:---:|:---:|:---:|:---:|
| PhysioNet-2012 | 81.6% | 86.5% | .960 | *coming soon* |
| MIMIC-III Waveform | 80.2% | 85.4% | .955 | *coming soon* |
| WESAD | 78.4% | 83.7% | .948 | *coming soon* |

---

## Citation

If you use TG-FDINet in your research, please cite:

```bibtex
@article{hasan2025tgfdinet,
  title     = {{TG-FDINet}: A Temporal-Graph Fusion Network for False Data
               Injection Detection in the Internet of Medical Things},
  author    = {Hasan, Md Mehedi},
  year      = {2025},
  note      = {Code: \url{https://github.com/mehedi93hasan/TG-FDINet}}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

The datasets used (PhysioNet/CinC 2012, MIMIC-III Waveform, WESAD) are subject to their respective access agreements. Please consult the original dataset documentation before use.

---

## Acknowledgements

This work was conducted at the **Connectivity Innovation Network (CIN)**, Charles Sturt University, Albury, NSW, Australia.

The authors acknowledge the PhysioNet platform for providing open access to the physiological benchmarks used in this study, and the authors of TranAD and TSCAN for releasing their source code, which facilitated fair baseline comparisons.

---

<p align="center">
  <i>For questions or issues, please open a GitHub Issue or contact the corresponding author.</i>
</p>
