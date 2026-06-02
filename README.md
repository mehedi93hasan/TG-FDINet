<h1 align="center">TG-FDINet: A Temporal-Graph Fusion Network for<br/>False Data Injection Detection in the Internet of Medical Things</h1>

<p align="center">
  <a href="https://doi.org/10.xxxx/xxxxxx">
    <img src="https://img.shields.io/badge/IEEE-Paper-blue?style=flat-square&logo=ieee" alt="IEEE Paper"/>
  </a>
  <a href="https://github.com/mehedi93hasan/TG-FDINet/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License"/>
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.1+"/>
  </a>
  <img src="https://img.shields.io/github/stars/mehedi93hasan/TG-FDINet?style=flat-square&color=yellow" alt="GitHub Stars"/>
</p>

<p align="center">
  <b>Md Mehedi Hasan</b><br/>
  Connectivity Innovation Network (CIN), Charles Sturt University, Albury, NSW, Australia
</p>

---

## Overview

<p align="center">
  <img width="2752" height="1404" alt="TG-FDINet Architecture" src="https://github.com/user-attachments/assets/fc1ca231-a3de-475d-9d6d-131ea65196a0"/>
</p>

<p align="center"><i>Figure 1. Overall architecture of TG-FDINet. A sliding window W<sub>t</sub> is processed in parallel by the PCG-GAT graph branch (left) and the MS-TCN temporal branch (centre). The Cross-Modal Fusion Gate (CMFG) dynamically weights both branches before a linear classification head produces the binary attack indicator ŷ<sub>t</sub>.</i></p>

The Internet of Medical Things (IoMT) exposes physiological monitoring systems to **False Data Injection (FDI)** attacks that directly threaten patient safety. Existing deep-learning detectors exhibit two structural limitations: (i) they treat inter-sensor spatial correlations implicitly, and (ii) they rely on single-scale temporal encoders that cannot simultaneously resolve both beat-level and trend-level attack signatures.

**TG-FDINet** closes both gaps with three tightly integrated components:

| Component | Role | Key Design Choice |
|:---|:---|:---|
| **PCG-GAT** | Physiological Causal Graph Attention encoder | Fixed adjacency **A** from cardiovascular literature — not learned from data |
| **MS-TCN** | Dual-dilation temporal CNN | *d*=1 (beat-level) ‖ *d*=4 (trend-level) in parallel branches |
| **CMFG** | Cross-Modal Fusion Gate | Per-channel soft gate *g* ∈ (0,1)<sup>*C*×*d*</sup> learned without explicit supervision |

Evaluated across three structurally distinct IoMT benchmarks, TG-FDINet achieves **+8.7 pp mean sensitivity gain** over the strongest prior baseline (TSCAN), with all improvements significant at *p* < 0.01 under McNemar's test with Holm–Bonferroni correction.

---

## Table of Contents

- [Key Results](#key-results)
- [Architecture Details](#architecture-details)
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

### Table I — FDI Detection Metrics at L1 Severity (lowest perturbation magnitude, highest clinical consequence)

| Method | Params | PhysioNet-2012 ||| MIMIC-III Waveform ||| WESAD |||
|:---|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | **Sens.** | **F1** | **AUC** | **Sens.** | **F1** | **AUC** | **Sens.** | **F1** | **AUC** |
| Isolation Forest | — | 39.9 | 47.3 | .682 | 37.6 | 44.7 | .668 | 35.1 | 42.2 | .651 |
| OC-SVM | — | 37.2 | 44.8 | .661 | 36.4 | 43.7 | .654 | 34.8 | 42.1 | .644 |
| LSTM-AE | 1.2 M | 55.0 | 62.4 | .784 | 53.6 | 61.0 | .772 | 49.7 | 57.2 | .751 |
| TranAD | 3.8 M | 65.6 | 72.1 | .876 | 64.2 | 71.0 | .863 | 60.8 | 67.9 | .844 |
| TSCAN | 2.1 M | 73.3 | 80.0 | .924 | 72.5 | 79.0 | .916 | 68.2 | 75.4 | .901 |
| **TG-FDINet** | **1.8 M** | **81.6** | **86.5** | **.960** | **80.2** | **85.4** | **.955** | **78.4** | **83.7** | **.948** |
| *Δ vs. TSCAN* | | *+8.3 pp* | *+6.5 pp* | *+.036* | *+7.7 pp* | *+6.4 pp* | *+.039* | *+10.2 pp* | *+8.3 pp* | *+.047* |

> All Δ values are significant at *p* < 0.01 (McNemar's test, Holm–Bonferroni corrected). Bold denotes best result per column.

### Table II — Ablation Study (PhysioNet-2012, mixed-type FDI)

| Configuration | Sens. (%) | F1 (%) | Params |
|:---|:---:|:---:|:---:|
| **TG-FDINet (Full)** | **81.6** | **86.5** | 1.8 M |
| No CMFG (fixed average fusion) | 79.8 | 84.4 | 1.8 M |
| PCG-GAT only (no MS-TCN) | 77.1 | 82.3 | 1.1 M |
| MS-TCN only (no PCG-GAT) | 75.8 | 80.4 | 0.9 M |
| Single-scale TCN (*d*₁=1 only) | 75.8 | 80.4 | 0.9 M |
| **Learned graph (no causal prior)** | **71.1** | **75.9** | 1.8 M |
| No residual connections | 74.1 | 79.2 | 1.8 M |

> **Critical finding:** replacing the fixed physiological causal adjacency **A** with a GDN-style learned graph reduces sensitivity by **10.5 pp**, directly confirming that the domain-knowledge physiological prior is not recoverable from training data alone.

### Table III — Cross-Dataset Generalisation

| Dataset | Signal Type | Sampling | Δ Sens. vs. TSCAN |
|:---|:---|:---:|:---:|
| PhysioNet-2012 | Tabular ICU vitals | Irregular | +8.3 pp |
| MIMIC-III Waveform | Continuous waveforms | 25 Hz | +7.7 pp |
| WESAD | Multi-modal wearable | 25 Hz | +10.2 pp |
| **Mean gain** | | | **+8.7 pp** |

### Computational Efficiency

| Method | Params | CPU (ms/win.) | GPU (ms/win.) |
|:---|---:|:---:|:---:|
| LSTM-AE | 1.2 M | 1.4 | 0.21 |
| TranAD | 3.8 M | 2.1 | 0.34 |
| TSCAN | 2.1 M | 0.9 | 0.18 |
| **TG-FDINet** | **1.8 M** | **0.9** | **0.17** |

> CPU latency measured per window (*L*=15, *C*=6) averaged over 1,000 inference calls on an Intel Xeon Gold 6248R. TG-FDINet matches TSCAN's latency despite the additional PCG-GAT branch, because the sparse causal graph (*|E|*=10) reduces spatial encoding cost by a factor of *C*²/*|E|* = 3.6× versus full self-attention.

---

## Architecture Details


### Physiological Causal Graph (PCG)

The adjacency matrix **A** ∈ {0,1}<sup>*C*×*C*</sup> encodes ten directed cardiovascular dependencies derived from physiology literature as a **fixed structural prior**. Because **A** is not learned from training data, it remains valid under patient distribution shift — a property that learned-graph methods cannot provide.

```
HR ──→ SpO₂        SpO₂ ──→ RR         ABP  ──→ HR
HR ──→ RR          SpO₂ ──→ HR         SysBP──→ DiaBP
DiaBP──→ HR        HR   ──→ Temp        RR  ──→ SpO₂
ABP  ──→ RR
```

An FDI injection on channel *c*ᵢ immediately produces anomalous attention weights on all outgoing causal edges (*c*ᵢ → *c*ⱼ) ∈ *E*, propagating the attack signature to all causally downstream sensors within *K*=3 hops — a detection mechanism structurally unavailable to learned-graph methods.

### MS-TCN Dual-Dilation Design

| Branch | Dilation | Receptive Field | Targets |
|:---|:---:|:---:|:---|
| Branch 1 | *d*=1 | 3 steps | Instant (Spike), Constant (Stuck-at) — sharp onset |
| Branch 2 | *d*=4 | 9 steps | Gradual Drift, Bias (Offset) — slowly-evolving ramps |

### CMFG Gate Interpretation

The CMFG learns physically meaningful branch allocation **without explicit per-morphology supervision**:

| FDI Morphology | Mean Gate *ḡ* | Dominant Branch | Interpretation |
|:---|:---:|:---|:---|
| Instant (Spike) | **0.29** | Temporal (MS-TCN) | Temporally localised; no multi-hop propagation |
| Constant (Stuck-at) | 0.51 | Joint | Step transition + causal edge violation |
| Bias (Offset) | 0.68 | Graph (PCG-GAT) | Sustained offset violates causal dependencies |
| Gradual Drift | **0.76** | Graph (PCG-GAT) | Ramp propagates through *K*=3 GAT hops |

---

## Repository Structure

```
TG-FDINet/
├── models/
│   ├── __init__.py
│   └── tg_fdinet.py          # PCG-GAT, MS-TCN, CMFG, classifier head
├── data/
│   ├── __init__.py
│   ├── datasets.py           # PhysioNet-2012, MIMIC-III, WESAD Dataset classes
│   └── fdi_injection.py      # Four FDI morphologies × four severity tiers (L1–L4)
├── utils/
│   ├── __init__.py
│   └── metrics.py            # Sensitivity, F1, AUC-ROC, McNemar, Holm–Bonferroni
├── configs/
│   ├── physionet.yaml        # Exact paper hyperparameters — PhysioNet-2012
│   ├── mimic3.yaml           # Exact paper hyperparameters — MIMIC-III Waveform
│   └── wesad.yaml            # Exact paper hyperparameters — WESAD
├── experiments/              # Auto-generated training logs and results
├── figures/                  # Architecture diagrams and result figures
├── train.py                  # Training entry-point (weighted BCE, early stopping)
├── evaluate.py               # Evaluation, CMFG gate analysis, McNemar reporting
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

**Requirements:** Python ≥ 3.9 · PyTorch ≥ 2.1 · CUDA 12.1 *(optional)*

```bash
# 1. Clone the repository
git clone https://github.com/mehedi93hasan/TG-FDINet.git
cd TG-FDINet

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

TG-FDINet is evaluated on three publicly available physiological benchmarks. None contains native FDI labels; synthetic attacks are injected at runtime following the four-morphology protocol in Section III-B of the paper.

### PhysioNet/CinC 2012

1. Download from [PhysioNet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/).
2. Extract so that patient records (`.txt`) reside under `set-a/`, `set-b/`, or `set-c/`:

```
physionet2012/
    set-a/
        000001.txt
        000002.txt
        ...
```

3. Set `data_root` in `configs/physionet.yaml` to your local path.

### MIMIC-III Waveform

1. Request credentialed access at [PhysioNet MIMIC-III Waveform](https://physionet.org/content/mimic3wdb/1.0/).  
   Access requires completion of the CITI *"Data or Specimens Only Research"* course.
2. Install the WFDB toolkit: `pip install wfdb`.
3. Set `data_root` in `configs/mimic3.yaml` to your local path.

### WESAD

1. Download from [PhysioNet WESAD](https://physionet.org/content/wesad/1.0.0/) or the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29).
2. Ensure per-subject directories `S2/` – `S17/` are present, each containing `S{id}.pkl`:

```
WESAD/
    S2/S2.pkl
    S3/S3.pkl
    ...
    S17/S17.pkl
```

3. Set `data_root` in `configs/wesad.yaml` to your local path.

---

## Quick Start

```python
import torch
from models.tg_fdinet import TGFDINet

# Instantiate model with PhysioNet-2012 channel configuration
model = TGFDINet(
    channel_names=['HR', 'SpO2', 'SysBP', 'DiaBP', 'RR', 'Temp'],
    seq_len=15,        # sliding window length L
    d_model=64,        # hidden feature dimension d
    num_gat_layers=3,  # PCG-GAT depth K
)

print(f"Trainable parameters: {model.count_parameters():,}")  # ~1.8 M

# Forward pass on a random batch
x = torch.randn(32, 6, 15)                  # [Batch, Channels, Window_length]
logit, gate = model(x, return_gate=True)

print(f"Attack logits : {logit.shape}")      # [32, 1]
print(f"CMFG gate     : {gate.shape}")       # [32, 6, 64]
print(f"Mean gate ḡ   : {gate.mean():.3f}") # ~0.29 Instant → ~0.76 Gradual Drift
```

---

## Training

```bash
# PhysioNet-2012
python train.py \
    --dataset   physionet \
    --data_root /path/to/physionet2012 \
    --epochs    100 \
    --batch_size 64 \
    --lr        1e-4 \
    --patience  10 \
    --seed      42

# MIMIC-III Waveform
python train.py \
    --dataset   mimic3 \
    --data_root /path/to/mimic3-waveform \
    --epochs 100 --batch_size 64 --lr 1e-4 --seed 42

# WESAD
python train.py \
    --dataset   wesad \
    --data_root /path/to/WESAD \
    --epochs 100 --batch_size 64 --lr 1e-4 --seed 42
```

Checkpoints are saved to `checkpoints/best_{dataset}.pt`. Training history (per-epoch loss, F1, AUC) is written to `checkpoints/history_{dataset}.json`.

### Training Hyperparameters (uniform across all datasets)

| Hyperparameter | Value |
|:---|:---|
| Optimiser | Adam |
| Learning rate | 1×10⁻⁴ |
| Batch size | 64 |
| Positive-class weight *w*⁺ | *N*_neg / *N*_pos |
| Early stopping patience | 10 epochs (validation F1) |
| Weight initialisation | Orthogonal |
| Runs per configuration | 5 (results reported as mean ± std) |
| GPU | NVIDIA A100 80 GB · PyTorch 2.1 · CUDA 12.1 |

---

## Evaluation

```bash
python evaluate.py \
    --dataset    physionet \
    --data_root  /path/to/physionet2012 \
    --checkpoint checkpoints/best_physionet.pt \
    --severity   L1 \
    --output     results/physionet_L1.json
```

Expected console output (PhysioNet-2012, L1 severity):

```
============================================================
TG-FDINet  |  PHYSIONET  |  Severity L1
  Sensitivity (Recall) : 81.60%
  Precision            : 92.10%
  F1-Score             : 86.50%
  AUC-ROC              : 0.9600

  Mean CMFG gate (attack windows): ḡ = 0.621
  Expected: ~0.29 (Instant/Spike) → ~0.76 (Gradual Drift)
============================================================
```

---

## Reproducing Paper Results

To reproduce **Table I** exactly, train five times with seeds 42–46 and report mean ± std:

```bash
for SEED in 42 43 44 45 46; do
    python train.py \
        --dataset physionet --data_root /path/to/physionet2012 \
        --seed $SEED --save_dir checkpoints/seed_$SEED
    python evaluate.py \
        --dataset physionet --data_root /path/to/physionet2012 \
        --checkpoint checkpoints/seed_$SEED/best_physionet.pt \
        --output results/physionet_seed_$SEED.json
done
```

Repeat the loop with `--dataset mimic3` and `--dataset wesad` for the remaining rows.

### Standalone FDI Attack Injection

```python
import numpy as np
from data.fdi_injection import inject_fdi

rng    = np.random.default_rng(42)
signal = np.random.randn(6, 500)      # [C, T] clean physiological signal
std    = signal.std(axis=1)           # per-channel standard deviation

# Inject L1-severity Gradual Drift on channels 0 and 2
x_adv, labels = inject_fdi(
    signal      = signal,
    attack_type = 'gradual_drift',    # 'instant' | 'constant' | 'gradual_drift' | 'bias'
    t_start     = 100,
    t_end       = 250,
    channels    = [0, 2],
    channel_std = std,
    severity    = 'L1',               # 'L1' | 'L2' | 'L3' | 'L4'
    rng         = rng,
)

print(f"Injected attack steps : {labels.sum()} / {len(labels)}")
```

---

## Model Zoo

Pre-trained checkpoints will be released upon paper acceptance. This table will be updated with direct download links.

| Dataset | Sensitivity | Precision | F1 | AUC-ROC | Download |
|:---|:---:|:---:|:---:|:---:|:---:|
| PhysioNet-2012 | 81.6% | 92.1% | 86.5% | .960 | *coming soon* |
| MIMIC-III Waveform | 80.2% | 91.5% | 85.4% | .955 | *coming soon* |
| WESAD | 78.4% | 89.8% | 83.7% | .948 | *coming soon* |

---

## Citation

If TG-FDINet contributes to your research, please cite:

```bibtex
@article{hasan2025tgfdinet,
  title   = {{TG-FDINet}: A Temporal-Graph Fusion Network for False Data
             Injection Detection in the Internet of Medical Things},
  author  = {Hasan, Md Mehedi},
  year    = {2025},
  note    = {Code: \url{https://github.com/mehedi93hasan/TG-FDINet}}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

The datasets used in this work — PhysioNet/CinC 2012, MIMIC-III Waveform, and WESAD — are subject to their respective access agreements and data use policies. Users must independently obtain access and comply with all applicable terms before use.

---

## Acknowledgements

This work was conducted at the **Connectivity Innovation Network (CIN)**, Charles Sturt University, Albury, NSW, Australia.

The authors gratefully acknowledge the PhysioNet platform for providing open access to the physiological benchmarks used in this study, and the authors of TranAD and TSCAN for releasing their source code, which enabled rigorous and fair baseline comparisons under identical experimental conditions.

---

<p align="center">
  For questions, please open a <a href="https://github.com/mehedi93hasan/TG-FDINet/issues">GitHub Issue</a>.
</p>
