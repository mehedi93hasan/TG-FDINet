TG-FDINet: Temporal-Graph Fusion Network for IoMT FDI DetectionThis repository contains the official PyTorch implementation of TG-FDINet, as presented in our paper: TG-FDINet: A Temporal-Graph Fusion Network for False Data Injection Detection in the Internet of Medical Things.📖 OverviewThe Internet of Medical Things (IoMT) exposes physiological monitoring systems to False Data Injection (FDI) attacks, directly threatening patient safety. Current deep learning detectors lack physiological inductive biases and rely on single-scale temporal encoders that miss both rapid and stealthy attack signatures.TG-FDINet addresses these limitations by introducing:PCG-GAT (Physiological Causal Graph Attention Encoder): Propagates anomaly signals along fixed, domain-knowledge cardiovascular dependencies.MS-TCN (Multi-Scale Temporal Convolutional Network): A dual-dilation network that simultaneously captures beat-level (spike) and trend-level (ramp) anomalies.CMFG (Cross-Modal Fusion Gate): Dynamically weights spatial and temporal representations without explicit supervision.Evaluated on PhysioNet-2012, MIMIC-III, and WESAD, TG-FDINet achieves state-of-the-art sensitivity against Instant, Constant, Gradual Drift, and Bias FDI attacks.🏗️ Architecture(Note: Upload your Figure 1 to your repository's root or assets folder and update this image link)⚙️ Installation & RequirementsClone the repository:Bashgit clone https://github.com/mehedi93hasan/TG-FDINet.git
cd TG-FDINet
Install the required dependencies:Bashpip install -r requirements.txt
Core Dependencies:Python 3.8+PyTorch 2.1+CUDA 12.1 (Recommended for GPU acceleration)NumPy, Pandas, Scikit-learn🚀 UsageThe core model is modular and readily available in models.py.Defining the Adjacency MatrixCrucial: TG-FDINet requires a fixed physiological causal graph adjacency matrix ($A \in \{0,1\}^{C \times C}$) derived from cardiovascular physiology. You must pass this binary matrix during the forward pass.Pythonimport torch
from models import TGFDINet

# 1. Initialize the model (C=6 channels, L=15 window length)
model = TGFDINet(num_channels=6, window_length=15, d=64, gat_layers=3)

# 2. Define your fixed physiological adjacency matrix (6x6 for 6 channels)
# Note: Replace this placeholder with your specific directed edge matrix
adj_matrix = torch.tensor([
    [0, 1, 0, 0, 0, 0], # HR -> SpO2
    [0, 0, 0, 1, 0, 0], # SpO2 -> RR
    # ... complete your 6x6 matrix based on physiology
    [0, 0, 0, 0, 0, 0]
], dtype=torch.float32)

# 3. Dummy input (Batch Size: 64, Channels: 6, Window: 15)
dummy_input = torch.randn(64, 6, 15)

# 4. Forward pass
attack_probabilities = model(dummy_input, adj_matrix)
print("Output shape:", attack_probabilities.shape) # Expected: (64, 1)
📊 DatasetsThe model was evaluated on three structurally distinct datasets using a controlled synthetic FDI injection protocol:PhysioNet/CinC 2012 (Tabular ICU vitals)MIMIC-III Waveform (Continuous 25 Hz waveforms)WESAD (Multi-modal wearable sensors)(Please refer to the respective official dataset repositories to download the raw physiological data).📝 CitationIf you find this code or our conceptual framework useful in your research, please consider citing our paper:Code snippet@article{hasan2026tgfdinet,
  title={TG-FDINet: A Temporal-Graph Fusion Network for False Data Injection Detection in the Internet of Medical Things},
  author={Hasan, Md Mehedi and [Co-authors]},
  journal={IEEE/Elsevier [Update upon acceptance]},
  year={2026}
}
✉️ ContactFor any questions regarding the code, experimental setup, or the FDI injection protocol, please open an issue in this repository or contact Md Mehedi Hasan.
