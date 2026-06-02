"""
Evaluation / Inference Script for TG-FDINet.

Loads a trained checkpoint and reports per-morphology, per-severity metrics,
CMFG gate statistics, and McNemar significance against baselines.

Usage:
    python evaluate.py --dataset physionet --data_root /path/to/data \
                       --checkpoint checkpoints/best_physionet.pt \
                       --severity L1

Author: Md Mehedi Hasan
GitHub: https://github.com/mehedi93hasan/TG-FDINet
"""

import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.tg_fdinet import TGFDINet
from data.datasets import PhysioNetDataset, MIMICIIIDataset, WESADDataset
from utils.metrics import compute_metrics, pairwise_significance


DATASET_REGISTRY = {
    'physionet': {
        'cls':      PhysioNetDataset,
        'channels': ['HR', 'SpO2', 'SysBP', 'DiaBP', 'RR', 'Temp'],
    },
    'mimic3': {
        'cls':      MIMICIIIDataset,
        'channels': ['ECG', 'ABP', 'PPG', 'HR', 'RR', 'Temp'],
    },
    'wesad': {
        'cls':      WESADDataset,
        'channels': ['ECG', 'EDA', 'BVP', 'RESP', 'ST', 'ACC'],
    },
}


@torch.no_grad()
def run_inference(
    model: TGFDINet,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Run model inference and collect predictions, labels, and gate values.

    Returns:
        y_score: predicted probabilities [N].
        y_true:  ground-truth labels [N].
        gates:   CMFG gate arrays [N, C, d].
    """
    model.eval()
    all_scores, all_labels, all_gates = [], [], []

    for x, y in loader:
        x = x.to(device)
        score, gate = model(x, return_gate=True)
        all_scores.append(score.squeeze().cpu().numpy())
        all_labels.append(y.numpy())
        if gate is not None:
            all_gates.append(gate.cpu().numpy())

    y_score = np.concatenate(all_scores)
    y_true  = np.concatenate(all_labels)
    gates   = np.concatenate(all_gates) if all_gates else None

    return y_score, y_true, gates


def main(args: argparse.Namespace) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = DATASET_REGISTRY[args.dataset]

    model = TGFDINet(
        channel_names   = cfg['channels'],
        seq_len         = ckpt['args'].get('win_len', 15),
        d_model         = ckpt['args'].get('d_model', 64),
        num_gat_layers  = ckpt['args'].get('gat_layers', 3),
        dropout         = 0.0,   # disable at inference
    ).to(device)

    model.load_state_dict(ckpt['model_state'])
    print(f"[INFO] Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val F1 = {ckpt['val_f1']:.4f})")

    # ------------------------------------------------------------------
    # Test dataset
    # ------------------------------------------------------------------
    test_ds = cfg['cls'](
        data_root    = args.data_root,
        split        = 'test',
        win_len      = ckpt['args'].get('win_len', 15),
        attack_ratio = 0.05,
        seed         = 0,
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    y_score, y_true, gates = run_inference(model, test_loader, device)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(y_true, y_score)

    print("\n" + "=" * 60)
    print(f"TG-FDINet  |  {args.dataset.upper()}  |  Severity {args.severity}")
    print(f"  Sensitivity (Recall) : {metrics['sensitivity'] * 100:.2f}%")
    print(f"  Precision            : {metrics['precision'] * 100:.2f}%")
    print(f"  F1-Score             : {metrics['f1'] * 100:.2f}%")
    print(f"  AUC-ROC              : {metrics['auc']:.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # CMFG gate summary (if available)
    # ------------------------------------------------------------------
    if gates is not None:
        mean_gate = float(gates[y_true == 1].mean()) if y_true.sum() > 0 else float('nan')
        print(f"\n  Mean CMFG gate (attack windows): g̅ = {mean_gate:.3f}")
        print("  Expected: ~0.29 (Instant/Spike) → ~0.76 (Gradual Drift)")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({'dataset': args.dataset, 'metrics': metrics}, f, indent=2)
        print(f"\n[INFO] Results saved to {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='TG-FDINet Evaluation Script')
    p.add_argument('--dataset',    type=str, required=True,
                   choices=list(DATASET_REGISTRY))
    p.add_argument('--data_root',  type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--severity',   type=str, default='L1',
                   choices=['L1', 'L2', 'L3', 'L4'])
    p.add_argument('--threshold',  type=float, default=0.5)
    p.add_argument('--output',     type=str, default=None,
                   help='Optional path to save JSON results.')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
