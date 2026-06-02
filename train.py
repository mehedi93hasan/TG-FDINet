"""
Training Script for TG-FDINet.

Usage:
    python train.py --dataset physionet --data_root /path/to/physionet \
                    --epochs 100 --batch_size 64 --lr 1e-4 --seed 42

Author: Md Mehedi Hasan
GitHub: https://github.com/mehedi93hasan/TG-FDINet
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.tg_fdinet import TGFDINet
from data.datasets import PhysioNetDataset, MIMICIIIDataset, WESADDataset
from utils.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Weighted binary cross-entropy (handles class imbalance)
# ---------------------------------------------------------------------------

def weighted_bce_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    """
    Weighted BCE:
        L = -(1/T) Σ [ w⁺ · y · log(p̂) + (1-y) · log(1-p̂) ]
    where w⁺ = N_neg / N_pos to counteract the 5% anomaly skew.
    """
    weight = torch.ones_like(targets)
    weight[targets == 1] = pos_weight
    return nn.functional.binary_cross_entropy(
        preds.squeeze(), targets, weight=weight
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: TGFDINet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: float,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds, _ = model(x)
        loss = weighted_bce_loss(preds, y, pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: TGFDINet,
    loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds, _ = model(x)
        loss = weighted_bce_loss(preds, y, pos_weight)
        total_loss += loss.item() * x.size(0)
        all_preds.append(preds.squeeze().cpu().numpy())
        all_labels.append(y.cpu().numpy())

    preds_arr  = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    metrics    = compute_metrics(labels_arr, preds_arr, threshold=threshold)
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device : {device}")

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    cfg = DATASET_REGISTRY[args.dataset]
    DatasetCls = cfg['cls']
    channels   = cfg['channels']

    train_ds = DatasetCls(args.data_root, split='train', win_len=args.win_len, seed=args.seed)
    val_ds   = DatasetCls(args.data_root, split='val',   win_len=args.win_len, seed=args.seed)
    test_ds  = DatasetCls(args.data_root, split='test',  win_len=args.win_len, seed=args.seed)

    print(f"[INFO] Dataset : {args.dataset.upper()} | "
          f"Train {len(train_ds):,} | Val {len(val_ds):,} | Test {len(test_ds):,}")

    n_pos = int(train_ds.labels.sum())
    n_neg = len(train_ds.labels) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"[INFO] Class balance — pos: {n_pos:,} | neg: {n_neg:,} | w⁺: {pos_weight:.2f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TGFDINet(
        channel_names  = channels,
        seq_len        = args.win_len,
        d_model        = args.d_model,
        num_gat_layers = args.gat_layers,
        dropout        = args.dropout,
    ).to(device)

    print(f"[INFO] Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )

    # ------------------------------------------------------------------
    # Training loop with early stopping on validation F1
    # ------------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_f1  = -1.0
    patience_ctr = 0
    history      = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight)
        val_metrics = evaluate(model, val_loader, device, pos_weight)

        scheduler.step(val_metrics['f1'])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Loss {train_loss:.4f} | "
            f"Val Sens {val_metrics['sensitivity']:.3f} "
            f"F1 {val_metrics['f1']:.3f} "
            f"AUC {val_metrics['auc']:.3f} | "
            f"{elapsed:.1f}s"
        )

        history.append({'epoch': epoch, 'train_loss': train_loss, **val_metrics})

        # Save best checkpoint
        if val_metrics['f1'] > best_val_f1:
            best_val_f1  = val_metrics['f1']
            patience_ctr = 0
            ckpt_path    = os.path.join(args.save_dir, f'best_{args.dataset}.pt')
            torch.save({
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'val_f1':       best_val_f1,
                'channel_names': channels,
                'args':         vars(args),
            }, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break

    # ------------------------------------------------------------------
    # Final evaluation on test set
    # ------------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    test_metrics = evaluate(model, test_loader, device, pos_weight)

    print("\n" + "=" * 60)
    print(f"Test Results ({args.dataset.upper()}) — Severity L1 avg")
    print(f"  Sensitivity : {test_metrics['sensitivity']:.4f}")
    print(f"  Precision   : {test_metrics['precision']:.4f}")
    print(f"  F1          : {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC     : {test_metrics['auc']:.4f}")
    print("=" * 60)

    # Save history and test metrics
    with open(os.path.join(args.save_dir, f'history_{args.dataset}.json'), 'w') as f:
        json.dump({'history': history, 'test': test_metrics}, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='TG-FDINet Training Script')

    # Data
    p.add_argument('--dataset',   type=str, default='physionet',
                   choices=list(DATASET_REGISTRY),
                   help='Dataset to train on.')
    p.add_argument('--data_root', type=str, required=True,
                   help='Root directory of the dataset.')
    p.add_argument('--win_len',   type=int, default=15,
                   help='Sliding window length L (default: 15).')

    # Model
    p.add_argument('--d_model',    type=int,   default=64,
                   help='Hidden feature dimension d (default: 64).')
    p.add_argument('--gat_layers', type=int,   default=3,
                   help='Number of PCG-GAT layers K (default: 3).')
    p.add_argument('--dropout',    type=float, default=0.1,
                   help='Dropout probability (default: 0.1).')

    # Training
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch_size', type=int,   default=64)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--patience',   type=int,   default=10,
                   help='Early stopping patience (default: 10).')
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--workers',    type=int,   default=4,
                   help='DataLoader worker processes.')

    # I/O
    p.add_argument('--save_dir', type=str, default='checkpoints',
                   help='Directory for saving checkpoints and logs.')

    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
