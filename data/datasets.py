"""
IoMT Dataset Loaders for TG-FDINet.

Provides PyTorch Dataset classes for the three evaluation benchmarks:
    - PhysioNetDataset  : PhysioNet/CinC 2012 (tabular ICU vitals)
    - MIMICIIIDataset   : MIMIC-III Waveform (continuous 25 Hz waveforms)
    - WESADDataset      : WESAD (multi-modal wearable signals)

All datasets:
    - Apply 70 : 15 : 15 subject-level splits (train / val / test).
    - Inject four FDI morphologies × four severity tiers.
    - Return sliding windows W_t ∈ R^{C×L} with binary label y_t.

Author: Md Mehedi Hasan
GitHub: https://github.com/mehedi93hasan/TG-FDINet
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

from data.fdi_injection import inject_fdi, ATTACK_FN, SEVERITY_SCALES


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def sliding_windows(
    signal: np.ndarray,     # [C, T]
    labels: np.ndarray,     # [T]
    win_len: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract overlapping sliding windows from a signal.

    For each time-step t, the window W_t = signal[:, t-L+1 : t+1] and
    the label y_t = max(labels[t-L+1 : t+1]) following the paper's
    aggregate-adversarial-presence formulation.

    Args:
        signal:  sensor array [C, T].
        labels:  per-step attack indicator [T].
        win_len: window length L.

    Returns:
        windows: array [N_windows, C, L].
        y:       binary labels [N_windows].
    """
    C, T = signal.shape
    if T < win_len:
        return np.empty((0, C, win_len), dtype=np.float32), np.empty(0, dtype=np.int64)

    windows, ys = [], []
    for t in range(win_len - 1, T):
        w  = signal[:, t - win_len + 1 : t + 1]   # [C, L]
        yt = int(labels[t - win_len + 1 : t + 1].max())
        windows.append(w)
        ys.append(yt)

    return np.array(windows, dtype=np.float32), np.array(ys, dtype=np.int64)


def add_fdi_attacks(
    signal: np.ndarray,         # [C, T]
    channel_std: np.ndarray,    # [C]
    attack_ratio: float = 0.05,
    win_len: int = 15,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly inject FDI attacks across all four morphologies and four
    severity tiers such that approximately `attack_ratio` windows are
    labelled as attacks.

    Args:
        signal:       clean signal [C, T].
        channel_std:  per-channel std used to scale attack magnitude.
        attack_ratio: target fraction of attack windows (default 0.05).
        win_len:      sliding window length L.
        rng:          numpy random generator.

    Returns:
        x_adv:  perturbed signal [C, T].
        labels: binary per-step attack labels [T].
    """
    if rng is None:
        rng = np.random.default_rng()

    C, T      = signal.shape
    x_adv     = signal.copy()
    labels    = np.zeros(T, dtype=np.int32)

    attack_types = list(ATTACK_FN.keys())
    severities   = list(SEVERITY_SCALES.keys())

    n_attacks = max(1, int(T * attack_ratio / win_len))

    for _ in range(n_attacks):
        atype    = rng.choice(attack_types)
        severity = rng.choice(severities)
        n_ch     = rng.integers(1, max(2, C // 2))
        channels = rng.choice(C, size=n_ch, replace=False).tolist()
        dur      = rng.integers(win_len, min(3 * win_len, T // 4))
        t_start  = rng.integers(0, T - dur)
        t_end    = t_start + dur

        x_adv, labels = inject_fdi(
            signal       = x_adv,
            attack_type  = atype,
            t_start      = t_start,
            t_end        = t_end,
            channels     = channels,
            channel_std  = channel_std,
            severity     = severity,
            rng          = rng,
        )

    return x_adv, labels


# ---------------------------------------------------------------------------
# PhysioNet/CinC 2012
# ---------------------------------------------------------------------------

class PhysioNetDataset(Dataset):
    """
    PhysioNet/CinC 2012 ICU telemetry dataset.

    Expected directory layout::

        data_root/
            set-a/  (or set-b/, set-c/)
                000001.txt
                000002.txt
                ...

    Each .txt file is a two-column time-series: 'Time,Parameter,Value'
    format (PhysioNet challenge standard). We extract six streams:
    HR, SpO2, SysBP, DiaBP, RR, Temp; records with >30% missing values
    are discarded.

    Args:
        data_root:   path to PhysioNet-2012 root directory.
        split:       'train', 'val', or 'test' (70:15:15 subject split).
        win_len:     sliding window length L (default 15).
        attack_ratio: fraction of windows to label as attacks.
        seed:        random seed for reproducibility.
    """

    CHANNELS     = ['HR', 'SpO2', 'SysBP', 'DiaBP', 'RR', 'Temp']
    CHANNEL_RANGES = {            # physiological plausibility bounds
        'HR':    (20,  300),
        'SpO2':  (50,  100),
        'SysBP': (50,  250),
        'DiaBP': (20,  150),
        'RR':    (4,   60),
        'Temp':  (25,  45),
    }
    MAX_MISSING = 0.30

    def __init__(
        self,
        data_root: str,
        split: str       = 'train',
        win_len: int     = 15,
        attack_ratio: float = 0.05,
        seed: int        = 42,
    ) -> None:
        super().__init__()
        self.win_len      = win_len
        self.attack_ratio = attack_ratio
        self.rng          = np.random.default_rng(seed)

        all_files = sorted(glob.glob(os.path.join(data_root, '**', '*.txt'), recursive=True))
        all_files = self._split_files(all_files, split)

        self.windows, self.labels = self._build(all_files)

    # ------------------------------------------------------------------
    def _split_files(self, files: list, split: str) -> list:
        n = len(files)
        idx = np.arange(n)
        np.random.RandomState(0).shuffle(idx)
        cuts = [int(0.70 * n), int(0.85 * n)]
        if split == 'train':
            return [files[i] for i in idx[:cuts[0]]]
        elif split == 'val':
            return [files[i] for i in idx[cuts[0]:cuts[1]]]
        else:
            return [files[i] for i in idx[cuts[1]:]]

    def _load_record(self, path: str) -> Optional[np.ndarray]:
        """Parse PhysioNet challenge text file into [C, T] float array."""
        import csv
        data = {c: [] for c in self.CHANNELS}
        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    param = row.get('Parameter', '').strip()
                    if param in data:
                        try:
                            val = float(row['Value'])
                            lo, hi = self.CHANNEL_RANGES[param]
                            if lo <= val <= hi:
                                data[param].append(val)
                        except (ValueError, KeyError):
                            pass
        except FileNotFoundError:
            return None

        min_len = min(len(data[c]) for c in self.CHANNELS)
        if min_len == 0:
            return None

        arr = np.stack([np.array(data[c][:min_len]) for c in self.CHANNELS])  # [C, T]

        # Check missing ratio (use NaN count from original; simplified here)
        missing_frac = (arr == 0).mean()
        if missing_frac > self.MAX_MISSING:
            return None

        # Z-score normalisation per channel
        mu  = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True) + 1e-8
        return (arr - mu) / std, std.squeeze()

    def _build(
        self, files: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_w, all_y = [], []
        for f in files:
            result = self._load_record(f)
            if result is None:
                continue
            signal, ch_std = result
            x_adv, labels  = add_fdi_attacks(
                signal, ch_std, self.attack_ratio, self.win_len, self.rng
            )
            ws, ys = sliding_windows(x_adv, labels, self.win_len)
            all_w.append(ws)
            all_y.append(ys)

        if not all_w:
            return np.empty((0, len(self.CHANNELS), self.win_len), dtype=np.float32), \
                   np.empty(0, dtype=np.int64)

        return np.concatenate(all_w), np.concatenate(all_y)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx])    # [C, L]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
# MIMIC-III Waveform
# ---------------------------------------------------------------------------

class MIMICIIIDataset(Dataset):
    """
    MIMIC-III Waveform dataset (continuous physiological signals at 25 Hz).

    Expected layout::

        data_root/
            p00/p000020/
                p000020-2183-04-28-17-47.hea
                p000020-2183-04-28-17-47n.mat
                ...

    Reads WFDB-format records via the `wfdb` library. Extracts six
    channels: ECG, ABP, PPG, HR, RR, Temp; downsampled to 25 Hz.

    Args:
        data_root:    path to MIMIC-III waveform root.
        split:        'train', 'val', or 'test'.
        win_len:      window length L (default 15).
        target_fs:    target sampling frequency in Hz (default 25).
        attack_ratio: fraction of attack windows.
        seed:         random seed.

    Note:
        Requires the `wfdb` package: `pip install wfdb`.
    """

    CHANNELS    = ['ECG', 'ABP', 'PPG', 'HR', 'RR', 'Temp']
    TARGET_FS   = 25

    def __init__(
        self,
        data_root: str,
        split: str           = 'train',
        win_len: int         = 15,
        attack_ratio: float  = 0.05,
        seed: int            = 42,
    ) -> None:
        super().__init__()
        self.data_root    = data_root
        self.win_len      = win_len
        self.attack_ratio = attack_ratio
        self.rng          = np.random.default_rng(seed)

        records = self._discover_records()
        records = self._split(records, split)
        self.windows, self.labels = self._build(records)

    def _discover_records(self) -> list:
        return sorted(glob.glob(
            os.path.join(self.data_root, '**', '*.hea'), recursive=True
        ))

    def _split(self, records: list, split: str) -> list:
        n = len(records)
        idx = np.arange(n)
        np.random.RandomState(0).shuffle(idx)
        cuts = [int(0.70 * n), int(0.85 * n)]
        s = {'train': idx[:cuts[0]], 'val': idx[cuts[0]:cuts[1]], 'test': idx[cuts[1]:]}
        return [records[i] for i in s[split]]

    def _load_record(self, header_path: str):
        try:
            import wfdb
        except ImportError:
            raise ImportError("Please install wfdb: pip install wfdb")

        rec_name = header_path[:-4]
        try:
            record = wfdb.rdrecord(rec_name)
        except Exception:
            return None

        sig_names = [s.upper() for s in record.sig_name]
        fs        = record.fs
        signals   = record.p_signal  # [T, C_all]

        # Downsample to TARGET_FS
        step = max(1, int(fs / self.TARGET_FS))
        signals = signals[::step]

        arrays = []
        for ch in self.CHANNELS:
            # Fuzzy match: ECG matches 'II', 'V', etc.
            match_idx = next(
                (i for i, s in enumerate(sig_names) if ch in s), None
            )
            if match_idx is not None:
                arrays.append(signals[:, match_idx])
            else:
                arrays.append(np.zeros(signals.shape[0]))

        arr = np.stack(arrays)   # [C, T]

        # Replace NaN/inf with 0
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        mu  = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True) + 1e-8
        return (arr - mu) / std, std.squeeze()

    def _build(self, records: list):
        all_w, all_y = [], []
        for r in records:
            result = self._load_record(r)
            if result is None:
                continue
            signal, ch_std = result
            x_adv, labels  = add_fdi_attacks(
                signal, ch_std, self.attack_ratio, self.win_len, self.rng
            )
            ws, ys = sliding_windows(x_adv, labels, self.win_len)
            all_w.append(ws)
            all_y.append(ys)

        if not all_w:
            return np.empty((0, len(self.CHANNELS), self.win_len), dtype=np.float32), \
                   np.empty(0, dtype=np.int64)
        return np.concatenate(all_w), np.concatenate(all_y)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
# WESAD
# ---------------------------------------------------------------------------

class WESADDataset(Dataset):
    """
    WESAD (Wearable Stress and Affect Detection) dataset.

    Expected layout::

        data_root/
            S2/S2.pkl
            S3/S3.pkl
            ...
            S17/S17.pkl

    Extracts six channels from the combined Empatica E4 / RespiBAN
    pickle files: ECG, EDA, BVP, RESP, ST, ACC (magnitude).
    Signals are downsampled or upsampled to a uniform 25 Hz.

    Args:
        data_root:    path to WESAD root directory.
        split:        'train', 'val', or 'test' (subject-level split).
        win_len:      window length L (default 15).
        attack_ratio: fraction of attack windows.
        seed:         random seed.
    """

    SUBJECTS    = list(range(2, 18))   # S2–S17 (S1 excluded in standard protocol)
    TARGET_FS   = 25
    CHANNELS    = ['ECG', 'EDA', 'BVP', 'RESP', 'ST', 'ACC']

    def __init__(
        self,
        data_root: str,
        split: str           = 'train',
        win_len: int         = 15,
        attack_ratio: float  = 0.05,
        seed: int            = 42,
    ) -> None:
        super().__init__()
        self.data_root    = data_root
        self.win_len      = win_len
        self.attack_ratio = attack_ratio
        self.rng          = np.random.default_rng(seed)

        subjects = self._split_subjects(split)
        self.windows, self.labels = self._build(subjects)

    def _split_subjects(self, split: str) -> list:
        n    = len(self.SUBJECTS)
        subj = list(self.SUBJECTS)
        np.random.RandomState(0).shuffle(subj)
        cuts = [int(0.70 * n), int(0.85 * n)]
        parts = {'train': subj[:cuts[0]], 'val': subj[cuts[0]:cuts[1]], 'test': subj[cuts[1]:]}
        return parts[split]

    def _load_subject(self, sid: int):
        import pickle
        path = os.path.join(self.data_root, f'S{sid}', f'S{sid}.pkl')
        if not os.path.isfile(path):
            return None

        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        # Chest (RespiBAN) at 700 Hz
        chest  = data.get('signal', {}).get('chest', {})
        # Wrist (Empatica E4) at various rates
        wrist  = data.get('signal', {}).get('wrist', {})

        def resample(arr: np.ndarray, src_fs: int) -> np.ndarray:
            factor = src_fs / self.TARGET_FS
            out_len = int(len(arr) / factor)
            return arr[::max(1, int(factor))][:out_len]

        ecg  = resample(chest.get('ECG', np.zeros(1)).squeeze(), 700)
        resp = resample(chest.get('Resp', np.zeros(1)).squeeze(), 700)
        eda  = resample(wrist.get('EDA', np.zeros(1)).squeeze(), 4)
        bvp  = resample(wrist.get('BVP', np.zeros(1)).squeeze(), 64)
        st   = resample(wrist.get('TEMP', np.zeros(1)).squeeze(), 4)
        acc  = resample(wrist.get('ACC', np.zeros((1, 3))), 32)
        if acc.ndim > 1:
            acc = np.linalg.norm(acc, axis=1)

        min_len = min(len(ecg), len(eda), len(bvp), len(resp), len(st), len(acc))
        arrays = np.stack([
            ecg[:min_len], eda[:min_len], bvp[:min_len],
            resp[:min_len], st[:min_len], acc[:min_len],
        ])  # [6, T]

        arrays = np.nan_to_num(arrays, nan=0.0)
        mu  = arrays.mean(axis=1, keepdims=True)
        std = arrays.std(axis=1, keepdims=True) + 1e-8
        return (arrays - mu) / std, std.squeeze()

    def _build(self, subjects: list):
        all_w, all_y = [], []
        for sid in subjects:
            result = self._load_subject(sid)
            if result is None:
                continue
            signal, ch_std = result
            x_adv, labels  = add_fdi_attacks(
                signal, ch_std, self.attack_ratio, self.win_len, self.rng
            )
            ws, ys = sliding_windows(x_adv, labels, self.win_len)
            all_w.append(ws)
            all_y.append(ys)

        if not all_w:
            return np.empty((0, len(self.CHANNELS), self.win_len), dtype=np.float32), \
                   np.empty(0, dtype=np.int64)
        return np.concatenate(all_w), np.concatenate(all_y)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.windows[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
