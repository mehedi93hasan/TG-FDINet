"""
FDI Attack Injection Utilities for IoMT Physiological Data.

Implements the four canonical FDI morphologies from Khan et al. (2025)
across four severity tiers (L1–L4), consistent with the TG-FDINet paper.

Attack morphologies:
    - Instant  (Spike):      abrupt single-step perturbation.
    - Constant (Stuck-at):   sensor reading fixed to a constant value.
    - Gradual Drift:         linearly increasing offset over a ramp window.
    - Bias (Offset):         constant additive offset for the attack window.

Severity tiers follow the magnitude scaling defined in the paper
(L1 = lowest perturbation magnitude / highest clinical consequence).

Author: Md Mehedi Hasan
GitHub: https://github.com/mehedi93hasan/TG-FDINet
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Severity magnitude multipliers (fraction of channel std)
# L1 → most subtle / clinically most dangerous
# ---------------------------------------------------------------------------
SEVERITY_SCALES = {
    'L1': 0.5,
    'L2': 1.0,
    'L3': 2.0,
    'L4': 4.0,
}


def inject_instant(
    signal: np.ndarray,
    t_attack: int,
    channels: list,
    channel_std: np.ndarray,
    severity: str = 'L1',
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject an Instant (Spike) FDI attack on selected channels.

    A single-step perturbation ±scale·σ_c is applied at t_attack.
    The attack label is 1 only at t_attack.

    Args:
        signal:      original signal array [C, T].
        t_attack:    time index of spike injection.
        channels:    list of channel indices to corrupt.
        channel_std: per-channel standard deviation [C].
        severity:    one of 'L1'–'L4'.
        rng:         numpy random generator (for sign randomisation).

    Returns:
        x_adv:  perturbed signal [C, T].
        labels: binary attack flag [T].
    """
    if rng is None:
        rng = np.random.default_rng()

    scale  = SEVERITY_SCALES[severity]
    x_adv  = signal.copy()
    labels = np.zeros(signal.shape[1], dtype=np.int32)

    for c in channels:
        sign = rng.choice([-1, 1])
        x_adv[c, t_attack] += sign * scale * channel_std[c]

    labels[t_attack] = 1
    return x_adv, labels


def inject_constant(
    signal: np.ndarray,
    t_start: int,
    t_end: int,
    channels: list,
    channel_std: np.ndarray,
    severity: str = 'L1',
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject a Constant (Stuck-at) FDI attack.

    The signal on selected channels is frozen to the value at t_start
    for the interval [t_start, t_end).

    Args:
        signal:      original signal array [C, T].
        t_start:     start of stuck-at window (inclusive).
        t_end:       end of stuck-at window (exclusive).
        channels:    channel indices to corrupt.
        channel_std: per-channel standard deviation [C] (unused; kept
                     for API consistency).
        severity:    severity tier (unused; kept for API consistency).
        rng:         numpy random generator (unused here).

    Returns:
        x_adv:  perturbed signal [C, T].
        labels: binary attack flag [T].
    """
    x_adv  = signal.copy()
    labels = np.zeros(signal.shape[1], dtype=np.int32)

    for c in channels:
        stuck_value = signal[c, t_start]
        x_adv[c, t_start:t_end] = stuck_value

    labels[t_start:t_end] = 1
    return x_adv, labels


def inject_gradual_drift(
    signal: np.ndarray,
    t_start: int,
    t_end: int,
    channels: list,
    channel_std: np.ndarray,
    severity: str = 'L1',
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject a Gradual Drift FDI attack (linearly increasing offset).

    The offset ramps from 0 to ±scale·σ_c over [t_start, t_end).
    This morphology is the hardest to detect with single-scale encoders
    and provides the largest PCG-GAT advantage.

    Args:
        signal:      original signal array [C, T].
        t_start:     start of ramp window (inclusive).
        t_end:       end of ramp window (exclusive).
        channels:    channel indices to corrupt.
        channel_std: per-channel standard deviation [C].
        severity:    one of 'L1'–'L4'.
        rng:         numpy random generator (for direction randomisation).

    Returns:
        x_adv:  perturbed signal [C, T].
        labels: binary attack flag [T].
    """
    if rng is None:
        rng = np.random.default_rng()

    scale  = SEVERITY_SCALES[severity]
    x_adv  = signal.copy()
    labels = np.zeros(signal.shape[1], dtype=np.int32)

    ramp_len = t_end - t_start
    ramp     = np.linspace(0.0, 1.0, ramp_len)

    for c in channels:
        direction = rng.choice([-1, 1])
        x_adv[c, t_start:t_end] += direction * scale * channel_std[c] * ramp

    labels[t_start:t_end] = 1
    return x_adv, labels


def inject_bias(
    signal: np.ndarray,
    t_start: int,
    t_end: int,
    channels: list,
    channel_std: np.ndarray,
    severity: str = 'L1',
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject a Bias (Offset) FDI attack (constant additive shift).

    A fixed offset ±scale·σ_c is added to selected channels over
    [t_start, t_end).

    Args:
        signal:      original signal array [C, T].
        t_start:     start of bias window (inclusive).
        t_end:       end of bias window (exclusive).
        channels:    channel indices to corrupt.
        channel_std: per-channel standard deviation [C].
        severity:    one of 'L1'–'L4'.
        rng:         numpy random generator.

    Returns:
        x_adv:  perturbed signal [C, T].
        labels: binary attack flag [T].
    """
    if rng is None:
        rng = np.random.default_rng()

    scale  = SEVERITY_SCALES[severity]
    x_adv  = signal.copy()
    labels = np.zeros(signal.shape[1], dtype=np.int32)

    for c in channels:
        direction = rng.choice([-1, 1])
        x_adv[c, t_start:t_end] += direction * scale * channel_std[c]

    labels[t_start:t_end] = 1
    return x_adv, labels


# ---------------------------------------------------------------------------
# Unified injection dispatcher
# ---------------------------------------------------------------------------

ATTACK_FN = {
    'instant':       inject_instant,
    'constant':      inject_constant,
    'gradual_drift': inject_gradual_drift,
    'bias':          inject_bias,
}


def inject_fdi(
    signal: np.ndarray,
    attack_type: str,
    t_start: int,
    t_end: int,
    channels: list,
    channel_std: np.ndarray,
    severity: str = 'L1',
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified FDI injection interface.

    Args:
        signal:      original signal array [C, T].
        attack_type: one of 'instant', 'constant', 'gradual_drift', 'bias'.
        t_start:     attack start index.
        t_end:       attack end index (exclusive; ignored for 'instant').
        channels:    list of channel indices to corrupt.
        channel_std: per-channel standard deviation array [C].
        severity:    severity tier 'L1'–'L4'.
        rng:         numpy random generator.

    Returns:
        x_adv:  perturbed signal [C, T].
        labels: binary attack labels [T].

    Raises:
        ValueError: if attack_type is not recognised.
    """
    if attack_type not in ATTACK_FN:
        raise ValueError(
            f"Unknown attack_type '{attack_type}'. "
            f"Choose from: {list(ATTACK_FN)}"
        )

    kwargs = dict(
        signal=signal,
        t_start=t_start if attack_type != 'instant' else t_start,
        channels=channels,
        channel_std=channel_std,
        severity=severity,
        rng=rng,
    )

    if attack_type == 'instant':
        return inject_instant(
            signal=signal,
            t_attack=t_start,
            channels=channels,
            channel_std=channel_std,
            severity=severity,
            rng=rng,
        )

    return ATTACK_FN[attack_type](
        signal=signal,
        t_start=t_start,
        t_end=t_end,
        channels=channels,
        channel_std=channel_std,
        severity=severity,
        rng=rng,
    )
