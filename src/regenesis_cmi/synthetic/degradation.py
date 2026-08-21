"""Information-destroying EMG and matched EEG-myogenic degradation controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .schema import SyntheticDataset


@dataclass(frozen=True)
class DegradationResult:
    alpha: float
    emg: NDArray[np.float64]
    eeg: NDArray[np.float64]
    realized_snr_db: float
    matched_myogenic: bool


def realized_snr_db(
    signal: NDArray[np.float64], noise: NDArray[np.float64]
) -> float:
    signal_power = float(np.mean(np.square(signal)))
    noise_power = float(np.mean(np.square(noise)))
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return float("-inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def degrade(
    dataset: SyntheticDataset, alpha: float, *, matched_myogenic: bool = False
) -> DegradationResult:
    """Apply M_alpha=alpha*S_M+N_M; optionally attenuate only EEG-side S_C."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    emg = alpha * dataset.emg_signal + dataset.emg_noise
    if matched_myogenic and dataset.scenario != "redundant":
        eeg = (
            dataset.cortical
            + alpha * dataset.artifact_signal_eeg
            + dataset.artifact_noise_eeg
            + dataset.eeg_noise
        )
    else:
        eeg = dataset.eeg.copy()
    return DegradationResult(
        alpha=alpha,
        emg=emg,
        eeg=eeg,
        realized_snr_db=realized_snr_db(alpha * dataset.emg_signal, dataset.emg_noise),
        matched_myogenic=matched_myogenic,
    )


def make_myogenic_proxy(
    dataset: SyntheticDataset, quality: float, *, seed: int
) -> NDArray[np.float64]:
    """Create C-hat_r=r*C_standardized+sqrt(1-r^2)*epsilon."""

    if not 0.0 <= quality <= 1.0:
        raise ValueError("quality must lie in [0, 1]")
    standardized = np.empty_like(dataset.cranial)
    keys = dataset.metadata["oracle_key"].to_numpy()
    for key in dataset.oracle_model.keys:
        positions = np.flatnonzero(keys == key)
        mean, std = dataset.oracle_model._cranial_standardization(key)
        standardized[positions] = (dataset.cranial[positions] - mean) / std
    noise = np.random.default_rng(seed).standard_normal(standardized.shape)
    return quality * standardized + np.sqrt(1.0 - quality**2) * noise

