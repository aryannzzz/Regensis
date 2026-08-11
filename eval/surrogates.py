"""eval/surrogates.py — Track E deliverable. See repo copy for full docstring."""
from __future__ import annotations
import numpy as np


def label_shuffle(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(labels)
    return rng.permutation(labels)


def phase_randomize(signal, rng):
    signal = np.asarray(signal, dtype=float)
    n = signal.shape[-1]
    spectrum = np.fft.rfft(signal, axis=-1)
    random_phase = rng.uniform(0, 2 * np.pi, size=spectrum.shape)
    random_phase[..., 0] = 0.0
    if n % 2 == 0:
        random_phase[..., -1] = 0.0
    randomized = np.abs(spectrum) * np.exp(1j * random_phase)
    return np.fft.irfft(randomized, n=n, axis=-1)


def spectrally_matched_noise(signal, rng):
    return phase_randomize(signal, rng)


def time_reversal(signal):
    signal = np.asarray(signal)
    return np.flip(signal, axis=-1)


def channel_permutation(signal, rng, channel_axis=0):
    signal = np.asarray(signal)
    n_channels = signal.shape[channel_axis]
    perm = rng.permutation(n_channels)
    return np.take(signal, perm, axis=channel_axis)


def validate_surrogate(real_effect_stat, null_stats_on_planted_real_effect,
                        artifact_stat, null_stats_on_planted_artifact, alpha=0.05):
    real_null = np.asarray(null_stats_on_planted_real_effect)
    art_null = np.asarray(null_stats_on_planted_artifact)
    real_threshold = np.quantile(real_null, 1 - alpha)
    real_effect_survives = bool(real_effect_stat > real_threshold)
    art_threshold = np.quantile(art_null, 1 - alpha)
    artifact_is_killed = bool(artifact_stat <= art_threshold)
    return {"real_effect_survives": real_effect_survives,
            "artifact_is_killed": artifact_is_killed,
            "valid": real_effect_survives and artifact_is_killed}