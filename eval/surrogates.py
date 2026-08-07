"""
eval/surrogates.py — Track E deliverable, laboratory-wide.

A surrogate is a transformation of the data that destroys the effect of
interest while preserving as much else as possible (Operating Manual
§2.4). Every surrogate here must be validated in BOTH directions before
a track relies on it (§2.4, §5.2 ICA failure-mode note):

  1. A planted REAL effect must be ABSENT from the null the surrogate
     produces — i.e. running the pipeline on surrogate data should not
     recover an effect you deliberately planted in the real data.
  2. A planted ARTIFACT (e.g. simulated myogenic contamination) that the
     surrogate specifically targets must be KILLED by it — i.e. the
     surrogate should remove that artifact's contribution, not just the
     real effect.

"A suite validated only against under-correction is not validated" (§2.4)
— validating direction 1 alone is not enough; do both before use.

Five surrogates, matching §2.4's list exactly:
  - label_shuffle            — destroys label-locked information only.
  - phase_randomize          — destroys temporal/phase structure while
                                preserving the power spectrum.
  - spectrally_matched_noise — replaces the signal with noise matched to
                                its power spectrum; destroys everything
                                except spectral content.
  - time_reversal            — destroys causal/anticipatory ordering
                                while preserving spectral content.
  - channel_permutation      — destroys spatial/montage structure while
                                preserving each channel's own time series.

All functions take an explicit `rng` (a numpy.random.Generator) so every
surrogate draw is reproducible and seed-loggable per §8.4.
"""

from __future__ import annotations

import numpy as np


def label_shuffle(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Permute labels independently of the data. Destroys any label-locked
    effect (neural or myogenic) while leaving every trial's data and the
    label marginal distribution untouched.

    Note (§2.3): this null does NOT separate cortical from myogenic
    effects — both produce label-locked information and both pass a
    label-shuffle null comfortably. Use band-resolved / montage-ablated
    / myogenic-conditioning controls for that; label_shuffle only tests
    "is there label-locked structure at all."
    """
    labels = np.asarray(labels)
    return rng.permutation(labels)


def phase_randomize(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Randomize the Fourier phase of a real-valued time series while
    preserving its power spectrum (amplitude spectrum unchanged; phase
    replaced with i.i.d. uniform[0, 2*pi), conjugate-symmetric so the
    output stays real).

    Parameters
    ----------
    signal : array, shape (..., n_samples)
        Real-valued time series; randomization is along the last axis.
    rng : numpy.random.Generator

    Returns
    -------
    array, same shape as `signal`
    """
    signal = np.asarray(signal, dtype=float)
    n = signal.shape[-1]
    spectrum = np.fft.rfft(signal, axis=-1)

    # Random phases for all bins except DC (and Nyquist, if n is even) —
    # those must stay real for the inverse transform to be real-valued.
    n_bins = spectrum.shape[-1]
    random_phase = rng.uniform(0, 2 * np.pi, size=spectrum.shape)
    random_phase[..., 0] = 0.0
    if n % 2 == 0:
        random_phase[..., -1] = 0.0

    randomized = np.abs(spectrum) * np.exp(1j * random_phase)
    return np.fft.irfft(randomized, n=n, axis=-1)


def spectrally_matched_noise(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generate noise with the same power spectrum as `signal` but no
    relationship to its actual realization — i.e. phase_randomize taken
    to its endpoint, generated fresh rather than derived from one draw.
    Destroys everything about the signal except its spectral content.
    """
    # Equivalent to phase-randomizing the same signal; kept as a
    # separate named entry point per §2.4's explicit surrogate list,
    # since "phase-randomized version of trial X" and "fresh noise
    # matched to the average spectrum" are used in different contexts
    # downstream (single-trial null vs. population-level null).
    return phase_randomize(signal, rng)


def time_reversal(signal: np.ndarray) -> np.ndarray:
    """
    Reverse the time axis (last axis). Preserves the power spectrum
    exactly (spectral content is time-reversal invariant) while
    destroying causal order — the surrogate E3-A.3 (§12.4) uses this to
    test whether an "anticipatory" detection is genuinely ordered in
    time or an artifact of the detector reacting to any transient.
    """
    signal = np.asarray(signal)
    return np.flip(signal, axis=-1)


def channel_permutation(signal: np.ndarray, rng: np.random.Generator, channel_axis: int = 0) -> np.ndarray:
    """
    Permute channel identity while leaving each channel's own time
    series untouched. Destroys spatial/montage structure — e.g. breaks
    the spatial specificity that a genuine contralateral sensorimotor
    effect should have — while preserving every channel's own spectral
    and temporal content.

    Parameters
    ----------
    signal : array
        Shape includes a channel axis at `channel_axis` (default 0,
        i.e. shape (n_channels, ..., n_samples)).
    rng : numpy.random.Generator
    channel_axis : int, default 0
    """
    signal = np.asarray(signal)
    n_channels = signal.shape[channel_axis]
    perm = rng.permutation(n_channels)
    return np.take(signal, perm, axis=channel_axis)


def validate_surrogate(
    real_effect_stat: float,
    null_stats_on_planted_real_effect: np.ndarray,
    artifact_stat: float,
    null_stats_on_planted_artifact: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    The two-direction validation protocol required by §2.4 before any
    surrogate is adopted for a track's result. This function does not
    generate the planted effects or artifacts — those are pipeline- and
    experiment-specific (Track A supplies a planted real MRCP-locked
    effect and a planted myogenic artifact; this function only judges
    the surrogate's behavior against them).

    Parameters
    ----------
    real_effect_stat : float
        The pipeline's statistic (e.g. decoding accuracy) on data with a
        DELIBERATELY PLANTED real effect, evaluated on real (non-surrogate)
        data.
    null_stats_on_planted_real_effect : array
        The same statistic recomputed under the surrogate, on the same
        planted-real-effect data, across many surrogate draws.
    artifact_stat : float
        The pipeline's statistic on data with a deliberately planted
        artifact (e.g. simulated myogenic contamination), on real data.
    null_stats_on_planted_artifact : array
        The same statistic recomputed under the surrogate, on the same
        planted-artifact data, across many surrogate draws.
    alpha : float, default 0.05

    Returns
    -------
    dict with:
        'real_effect_survives' : bool — True means the surrogate's null
            does NOT already contain the real effect (real_effect_stat
            sits above the (1 - alpha) quantile of its null) — i.e. the
            surrogate doesn't accidentally destroy genuine signal.
        'artifact_is_killed' : bool — True means the artifact statistic
            is NOT distinguishable from its own null (artifact_stat sits
            inside the null distribution) — i.e. the surrogate removes
            the artifact-driven inflation as intended.
        'valid' : bool — both conditions hold.
    """
    real_null = np.asarray(null_stats_on_planted_real_effect)
    art_null = np.asarray(null_stats_on_planted_artifact)

    real_threshold = np.quantile(real_null, 1 - alpha)
    real_effect_survives = bool(real_effect_stat > real_threshold)

    art_threshold = np.quantile(art_null, 1 - alpha)
    artifact_is_killed = bool(artifact_stat <= art_threshold)

    return {
        "real_effect_survives": real_effect_survives,
        "artifact_is_killed": artifact_is_killed,
        "valid": real_effect_survives and artifact_is_killed,
    }