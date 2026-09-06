"""
preprocessing/eeg_artifacts.py

Trial-level artifact rejection.

E3-A (prereg/PR-2026-02.md, per the E3-A protocol doc's Pipeline step 4)
explicitly does NOT require ICA: "Clean the signal: spatial filter
(surface Laplacian) plus rejection of trials with amplitude spikes or
abnormal statistics. No ICA required." That's deliberate -- MRCP is a
slow (0.3-3 Hz) potential and ICA's usual job here is ocular-artifact
removal for faster components; E2-A's protocol calls for ICA (manual
S5.2, "validated against a planted effect before adoption") but E3-A
does not, so an ICA step belongs in a future E2-A artifact module, not
here. Keep this file to the two rejection criteria the doc actually
asks for -- don't reach for ICA "to be safe."

Provides reject_trials(), which flags (not silently drops) epochs by:
  1. absolute-amplitude threshold ("amplitude spikes")
  2. per-epoch kurtosis outlier ("abnormal statistics") -- a standard
     proxy for non-Gaussian, spike-like contamination that a raw
     amplitude threshold alone can miss (e.g. a spike that stays under
     the absolute cutoff but is wildly peaked relative to the rest of
     the epoch's own distribution).

Everything here is a *rejection mask*, not a rewrite of the data --
callers apply the mask and log how many trials were dropped and why
(S8.4: every preprocessing parameter, including rejection criteria and
the resulting reject count, must be logged).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis as _kurtosis


@dataclass
class RejectionConfig:
    amplitude_uv: float = 100.0     # abs amplitude threshold, microvolts
    kurtosis_z: float = 5.0         # per-epoch kurtosis, z-score vs. the
                                     # trial population, above which an
                                     # epoch is flagged as spike-like


def reject_trials(
    X: np.ndarray,
    cfg: RejectionConfig = RejectionConfig(),
) -> dict:
    """
    Args:
        X: (n_epochs, n_channels, n_times), already spatially filtered
           (surface Laplacian; see preprocessing/eeg_filters.py) and
           band-passed to the MRCP band (0.3-3 Hz).
        cfg: thresholds. Both are real, loggable parameters (S8.4) --
             never hardcode a threshold inline at the call site.

    Returns:
        dict with:
          keep_mask: (n_epochs,) bool, True = trial survives
          amplitude_flagged: (n_epochs,) bool
          kurtosis_flagged: (n_epochs,) bool
          n_rejected: int
          config: the RejectionConfig used (for the log)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"Expected (n_epochs, n_channels, n_times), got shape {X.shape}")

    # Criterion 1: amplitude spikes. Peak absolute amplitude per epoch,
    # across all channels -- one contaminated channel is enough to
    # flag the trial (MRCP is a whole-scalp slow potential; a spike on
    # any channel is not the signal of interest).
    peak_amp = np.max(np.abs(X), axis=(1, 2))
    amplitude_flagged = peak_amp > cfg.amplitude_uv

    # Criterion 2: abnormal statistics. Per-epoch kurtosis, averaged
    # across channels, then z-scored against the trial population --
    # "abnormal" is relative to this recording's own epochs, per the
    # doc, not an absolute kurtosis cutoff (which would depend on
    # window length and would not transfer across subjects).
    per_epoch_kurtosis = np.mean(_kurtosis(X, axis=2, fisher=True), axis=1)
    mu, sigma = np.mean(per_epoch_kurtosis), np.std(per_epoch_kurtosis)
    if sigma == 0:
        kurtosis_flagged = np.zeros(len(X), dtype=bool)
    else:
        z = (per_epoch_kurtosis - mu) / sigma
        kurtosis_flagged = z > cfg.kurtosis_z

    reject_mask = amplitude_flagged | kurtosis_flagged
    keep_mask = ~reject_mask

    return {
        "keep_mask": keep_mask,
        "amplitude_flagged": amplitude_flagged,
        "kurtosis_flagged": kurtosis_flagged,
        "n_rejected": int(reject_mask.sum()),
        "n_total": int(len(X)),
        "config": cfg,
    }