"""
preprocessing/eeg_filters.py

Band-pass filtering for continuous EEG, applied before epoching.
Per prereg/PR-2026-01.md: 8-30 Hz, matching the source paper's band.

PATCH NOTE (for Developer A, from the run_t1r.py integration pass):
  1. Default `picks` looked for the 'EEG:' (colon) prefix, copied from
     the 2b filter. bci_iv_2a.py's own docstring documents 2a's real
     channel names as 'EEG-C3' etc. (hyphen) -- confirmed by running
     this against a synthetic Raw built with 2a's documented naming;
     the original crashed with "No appropriate channels found for the
     given picks ([])" before ever reaching epoching. Fixed by
     importing bci_iv_2a.EEG_CHANNEL_PREFIX directly so the two files
     can't drift out of sync again.
  2. Added an explicit `order` parameter (default 4, forward-backward
     Butterworth via MNE's iir_params) so filter order is a real,
     loggable value per §8.4, instead of an unparameterized MNE
     default that a reviewer can't recover from the log.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.bci_iv_2a import EEG_CHANNEL_PREFIX


def bandpass_filter(
    raw: mne.io.BaseRaw,
    low: float = 8.0,
    high: float = 30.0,
    order: int = 4,
    picks: list | None = None,
) -> mne.io.BaseRaw:
    """
    Args:
        raw: mne Raw object. Modified via .copy(), original untouched.
        low: low cutoff frequency in Hz (default 8.0, per prereg)
        high: high cutoff frequency in Hz (default 30.0, per prereg)
        order: Butterworth filter order, forward-backward (filtfilt),
            so effective attenuation is steeper than `order` alone
            suggests -- log the passed value, not the effective one,
            since that's what was configured (§8.4).
        picks: channel names to filter. Defaults to this dataset's EEG
            channels (excludes EOG), using the single source of truth
            in data.bci_iv_2a.EEG_CHANNEL_PREFIX rather than a
            hardcoded prefix that can silently drift out of sync with
            the loader.

    Returns:
        A new Raw object, band-pass filtered, EEG channels only if
        picks not specified.
    """
    filtered = raw.copy()

    if picks is None:
        picks = [ch for ch in filtered.ch_names if ch.startswith(EEG_CHANNEL_PREFIX)]
        if not picks:
            raise ValueError(
                f"No channels matched prefix '{EEG_CHANNEL_PREFIX}' in "
                f"{filtered.ch_names}. Channel naming convention may "
                f"have changed -- check data/bci_iv_2a.py before "
                f"silently filtering zero channels."
            )

    filtered.pick(picks)
    filtered.filter(
        l_freq=low,
        h_freq=high,
        picks=picks,
        method="iir",
        iir_params=dict(order=order, ftype="butter"),
        verbose=False,
    )
    return filtered

def surface_laplacian(
    raw: mne.io.BaseRaw,
    lambda2: float = 1e-5,
    stiffness: int = 4,
    picks: list | None = None,
) -> mne.io.BaseRaw:
    """
    Surface Laplacian (current source density) spatial filter, per the
    E3-A protocol doc's pipeline step 4 ("Clean the signal: spatial
    filter (surface Laplacian) ... No ICA required").

    Wraps mne.preprocessing.compute_current_source_density -- this is
    Track A's single implementation of "surface Laplacian" (per
    S8.1/S0.1: no local reimplementations of a shared preprocessing
    step). Requires channel positions (a montage) to already be set on
    `raw`; data/way_eeg_gal.py's loader is responsible for attaching
    the montage before this is called, same as it is for
    bandpass_filter's channel picks.

    Args:
        raw: mne Raw object with a montage set. Modified via .copy().
        lambda2: regularization parameter (mne default 1e-5).
        stiffness: spline stiffness (mne default 4).
        picks: channels to include. Defaults to all EEG channels.

    Returns:
        A new Raw object, spatially filtered. Units change from V to
        V/m^2 (mne's CSD convention) -- log this if downstream
        amplitude thresholds (preprocessing/eeg_artifacts.py) are
        compared against raw-EEG-scale values.
    """
    if raw.get_montage() is None:
        raise ValueError(
            "surface_laplacian requires a montage (channel positions) "
            "to be set on `raw` before calling this -- the loader "
            "(data/way_eeg_gal.py) must attach one. Refusing to run "
            "with an undefined spatial filter rather than silently "
            "falling back to a nearest-neighbor approximation."
        )
    filtered = raw.copy()
    if picks is not None:
        filtered.pick(picks)
    return mne.preprocessing.compute_current_source_density(
        filtered, lambda2=lambda2, stiffness=stiffness, copy=False
    )