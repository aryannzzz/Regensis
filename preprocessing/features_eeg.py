"""
preprocessing/features_eeg.py

MRCP (movement-related cortical potential) feature extraction in
sliding windows preceding EMG-marked onset, per the E3-A protocol
doc's pipeline step 3: "Extract MRCP features (0.3-3 Hz) in sliding
windows preceding the marked onset."

This is deliberately a *different* feature representation from
eeg/decoders_linear.py's CSP path -- E3-A's detectors (eeg/anticipation.py)
classify "movement coming" vs. "nothing happening" from short windows
well before a labeled event, not from a single epoch cut to a known
trial. Band power (E2-A's second feature path, 8-30 Hz) is a different
question entirely -- this file is MRCP-only, per prereg/PR-2026-02.md.

Provides:
  mrcp_bandpass(raw, ...)        -- thin wrapper over eeg_filters.bandpass_filter
                                     pinned to the MRCP band, so a caller
                                     can't accidentally reuse the 8-30 Hz
                                     E2-A band here (S6.1's #4 common
                                     mistake: pooling bands across
                                     experiments is a distinct scope error
                                     from pooling pre/post-onset windows).
  sliding_windows(X, onset_idx, ...) -- cut fixed-length windows ending at
                                     a grid of offsets before onset,
                                     labeled 1 ("movement coming", within
                                     the pre-onset horizon) or 0 ("nothing
                                     happening", far pre-onset / rest).
  window_features(X_windows)     -- per-window feature vector: mean
                                     amplitude and slope per channel.
                                     Input to eeg/anticipation.py's sLDA
                                     detector. EEGNet
                                     (eeg/decoder_eegnet.py) instead
                                     consumes X_windows directly (raw
                                     windowed signal, not these
                                     hand-crafted features) -- see that
                                     file's docstring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# mne is only needed by mrcp_bandpass() (continuous-Raw filtering) --
# NOT by sliding_windows()/window_features(), which operate on plain
# numpy arrays already cut from a trial. Guarded/lazy import, same
# pattern as data/bci_iv_2a.py's mne/scipy guards, so callers that only
# need the array-level functions (e.g. experiments/run_e3a.py's demo
# mode, or any unit test) don't need mne installed at all.
MRCP_LOW_HZ = 0.3
MRCP_HIGH_HZ = 3.0


def mrcp_bandpass(raw, order: int = 4, picks: list | None = None):
    """MRCP-band (0.3-3 Hz) filtering of a continuous mne Raw. Pinned
    band -- see module docstring."""
    try:
        import mne  # noqa: F401  -- import kept local, see module note above
    except ImportError as e:
        raise ImportError("mrcp_bandpass requires mne. Install with: pip install mne") from e
    from preprocessing.eeg_filters import bandpass_filter
    return bandpass_filter(raw, low=MRCP_LOW_HZ, high=MRCP_HIGH_HZ,
                            order=order, picks=picks)


@dataclass
class SlidingWindowConfig:
    window_s: float = 0.5           # window length, seconds
    step_s: float = 0.05            # hop between window centers, seconds
    horizon_s: float = 1.5          # "movement coming" horizon: windows whose
                                     # right edge falls within this many
                                     # seconds before onset are labeled 1
    rest_margin_s: float = 2.0      # windows must end at least this long
                                     # before the horizon (or after a prior
                                     # trial's onset) to count as label-0
                                     # "nothing happening" rest


def sliding_windows(
    X: np.ndarray,
    fs: float,
    onset_sample: int,
    cfg: SlidingWindowConfig = SlidingWindowConfig(),
    trial_start_sample: int = 0,
) -> dict:
    """
    Cut sliding windows from one continuous (or long-epoch) MRCP-filtered
    trial and label them relative to EMG-marked onset.

    Args:
        X: (n_channels, n_times) MRCP-filtered signal for one trial,
           already onset-anchored in time (sample `onset_sample` is the
           EMG threshold-crossing sample -- from eeg/anticipation.py's
           onset detector, never a cue or mechanical trigger, per the
           protocol doc's Dataset section).
        fs: sampling rate, Hz.
        onset_sample: index into X's time axis of EMG-marked onset.
        cfg: window/horizon parameters -- real, loggable (S8.4).
        trial_start_sample: earliest usable sample (e.g. after a
            settle period) -- windows before this are dropped.

    Returns:
        dict with:
          X_windows: (n_windows, n_channels, window_len) array
          labels: (n_windows,) int, 1 = "movement coming" (within horizon,
                   pre-onset), 0 = "nothing happening" (rest)
          window_end_sample: (n_windows,) sample index of each window's
                   right edge, for computing lead time later
          config: the SlidingWindowConfig used
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected (n_channels, n_times), got shape {X.shape}")
    n_channels, n_times = X.shape

    window_len = int(round(cfg.window_s * fs))
    step = max(1, int(round(cfg.step_s * fs)))
    horizon_samples = int(round(cfg.horizon_s * fs))
    rest_margin_samples = int(round(cfg.rest_margin_s * fs))

    X_windows, labels, ends = [], [], []
    for end in range(trial_start_sample + window_len, n_times, step):
        start = end - window_len
        lead_samples = onset_sample - end  # positive = window ends before onset
        if lead_samples < 0:
            # window overlaps or follows onset -- E3-A is an
            # *anticipation* detector; post-onset windows belong to a
            # different (post-hoc decoding) question and are dropped
            # here rather than silently mixed in as either class.
            continue
        if lead_samples <= horizon_samples:
            label = 1
        elif lead_samples >= horizon_samples + rest_margin_samples:
            label = 0
        else:
            # buffer zone between "coming" and "rest" -- ambiguous,
            # dropped rather than forced into either class.
            continue
        X_windows.append(X[:, start:end])
        labels.append(label)
        ends.append(end)

    return {
        "X_windows": np.stack(X_windows) if X_windows else np.empty((0, n_channels, window_len)),
        "labels": np.array(labels, dtype=int),
        "window_end_sample": np.array(ends, dtype=int),
        "config": cfg,
    }


def window_features(X_windows: np.ndarray, fs: float) -> np.ndarray:
    """
    Hand-crafted per-window feature vector for the sLDA detector:
    mean amplitude and linear slope per channel (the two quantities
    the MRCP literature actually reads off a pre-movement window --
    a negative-going slow shift and its rate of change). EEGNet
    (eeg/decoder_eegnet.py) does NOT use this -- it consumes
    X_windows directly so the CNN can learn its own spatial/temporal
    filters, per the protocol doc's "sLDA (background baseline) and
    EEGNet (main model)" framing: sLDA is deliberately the simple,
    hand-featured baseline.

    Args:
        X_windows: (n_windows, n_channels, window_len)
        fs: sampling rate, Hz (for the slope's time units)

    Returns:
        (n_windows, 2 * n_channels) feature matrix: [mean_ch0, ..., mean_chN,
        slope_ch0, ..., slope_chN]
    """
    X_windows = np.asarray(X_windows)
    n_windows, n_channels, window_len = X_windows.shape
    t = np.arange(window_len) / fs

    means = X_windows.mean(axis=2)  # (n_windows, n_channels)

    # Per-window, per-channel linear slope via least-squares. Vectorized
    # over windows*channels rather than a python double loop.
    t_centered = t - t.mean()
    denom = np.sum(t_centered ** 2)
    x_centered = X_windows - X_windows.mean(axis=2, keepdims=True)
    slopes = np.einsum("wct,t->wc", x_centered, t_centered) / denom  # (n_windows, n_channels)

    return np.concatenate([means, slopes], axis=1)