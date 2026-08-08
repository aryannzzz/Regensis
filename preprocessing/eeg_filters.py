"""
preprocessing/eeg_filters.py

Band-pass filtering for continuous EEG, applied before epoching.
Per prereg/PR-2026-01.md: 8-30 Hz, matching the source paper's band.
"""

from __future__ import annotations

import mne


def bandpass_filter(
    raw: mne.io.BaseRaw,
    low: float = 8.0,
    high: float = 30.0,
    picks: list[str] | None = None,
) -> mne.io.BaseRaw:
    """
    Apply a band-pass filter to continuous EEG data.

    Args:
        raw: mne Raw object (e.g. from bci_iv_2b.load_subject's
             raw_by_session values). Modified via .copy(), original
             is untouched.
        low: low cutoff frequency in Hz (default 8.0, per prereg)
        high: high cutoff frequency in Hz (default 30.0, per prereg)
        picks: channel names to filter. Defaults to EEG channels only
               (C3, Cz, C4) -- EOG channels are excluded per the
               dataset card, since they're not used for classification.

    Returns:
        A new Raw object, band-pass filtered, EEG channels only if
        picks not specified.
    """
    filtered = raw.copy()

    if picks is None:
        # This dataset's EEG channels are named 'EEG:C3', 'EEG:Cz', 'EEG:C4'
        picks = [ch for ch in filtered.ch_names if ch.startswith("EEG:")]

    filtered.pick(picks)
    filtered.filter(
        l_freq=low,
        h_freq=high,
        picks=picks,
        method="iir",
        verbose=False,
    )
    return filtered


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.bci_iv_2b import load_subject

    data_dir = Path("data/raw/bci_iv_2b")
    subj = load_subject(data_dir, subject_num=1)

    raw = subj.raw_by_session["01T"]
    print(f"Before filtering: {raw.ch_names}, {raw.info['sfreq']} Hz")

    filtered = bandpass_filter(raw)
    print(f"After filtering: {filtered.ch_names}")
    print(f"Filter applied: {filtered.info['highpass']}-{filtered.info['lowpass']} Hz")