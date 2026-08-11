"""
preprocessing/epoching.py

Cue-relative epoching for T1-R, per prereg/PR-2026-01.md (0.5-3.5s
post-cue window, matching the source paper's timing).

NOTE: this is deliberately cue-relative, NOT onset-relative. The manual
flags onset-relative epoching (relative to EMG threshold) as a
requirement for E2/anticipation work specifically -- cue-aligned vs.
onset-aligned epoching answer different questions, and mixing them up
silently converts one kind of result into another (manual §"Epoching is
relative to EMG threshold-onset, not to trial start or cue"). T1-R is a
motor-imagery decoding reproduction, cued by design -- cue-relative is
correct here. Don't reuse this file's logic if E2 epoching is ever built;
that needs its own onset-detection step first.

Provides `epoch()`, matching the interface experiments/run_t1r.py
already expects: epoch(raw, events, labels, tmin, tmax, baseline) ->
(X, y) numpy arrays.

Replaces the epoch() functions duplicated inline in premodel_checks.py
and premodel_checks_2a.py (those return mne.Epochs, not arrays -- fine
for plotting, but keep this file's array-returning version as the one
experiments/run_t1r.py imports).
"""

from __future__ import annotations

import mne
import numpy as np

# Per prereg/PR-2026-01.md -- do not change without a deviations.md entry
EPOCH_TMIN = 0.5   # seconds post-cue
EPOCH_TMAX = 3.5   # seconds post-cue


def epoch(
    raw_filtered: mne.io.BaseRaw,
    events: np.ndarray,
    labels: np.ndarray,
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    baseline: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cut epochs relative to cue onset, return as numpy arrays.

    NOTE: named `epoch`, not `epoch_cue_relative` -- this matches the
    interface experiments/run_t1r.py already imports and calls
    (preprocessing.epoching.epoch(raw, events, labels, tmin, tmax,
    baseline) -> (X, y)). Keep this signature/name stable; changing it
    breaks the orchestrator without warning beyond a fallback to
    synthetic demo mode.

    Args:
        raw_filtered: band-passed continuous data (see eeg_filters.py)
        events: (n_events, 3) array of [sample, prev_id, code] -- from
                the loader's events_by_session, already filtered to
                left/right-hand trials and aligned with `labels`
        labels: (n_events,) array of 0/1, same order as events
        tmin, tmax: window relative to cue onset, in seconds. Defaults
                    to the pre-registered window -- pass explicit values
                    only for diagnostic/exploratory work, never for
                    anything that touches results/PR-2026-01/
        baseline: mne baseline-correction tuple, or None for no
                  correction. T1-R's pre-reg doesn't specify baseline
                  correction, so default is None.

    Returns:
        (X, y): X shape (n_epochs, n_channels, n_times), y shape (n_epochs,)
    """
    if len(events) != len(labels):
        raise ValueError(
            f"events ({len(events)}) and labels ({len(labels)}) length "
            f"mismatch -- these must be aligned before calling this."
        )

    event_id = {"left": 0, "right": 1}
    mne_events = np.column_stack([events[:, 0], events[:, 1], labels])

    mne_epochs = mne.Epochs(
        raw_filtered, mne_events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=baseline, preload=True, verbose=False,
    )

    X = mne_epochs.get_data()          # (n_epochs, n_channels, n_times)
    y = mne_epochs.events[:, -1]       # (n_epochs,)
    return X, y


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.bci_iv_2b import load_subject
    from preprocessing.eeg_filters import bandpass_filter

    data_dir = Path("data/raw/bci_iv_2b")
    subj = load_subject(data_dir, subject_num=1)

    raw = subj.raw_by_session["01T"]
    events = subj.events_by_session["01T"]
    labels = subj.labels_by_session["01T"]

    filtered = bandpass_filter(raw)
    X, y = epoch(filtered, events, labels)

    print(f"Epoched {X.shape[0]} trials, shape {X.shape}, window [{EPOCH_TMIN}, {EPOCH_TMAX}]s post-cue")
    print(f"Left: {int((y==0).sum())}, Right: {int((y==1).sum())}")