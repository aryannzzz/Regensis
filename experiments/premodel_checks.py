"""
preprocessing/premodel_checks_2a.py

Same three required visualizations as premodel_checks.py, adapted for
BCI-IV-2a: 22 EEG channels (vs. 2b's 3), T/E session structure (vs.
2b's 5 sessions), channel naming 'EEG-C3' (hyphen, not colon).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.bci_iv_2a import load_all, load_subject

EPOCH_TMIN = 0.5   # seconds post-cue, per prereg
EPOCH_TMAX = 3.5   # seconds post-cue, per prereg


def _bandpass(raw: mne.io.BaseRaw, low: float, high: float, eeg_only: bool = True) -> mne.io.BaseRaw:
    filtered = raw.copy()
    if eeg_only:
        picks = [ch for ch in filtered.ch_names if ch.startswith("EEG-")]
        filtered.pick(picks)
        filtered.filter(l_freq=low, h_freq=high, picks=picks, method="iir", verbose=False)
    else:
        filtered.filter(l_freq=low, h_freq=high, method="iir", verbose=False)
    return filtered


def epoch(raw_filtered: mne.io.BaseRaw, events: np.ndarray, labels: np.ndarray):
    event_id = {"left": 0, "right": 1}
    mne_events = np.column_stack([events[:, 0], events[:, 1], labels])
    return mne.Epochs(
        raw_filtered, mne_events, event_id=event_id,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True, verbose=False,
    )


def plot_erd_maps(subject_id: str = "A01", session: str = "T", data_dir: Path = Path("data/raw/bci_iv_2a")):
    """Per-class ERD time-frequency maps over C3/C4 (now selecting from 22 channels)."""
    subj = load_subject(data_dir, int(subject_id[1:]))
    raw = subj.raw_by_session[session]
    events = subj.events_by_session[session]
    labels = subj.labels_by_session[session]

    filtered = _bandpass(raw, low=1.0, high=40.0)
    epochs = epoch(filtered, events, labels)

    freqs = np.arange(6, 35, 1)
    n_cycles = freqs / 2.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, ch_name in enumerate(["EEG-C3", "EEG-C4"]):
        for col, cls in enumerate(["left", "right"]):
            power = mne.time_frequency.tfr_morlet(
                epochs[cls], freqs=freqs, n_cycles=n_cycles,
                picks=[ch_name], return_itc=False, average=True,
                verbose=False,
            )
            ax = axes[row, col]
            power.plot([0], baseline=(EPOCH_TMIN, EPOCH_TMIN + 0.5),
                       mode="percent", axes=ax, show=False,
                       colorbar=(col == 1))
            ax.set_title(f"{ch_name} — {cls} hand")

    fig.suptitle(f"ERD time-frequency maps — {subject_id}, session {session}")
    fig.tight_layout()
    fig.savefig(f"preprocessing/_erd_maps_2a_{subject_id}.png", dpi=120)
    print(f"Saved preprocessing/_erd_maps_2a_{subject_id}.png — expect ERD "
          f"(power decrease) contralateral to imagined hand.")
    plt.close(fig)


def plot_csp_components(subject_id: str = "A01", session: str = "T", data_dir: Path = Path("data/raw/bci_iv_2a")):
    """First two CSP components, colored by class -- now using all 22 EEG channels."""
    subj = load_subject(data_dir, int(subject_id[1:]))
    raw = subj.raw_by_session[session]
    events = subj.events_by_session[session]
    labels = subj.labels_by_session[session]

    filtered = _bandpass(raw, low=8.0, high=30.0)
    epochs = epoch(filtered, events, labels)

    eeg_picks = [ch for ch in filtered.ch_names if ch.startswith("EEG-")]
    X = epochs.get_data(picks=eeg_picks)
    y = epochs.events[:, -1]

    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    features = csp.fit_transform(X, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    for cls, label, color in [(0, "left", "tab:blue"), (1, "right", "tab:orange")]:
        mask = y == cls
        ax.scatter(features[mask, 0], features[mask, 1], c=color, label=label, alpha=0.7)
    ax.set_xlabel("CSP component 1 (log-variance)")
    ax.set_ylabel("CSP component 2 (log-variance)")
    ax.set_title(f"CSP components by class — {subject_id}, session {session} (22ch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"preprocessing/_csp_components_2a_{subject_id}.png", dpi=120)
    print(f"Saved preprocessing/_csp_components_2a_{subject_id}.png — "
          f"STOP-AND-DEBUG GATE: with 22 channels this should separate "
          f"more cleanly than the 3-channel 2b version did. If it doesn't, "
          f"that's a real signal something's wrong (epoch timing, channel "
          f"mapping), not just 'expected noise' like it was with 2b.")
    plt.close(fig)


def plot_class_balance(data_dir: Path = Path("data/raw/bci_iv_2a")):
    all_subjects = load_all(data_dir)
    subject_ids = sorted(all_subjects.keys())
    left_counts, right_counts = [], []
    for sid in subject_ids:
        subj = all_subjects[sid]
        all_labels = np.concatenate([subj.labels_by_session[s] for s in subj.session_ids])
        left_counts.append(int(np.sum(all_labels == 0)))
        right_counts.append(int(np.sum(all_labels == 1)))

    x = np.arange(len(subject_ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, left_counts, width, label="left hand")
    ax.bar(x + width / 2, right_counts, width, label="right hand")
    ax.set_xticks(x)
    ax.set_xticklabels(subject_ids)
    ax.set_ylabel("Trial count")
    ax.set_title("Per-subject class balance (T+E sessions combined) — Dataset 2a")
    ax.legend()
    fig.tight_layout()
    fig.savefig("preprocessing/_class_balance_2a.png", dpi=120)
    print("Saved preprocessing/_class_balance_2a.png — expect ~144 per "
          "class per subject (72 left + 72 right per session x 2 sessions).")
    plt.close(fig)


if __name__ == "__main__":
    data_dir = Path("BCICIV_2a_gdf")

    print("1/3 — ERD time-frequency maps...")
    plot_erd_maps(data_dir=data_dir)

    print("2/3 — CSP component separability (22 channels)...")
    plot_csp_components(data_dir=data_dir)

    print("3/3 — per-subject class balance...")
    plot_class_balance(data_dir=data_dir)

    print("\nAll three required pre-modeling visualizations saved to preprocessing/.")