"""
preprocessing/premodel_checks.py

The three visualizations the Track A manual requires BEFORE any
modeling: per-class ERD time-frequency maps, first two CSP components
colored by class, and per-subject class balance.

This is a diagnostic/gate script, not part of the final pipeline --
run it once, look at the plots, and only proceed to the classifier if
what you see looks right (see the stop-and-debug gate in the T1-R plan).

NOTE: epoching here is done inline, self-contained, so this script
isn't blocked waiting on a separate preprocessing/epoching.py. If/when
that file exists, swap the inline epoch() call below for it -- keep
the window (0.5-3.5s post-cue) consistent with prereg/PR-2026-01.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.bci_iv_2b import load_all, load_subject
from preprocessing.eeg_filters import bandpass_filter

EPOCH_TMIN = 0.5   # seconds post-cue, per prereg
EPOCH_TMAX = 3.5   # seconds post-cue, per prereg


def epoch(raw_filtered: mne.io.BaseRaw, events: np.ndarray, labels: np.ndarray):
    """
    Minimal inline epoching. Cuts trials from EPOCH_TMIN to EPOCH_TMAX
    relative to each event's onset sample.
    """
    event_id = {"left": 0, "right": 1}
    mne_events = np.column_stack([
        events[:, 0], events[:, 1], labels
    ])
    epochs = mne.Epochs(
        raw_filtered, mne_events, event_id=event_id,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True, verbose=False,
    )
    return epochs


def plot_erd_maps(subject_id: str = "B01", session: str = "01T", data_dir: Path = Path("data/raw/bci_iv_2b")):
    """Per-class ERD time-frequency maps over C3/C4."""
    subj = load_subject(data_dir, int(subject_id[1:]))
    raw = subj.raw_by_session[session]
    events = subj.events_by_session[session]
    labels = subj.labels_by_session[session]

    filtered = bandpass_filter(raw, low=1.0, high=40.0)  # wider band for TF map
    epochs = epoch(filtered, events, labels)

    freqs = np.arange(6, 35, 1)
    n_cycles = freqs / 2.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, ch_name in enumerate(["EEG:C3", "EEG:C4"]):
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
    fig.savefig(f"preprocessing/_erd_maps_{subject_id}.png", dpi=120)
    print(f"Saved preprocessing/_erd_maps_{subject_id}.png — check that "
          f"ERD (power decrease, blue) appears contralateral to the "
          f"imagined hand: right-hand imagery -> ERD over C3, "
          f"left-hand imagery -> ERD over C4.")
    plt.close(fig)


def plot_csp_components(subject_id: str = "B01", session: str = "01T", data_dir: Path = Path("data/raw/bci_iv_2b")):
    """First two CSP components, colored by class."""
    subj = load_subject(data_dir, int(subject_id[1:]))
    raw = subj.raw_by_session[session]
    events = subj.events_by_session[session]
    labels = subj.labels_by_session[session]

    filtered = bandpass_filter(raw, low=8.0, high=30.0)  # analysis band
    epochs = epoch(filtered, events, labels)

    X = epochs.get_data(picks=["EEG:C3", "EEG:Cz", "EEG:C4"])
    y = epochs.events[:, -1]

    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    features = csp.fit_transform(X, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    for cls, label, color in [(0, "left", "tab:blue"), (1, "right", "tab:orange")]:
        mask = y == cls
        ax.scatter(features[mask, 0], features[mask, 1], c=color, label=label, alpha=0.7)
    ax.set_xlabel("CSP component 1 (log-variance)")
    ax.set_ylabel("CSP component 2 (log-variance)")
    ax.set_title(f"CSP components by class — {subject_id}, session {session}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"preprocessing/_csp_components_{subject_id}.png", dpi=120)
    print(f"Saved preprocessing/_csp_components_{subject_id}.png — "
          f"STOP-AND-DEBUG GATE: if left/right classes aren't visibly "
          f"separated here for your best subjects, do not proceed to "
          f"classifier tuning. Debug the feature extraction first.")
    plt.close(fig)


def plot_class_balance(data_dir: Path = Path("data/raw/bci_iv_2b")):
    """Per-subject class balance across all sessions."""
    all_subjects = load_all(data_dir)

    subject_ids = sorted(all_subjects.keys())
    left_counts = []
    right_counts = []
    for sid in subject_ids:
        subj = all_subjects[sid]
        all_labels = np.concatenate([
            subj.labels_by_session[s] for s in subj.session_ids
        ])
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
    ax.set_title("Per-subject class balance (all sessions combined)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("preprocessing/_class_balance.png", dpi=120)
    print("Saved preprocessing/_class_balance.png — check no subject is "
          "badly imbalanced; roughly equal left/right counts are expected "
          "by design for this dataset.")
    plt.close(fig)


if __name__ == "__main__":
    data_dir = Path("BCICIV_2b_gdf")

    print("1/3 — ERD time-frequency maps...")
    plot_erd_maps(data_dir=data_dir)

    print("2/3 — CSP component separability...")
    plot_csp_components(data_dir=data_dir)

    print("3/3 — per-subject class balance...")
    plot_class_balance(data_dir=data_dir)

    print("\nAll three required pre-modeling visualizations saved to "
          "preprocessing/. Review them before writing the classifier.")