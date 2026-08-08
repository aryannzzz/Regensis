"""
data/bci_iv_2a.py

Loader for BCI Competition IV Dataset 2a, filtered to the 2-class
(left hand vs. right hand) subset per prereg/PR-2026-01.md.

Expects raw .gdf + true-label .mat files named:
    A0101T.gdf / A01T.mat   -> written here as A01T.gdf / A01T.mat
    A01E.gdf / A01E.mat
    ... through subject 9 (A09*)

Per-subject structure (confirmed via check_labels.py on real data):
    2 sessions: T (training) and E (evaluation)
    288 trials/session, 72 per class, 4 classes total:
        1 = left hand, 2 = right hand, 3 = foot, 4 = tongue
    We keep only classes 1 and 2.

Channels (25 total): 22 EEG (10-20 system) + 3 EOG. EOG excluded before
classification, same as the 2b loader.

Event codes (confirmed empirically via check_labels.py on real data):
    T session: real per-class codes present directly --
               769=left, 770=right, 771=foot, 772=tongue
    E session: 768 -> start of trial
               783 -> cue onset, class MASKED -- real class comes from
                      the paired .mat file's 'classlabel' field
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import mne
except ImportError as e:
    raise ImportError("This loader requires mne. Install with: pip install mne") from e

try:
    from scipy.io import loadmat
except ImportError as e:
    raise ImportError("This loader requires scipy. Install with: pip install scipy") from e


LEFT_HAND = 1
RIGHT_HAND = 2
KEEP_CLASSES = {LEFT_HAND, RIGHT_HAND}
CLASS_REMAP = {LEFT_HAND: 0, RIGHT_HAND: 1}  # 0=left, 1=right, matches 2b convention

# T-session raw GDF codes map directly to the same 1-4 class numbering
# as the .mat 'classlabel' field.
T_SESSION_CODE_TO_CLASS = {769: 1, 770: 2, 771: 3, 772: 4}

EEG_CHANNEL_PREFIX = "EEG-"  # note: hyphen, not colon -- differs from 2b's naming


def _load_true_labels(mat_path: Path) -> np.ndarray:
    """Load raw 1-4 class labels from a true_labels .mat file. NOT remapped yet."""
    mat = loadmat(mat_path)
    if "classlabel" not in mat:
        raise KeyError(
            f"Expected 'classlabel' key in {mat_path}, found: "
            f"{[k for k in mat.keys() if not k.startswith('__')]}"
        )
    return mat["classlabel"].ravel()  # values 1-4


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_hash(data_dir: Path) -> str:
    all_files = sorted(data_dir.glob("*.gdf")) + sorted(data_dir.glob("*.mat"))
    if not all_files:
        raise FileNotFoundError(f"No .gdf or .mat files found in {data_dir}")
    combined = hashlib.sha256()
    for f in all_files:
        combined.update(_sha256_of_file(f).encode())
    return combined.hexdigest()


@dataclass
class BCIIV2aData:
    subject_id: str                     # e.g. "A01"
    session_ids: list[str]              # ["T", "E"]
    raw_by_session: dict                # session_id -> mne.io.Raw (EEG+EOG, unfiltered)
    events_by_session: dict             # session_id -> np.ndarray, filtered to left/right only
    labels_by_session: dict             # session_id -> np.ndarray of 0/1 (0=left, 1=right)
    fs: float
    channel_names: list[str] = field(default_factory=list)


def load_subject(data_dir: Path, subject_num: int) -> BCIIV2aData:
    """
    Load T and E sessions for one subject, filtered to left/right-hand
    trials only.

    Args:
        data_dir: folder containing .gdf and paired .mat files
        subject_num: 1 through 9

    Returns:
        BCIIV2aData with 2-class-filtered events and labels.
    """
    subject_id = f"A{subject_num:02d}"
    session_suffixes = ["T", "E"]

    raw_by_session = {}
    events_by_session = {}
    labels_by_session = {}
    fs = None
    channel_names = []

    for suffix in session_suffixes:
        gdf_path = data_dir / f"{subject_id}{suffix}.gdf"

        if not gdf_path.exists():
            raise FileNotFoundError(f"Expected file not found: {gdf_path}")

        raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)

        code_lookup = {
            mne_code: int(desc)
            for desc, mne_code in event_id_map.items()
            if desc.lstrip("-").isdigit()
        }

        if suffix == "T":
            # Real per-class codes (769-772) are embedded directly.
            keep_mask = np.array([
                code_lookup.get(row[2], -1) in T_SESSION_CODE_TO_CLASS
                for row in events
            ])
            all_class_events = events[keep_mask]
            raw_labels_4class = np.array([
                T_SESSION_CODE_TO_CLASS[code_lookup[row[2]]]
                for row in all_class_events
            ])
        else:
            # E session: cue is masked as 783, real class comes from .mat
            mat_path = data_dir / f"{subject_id}{suffix}.mat"
            if not mat_path.exists():
                raise FileNotFoundError(
                    f"True-labels file not found: {mat_path}. Download from "
                    f"https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip"
                )

            cue_mask = np.array([
                code_lookup.get(row[2], -1) == 783 for row in events
            ])
            all_class_events = events[cue_mask]
            raw_labels_4class = _load_true_labels(mat_path)  # values 1-4, len 288

            if len(raw_labels_4class) != len(all_class_events):
                raise ValueError(
                    f"{subject_id}{suffix}: found {len(all_class_events)} "
                    f"cue events in the .gdf but {len(raw_labels_4class)} "
                    f"labels in the .mat file -- these should match "
                    f"(both should be 288)."
                )

        # Filter down to left/right hand only, keep alignment intact.
        keep_mask = np.isin(raw_labels_4class, list(KEEP_CLASSES))
        class_events = all_class_events[keep_mask]
        labels = np.array([CLASS_REMAP[lab] for lab in raw_labels_4class[keep_mask]])

        raw_by_session[suffix] = raw
        events_by_session[suffix] = class_events
        labels_by_session[suffix] = labels

        if fs is None:
            fs = raw.info["sfreq"]
            channel_names = raw.ch_names

    return BCIIV2aData(
        subject_id=subject_id,
        session_ids=session_suffixes,
        raw_by_session=raw_by_session,
        events_by_session=events_by_session,
        labels_by_session=labels_by_session,
        fs=fs,
        channel_names=channel_names,
    )


def load_all(data_dir: Path) -> dict[str, BCIIV2aData]:
    return {f"A{n:02d}": load_subject(data_dir, n) for n in range(1, 10)}


if __name__ == "__main__":
    data_dir = Path("data/raw/bci_iv_2a")
    subj = load_subject(data_dir, subject_num=1)
    print(f"Loaded {subj.subject_id}")
    print(f"Sessions: {subj.session_ids}")
    print(f"Sampling rate: {subj.fs} Hz")
    print(f"Channels: {subj.channel_names}")
    for sid in subj.session_ids:
        n = len(subj.labels_by_session[sid])
        n_left = int(np.sum(subj.labels_by_session[sid] == 0))
        n_right = int(np.sum(subj.labels_by_session[sid] == 1))
        print(f"  Session {sid}: {n} left/right trials ({n_left} left, {n_right} right)")