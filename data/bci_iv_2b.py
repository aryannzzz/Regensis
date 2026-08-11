"""
data/bci_iv_2b.py

Loader for BCI Competition IV Dataset 2b.

Expects the raw .gdf files as distributed by BNCI Horizon 2020 / the
original BCI Competition IV site, named like:
    B0101T.gdf, B0102T.gdf, B0103T.gdf   (subject 1, training sessions 1-3)
    B0104E.gdf, B0105E.gdf               (subject 1, evaluation sessions 4-5)
    ... through subject 9 (B09*)

'T' = training session (labels included in the file's event markers,
      codes 769/770)
'E' = evaluation session (the .gdf event markers are masked as code 783,
      "Cue unknown" -- real labels come from a separate .mat file,
      e.g. B0104E.mat, downloaded from
      https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip
      and placed in the same folder as the .gdf files)

Class event codes in this dataset (per the official description):
    769 -> left hand  (class 0 here)
    770 -> right hand (class 1 here)
    Other codes (eye movement calibration, etc.) are ignored.

Install requirement: mne (`pip install mne --break-system-packages` if
running standalone; add to env.lock once confirmed).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import mne
except ImportError as e:
    raise ImportError(
        "This loader requires mne. Install with: pip install mne"
    ) from e

try:
    from scipy.io import loadmat
except ImportError as e:
    raise ImportError(
        "This loader requires scipy for .mat label files. "
        "Install with: pip install scipy"
    ) from e


def _load_true_labels(mat_path: Path) -> np.ndarray:
    """
    Load true labels from a BCI Competition IV true_labels .mat file.

    These files use the standard competition format: a struct with a
    'classlabel' field, values 1 (left hand) or 2 (right hand), one per
    trial, in trial order matching the corresponding .gdf file.

    Returns labels remapped to 0/1 to match CLASS_MAP (0=left, 1=right).
    """
    mat = loadmat(mat_path)
    if "classlabel" not in mat:
        raise KeyError(
            f"Expected 'classlabel' key in {mat_path}, found: "
            f"{[k for k in mat.keys() if not k.startswith('__')]}"
        )
    raw_labels = mat["classlabel"].ravel()  # values are 1 or 2
    return raw_labels - 1  # remap to 0 (left) / 1 (right)


LEFT_HAND_CODE = 769
RIGHT_HAND_CODE = 770
CLASS_MAP = {LEFT_HAND_CODE: 0, RIGHT_HAND_CODE: 1}  # 0=left, 1=right


@dataclass
class BCIIV2bData:
    """Canonical container returned by load_subject / load_all.

    X: continuous EEG, shape (n_sessions, n_channels, n_timepoints) is NOT
       what you get here — see epoching.py for that. This loader returns
       CONTINUOUS per-session recordings plus event onsets, because
       epoching decisions (window, baseline) belong to preprocessing,
       not to the loader.
    """
    subject_id: str                     # e.g. "B01"
    session_ids: list[str]              # e.g. ["01T", "02T", "03T", "04E", "05E"]
    raw_by_session: dict                # session_id -> mne.io.Raw
    events_by_session: dict             # session_id -> np.ndarray (n_events, 3) [sample, 0, code]
    labels_by_session: dict             # session_id -> np.ndarray (n_events,) of 0/1, only for events matching CLASS_MAP
    fs: float                           # sampling rate, should be 250.0
    channel_names: list[str] = field(default_factory=list)  # should be ~['C3','Cz','C4', EOG...]


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_hash(data_dir: Path) -> str:
    """
    Hash all .gdf AND .mat files in data_dir together (sorted by filename
    for determinism) to get a single reproducibility fingerprint for
    data.sha256. Run this once you've confirmed your download is complete
    and record the output in results/PR-2026-01/data.sha256.
    """
    all_files = sorted(data_dir.glob("*.gdf")) + sorted(data_dir.glob("*.mat"))
    if not all_files:
        raise FileNotFoundError(f"No .gdf or .mat files found in {data_dir}")
    combined = hashlib.sha256()
    for f in all_files:
        combined.update(_sha256_of_file(f).encode())
    return combined.hexdigest()


def load_subject(data_dir: Path, subject_num: int) -> BCIIV2bData:
    """
    Load sessions for one subject.

    The 04E/05E evaluation sessions ship with placeholder event code 783
    ("Cue unknown") instead of real 769/770 labels in the .gdf files
    themselves. True labels for these sessions come from separate .mat
    files (downloaded from
    https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip),
    expected to sit alongside the .gdf files in data_dir, named e.g.
    B0104E.mat next to B0104E.gdf.

    Args:
        data_dir: folder containing both the raw .gdf files and the
            true_labels .mat files for E sessions
        subject_num: 1 through 9

    Returns:
        BCIIV2bData with continuous raw recordings and extracted events,
        keyed by session id.
    """
    subject_id = f"B{subject_num:02d}"
    session_suffixes = ["01T", "02T", "03T", "04E", "05E"]

    raw_by_session = {}
    events_by_session = {}
    labels_by_session = {}
    fs = None
    channel_names = []

    for suffix in session_suffixes:
        fpath = data_dir / f"{subject_id}{suffix}.gdf"
        if not fpath.exists():
            raise FileNotFoundError(f"Expected file not found: {fpath}")

        raw = mne.io.read_raw_gdf(fpath, preload=True, verbose=False)
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)

        # event_id_map maps annotation description strings -> integer codes
        # MNE's own codes, not the dataset's original 769/770/768 -- we
        # need to find which of MNE's codes correspond to the original
        # ones. The GDF annotation descriptions for this dataset are
        # literally the numeric codes as strings, e.g. "769", "770".
        code_lookup = {
            mne_code: int(desc)
            for desc, mne_code in event_id_map.items()
            if desc.lstrip("-").isdigit()
        }

        is_evaluation_session = suffix.endswith("E")

        if is_evaluation_session:
            # Trial onsets are marked by code 768 ("Start of a trial") in
            # E sessions -- 769/770 are NOT present (they're masked as
            # 783, "Cue unknown"). Use 768 to find trial positions, then
            # pull the real label from the separate .mat file, in order.
            trial_start_mask = np.array([
                code_lookup.get(row[2], -1) == 768 for row in events
            ])
            class_events = events[trial_start_mask]

            mat_path = data_dir / f"{subject_id}{suffix}.mat"
            if not mat_path.exists():
                raise FileNotFoundError(
                    f"True-labels file not found: {mat_path}. "
                    f"Download from "
                    f"https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip"
                )
            labels = _load_true_labels(mat_path)

            if len(labels) != len(class_events):
                raise ValueError(
                    f"{subject_id}{suffix}: found {len(class_events)} "
                    f"trial-start events in the .gdf but {len(labels)} "
                    f"labels in the .mat file -- these should match. "
                    f"Check the file pairing before trusting this session."
                )
        else:
            # Training sessions: real 769/770 codes are embedded directly.
            keep_mask = np.array([
                code_lookup.get(row[2], -1) in CLASS_MAP for row in events
            ])
            class_events = events[keep_mask]
            labels = np.array([
                CLASS_MAP[code_lookup[row[2]]] for row in class_events
            ])

        raw_by_session[suffix] = raw
        events_by_session[suffix] = class_events
        labels_by_session[suffix] = labels

        if fs is None:
            fs = raw.info["sfreq"]
            channel_names = raw.ch_names

    return BCIIV2bData(
        subject_id=subject_id,
        session_ids=session_suffixes,
        raw_by_session=raw_by_session,
        events_by_session=events_by_session,
        labels_by_session=labels_by_session,
        fs=fs,
        channel_names=channel_names,
    )


def load_all(data_dir: Path) -> dict[str, BCIIV2bData]:
    """Load all 9 subjects. Returns dict keyed by subject_id, e.g. 'B01'."""
    return {
        f"B{n:02d}": load_subject(data_dir, n)
        for n in range(1, 10)
    }


if __name__ == "__main__":
    # Quick manual smoke test -- adjust path to your local download.
    data_dir = Path("BCICIV_2b_gdf")
    subj = load_subject(data_dir, subject_num=1)
    print(f"Loaded {subj.subject_id}")
    print(f"Sessions: {subj.session_ids}")
    print(f"Sampling rate: {subj.fs} Hz")
    print(f"Channels: {subj.channel_names}")
    for sid in subj.session_ids:
        n_events = len(subj.labels_by_session[sid])
        print(f"  Session {sid}: {n_events} left/right-hand trials")