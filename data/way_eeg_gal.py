"""
data/way_eeg_gal.py

Loader for WAY-EEG-GAL (Luciw, Jarocka & Edin, Scientific Data 2014,
doi:10.1038/sdata.2014.47; corpus on figshare, doi:10.6084/m9.figshare.988376).
Per prereg/PR-2026-02.md and the E3-A protocol doc: "Only available
dataset with synchronized EEG and EMG."

Confirmed from the data descriptor (S6.1 -- corpus facts, verified
against the publication, not assumed):
  - 12 participants (P1-P12), 10 series each, ~328 trials/participant,
    3936 trials total.
  - 32 EEG channels, 5 EMG channels (arm/hand muscles).
  - Two MATLAB file types per participant:
      PX_AllLifts.mat  -- per-lift windowed/holistic trial-info structure
                           `P.AllLifts` (weight/surface conditions, 16
                           event times, 18 behavioural measures per trial).
      HS_PX_SY.mat      -- per-series CONTINUOUS recording, e.g.
                           HS_P3_S2.mat = participant 3, series 2. This
                           is the file this loader reads: E3-A needs
                           continuous EEG+EMG to epoch relative to
                           EMG threshold-onset, not the pre-cut lift
                           windows in *_AllLifts.mat.
  - EEG and EMG were recorded on separate acquisition systems (SC/ZOOM
    and BCI2000) and synchronized post-hoc by the dataset authors via
    cross-correlation of two sync signals (max lag search: 5000
    samples). That is a coarse, once-off alignment done at corpus-prep
    time, not a guarantee of zero residual offset for any given
    series -- which is exactly why the E3-A protocol doc requires
    *this loader's caller* to independently measure the EEG/EMG timing
    gap (see measure_sync_gap below) rather than trust it (S6.1 common
    mistake #2: "trusting synchronisation instead of measuring it").
  - Only preprocessing applied by the dataset authors: EMG mean-removed.
    No artifact rejection. EEG is raw.

NOT YET CONFIRMED against a real downloaded file (no corpus file was
available while writing this loader -- S6 requires every corpus to be
"downloaded, opened, recounted and verified against its publication
before any modelling," which has not happened yet for WAY-EEG-GAL).
Internal MATLAB struct field names below (EEG_FIELD_CANDIDATES etc.)
are best-effort from public descriptions of the format and are NOT
guaranteed to match the real file's exact capitalization/nesting.
_pick_field() is written to fail loudly with the real available field
names if a guess is wrong, rather than silently reading the wrong
array -- but before this loader is used for any real E3-A result,
someone must run it against a downloaded HS_*.mat file, fix any field
name mismatch it reports, and record that verification in
data/cards/way_eeg_gal.md (this file's card is currently also a stub
pending that same step -- see TODO there).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from scipy.io import loadmat
except ImportError as e:
    raise ImportError("This loader requires scipy. Install with: pip install scipy") from e


N_PARTICIPANTS = 12
N_SERIES_PER_PARTICIPANT = 10
N_EEG_CHANNELS = 32
N_EMG_CHANNELS = 5

# Best-effort candidate field names for the continuous HS_PX_SY.mat
# struct -- see module docstring. First match wins; _pick_field raises
# with the real key list if none match, so a wrong guess here is a
# loud KeyError on first real use, not a silent misread.
EEG_DATA_FIELD_CANDIDATES = ["eeg", "EEG"]
EMG_DATA_FIELD_CANDIDATES = ["emg", "EMG"]
EEG_SFREQ_FIELD_CANDIDATES = ["sr", "srate", "fs", "sf"]
EMG_SFREQ_FIELD_CANDIDATES = ["sr", "srate", "fs", "sf"]
EEG_CHANNEL_NAME_FIELD_CANDIDATES = ["chan_names", "chanNames", "channels"]


def _pick_field(struct, candidates: list[str], context: str):
    available = [f for f in (struct._fieldnames if hasattr(struct, "_fieldnames") else [])]
    for name in candidates:
        if hasattr(struct, name):
            return getattr(struct, name)
    raise KeyError(
        f"None of the candidate field names {candidates} found on the "
        f"{context} struct. Available fields: {available or '(unknown -- '
        f'struct introspection failed, inspect the .mat file directly)'}. "
        f"This loader's field-name guesses (module docstring) have not "
        f"been confirmed against a real file -- fix the candidate list "
        f"above once you can see the real struct, then record the "
        f"confirmed names in data/cards/way_eeg_gal.md."
    )


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_hash(data_dir: Path) -> str:
    """Per S8.4/S6: dataset version hash, recorded before any modelling."""
    all_files = sorted(Path(data_dir).glob("HS_P*_S*.mat")) + sorted(Path(data_dir).glob("P*_AllLifts.mat"))
    if not all_files:
        raise FileNotFoundError(f"No HS_P*_S*.mat or P*_AllLifts.mat files found in {data_dir}")
    combined = hashlib.sha256()
    for f in all_files:
        combined.update(_sha256_of_file(f).encode())
    return combined.hexdigest()


@dataclass
class WayEEGGALSeries:
    participant_id: str          # e.g. "P3"
    series_id: int                # e.g. 2
    eeg: np.ndarray                # (n_eeg_channels, n_eeg_samples)
    emg: np.ndarray                # (n_emg_channels, n_emg_samples)
    eeg_fs: float
    emg_fs: float
    eeg_channel_names: list = field(default_factory=list)


def load_series(data_dir: Path, participant_num: int, series_num: int) -> WayEEGGALSeries:
    """
    Load one continuous series (HS_P<participant_num>_S<series_num>.mat).

    Args:
        data_dir: folder containing the HS_*.mat files.
        participant_num: 1-12.
        series_num: 1-10.

    Returns:
        WayEEGGALSeries with raw (unfiltered, unsynchronized-beyond-the-
        corpus's-own-cross-correlation-step) EEG and EMG arrays. Caller
        (data/way_eeg_gal.py's measure_sync_gap, then
        eeg/anticipation.py) is responsible for measuring the residual
        EEG/EMG timing gap before treating the two as aligned.
    """
    participant_id = f"P{participant_num}"
    path = Path(data_dir) / f"HS_{participant_id}_S{series_num}.mat"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {path}. Download source: figshare, "
            f"doi:10.6084/m9.figshare.988376 (see data/cards/way_eeg_gal.md "
            f"for the exact resolved URL and access date once recorded)."
        )

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    top_keys = [k for k in mat if not k.startswith("__")]
    if len(top_keys) != 1:
        raise KeyError(
            f"Expected exactly one top-level struct in {path}, found: "
            f"{top_keys}. Loader assumption (single per-series struct, "
            f"conventionally named 'hs' or similar) needs updating -- "
            f"inspect the file directly."
        )
    root = getattr(mat[top_keys[0]], "__dict__", None) or mat[top_keys[0]]

    eeg_struct = _pick_field(root, EEG_DATA_FIELD_CANDIDATES, "root")
    emg_struct = _pick_field(root, EMG_DATA_FIELD_CANDIDATES, "root")

    eeg_data = np.asarray(_pick_field(eeg_struct, ["sig", "data", "d"], "eeg"), dtype=float)
    emg_data = np.asarray(_pick_field(emg_struct, ["sig", "data", "d"], "emg"), dtype=float)
    # Orient as (n_channels, n_samples) -- fewer channels than samples
    # in essentially any real recording, so the shorter axis is channels.
    if eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T
    if emg_data.shape[0] > emg_data.shape[1]:
        emg_data = emg_data.T

    eeg_fs = float(_pick_field(eeg_struct, EEG_SFREQ_FIELD_CANDIDATES, "eeg"))
    emg_fs = float(_pick_field(emg_struct, EMG_SFREQ_FIELD_CANDIDATES, "emg"))

    try:
        eeg_channel_names = list(_pick_field(eeg_struct, EEG_CHANNEL_NAME_FIELD_CANDIDATES, "eeg"))
    except KeyError:
        eeg_channel_names = [f"EEG{i}" for i in range(eeg_data.shape[0])]

    return WayEEGGALSeries(
        participant_id=participant_id,
        series_id=series_num,
        eeg=eeg_data,
        emg=emg_data,
        eeg_fs=eeg_fs,
        emg_fs=emg_fs,
        eeg_channel_names=eeg_channel_names,
    )


def measure_sync_gap(series: WayEEGGALSeries, max_lag_s: float = 0.5) -> dict:
    """
    Measure (not assume) the EEG-EMG timing offset for one series, per
    the E3-A protocol doc's pipeline step 1: "Measure the actual timing
    gap between EEG and EMG recordings -- do not assume they are
    aligned," and the manual's S6.1 known-issue: synchronisation
    between modalities must be measured.

    Cross-correlates the EEG-channel-average envelope against the
    EMG-channel-average envelope (both amplitude-rectified and
    resampled to a common low rate) and reports the lag at peak
    correlation. This is diagnostic, not a resync step -- it exists so
    the residual gap is a logged, reported number (S8.4), not a
    silent assumption. If the reported gap is large relative to the
    corpus's own claimed cross-correlation-based alignment (module
    docstring), treat that as a per-series data-quality flag, not
    something to silently correct away.

    Args:
        series: from load_series().
        max_lag_s: search window for the lag, seconds.

    Returns:
        dict with measured_lag_s (positive = EMG leads EEG),
        peak_correlation, and the common resample rate used.
    """
    common_fs = 100.0  # coarse rate is fine for a diagnostic envelope check
    eeg_env = np.abs(series.eeg).mean(axis=0)
    emg_env = np.abs(series.emg).mean(axis=0)

    def _resample(x, orig_fs, target_fs):
        n_target = int(round(len(x) * target_fs / orig_fs))
        return np.interp(
            np.linspace(0, len(x) - 1, n_target),
            np.arange(len(x)), x,
        )

    eeg_r = _resample(eeg_env, series.eeg_fs, common_fs)
    emg_r = _resample(emg_env, series.emg_fs, common_fs)
    eeg_r = (eeg_r - eeg_r.mean()) / (eeg_r.std() + 1e-12)
    emg_r = (emg_r - emg_r.mean()) / (emg_r.std() + 1e-12)

    max_lag_samples = int(round(max_lag_s * common_fs))
    corr = np.correlate(eeg_r, emg_r, mode="full")
    lags = np.arange(-len(emg_r) + 1, len(eeg_r))
    center = len(corr) // 2
    window = slice(max(0, center - max_lag_samples), min(len(corr), center + max_lag_samples + 1))

    windowed_corr = corr[window]
    windowed_lags = lags[window]
    peak_idx = int(np.argmax(windowed_corr))

    return {
        "measured_lag_s": float(windowed_lags[peak_idx] / common_fs),
        "peak_correlation": float(windowed_corr[peak_idx] / len(eeg_r)),
        "resample_fs": common_fs,
        "search_window_s": max_lag_s,
    }


if __name__ == "__main__":
    import sys
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/way_eeg_gal")
    s = load_series(data_dir, participant_num=1, series_num=1)
    print(f"Loaded {s.participant_id} series {s.series_id}")
    print(f"EEG: {s.eeg.shape} @ {s.eeg_fs} Hz, EMG: {s.emg.shape} @ {s.emg_fs} Hz")
    gap = measure_sync_gap(s)
    print(f"Measured EEG/EMG sync gap: {gap['measured_lag_s']*1000:.1f} ms "
          f"(peak corr={gap['peak_correlation']:.3f})")