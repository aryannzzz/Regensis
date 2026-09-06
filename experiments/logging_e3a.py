"""
experiments/logging_e3a.py — structured logging for E3-A, per S8.4.

Mirrors experiments/logging_t1r.py's pattern (named dataclass fields,
not a free-form dict, so a missing S8.4-mandated field is a loud
construction-time error) but with E3-A's own fields: onset-detection
parameters, sliding-window/horizon config, detector family, threshold,
and the three controls' pass/fail status -- none of which T1RLogger's
RunRecord has a slot for.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class E3ARunRecord:
    stage: str
    timestamp: float = field(default_factory=time.time)

    # Dataset
    dataset_version_hash: str | None = None
    data_access_date: str | None = None
    measured_sync_gap_s: float | None = None   # from data.way_eeg_gal.measure_sync_gap

    # Onset marking
    onset_threshold_sd: float | None = None
    onset_baseline_window_s: list | None = None
    n_hand_verified_trials: int | None = None  # protocol doc: "at least 30 trials"

    # Preprocessing
    mrcp_band_hz: list | None = None
    filter_order: int | None = None
    laplacian_lambda2: float | None = None
    rejection_amplitude_uv: float | None = None
    rejection_kurtosis_z: float | None = None
    n_trials_rejected: int | None = None
    n_trials_total: int | None = None

    # Windowing
    window_s: float | None = None
    step_s: float | None = None
    horizon_s: float | None = None
    rest_margin_s: float | None = None

    # Detector / model / search
    detector_family: str | None = None    # "slda" | "eegnet"
    hyperparam_space: dict | None = None
    hyperparam_budget: int | None = None
    selected_hyperparams: dict | None = None

    # Split
    split_policy: str | None = None
    fold_index: int | None = None
    subject_id: str | None = None

    # Controls
    time_reversed_control_passed: bool | None = None
    shuffled_onset_control_passed: bool | None = None
    pre_baseline_shift_control_passed: bool | None = None

    # Reproducibility
    seed: int | None = None
    git_commit: str | None = None

    # Timing
    runtime_seconds: float | None = None

    extra: dict = field(default_factory=dict)


class E3ALogger:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        self.path = self.out_dir / f"e3a_run_{ts}.jsonl"

    def log(self, record: E3ARunRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")

    def log_missing_param_error(self, stage: str, param_name: str) -> None:
        raise ValueError(
            f"[{stage}] Required parameter '{param_name}' is unset "
            f"(null in config). Per S8.4 this parameter must be logged "
            f"explicitly before the run proceeds -- fill it from the "
            f"signed pre-registration, do not default it silently."
        )