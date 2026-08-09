"""
experiments/logging_t1r.py — structured logging per §8.4.

"A parameter that is not logged does not exist." Every field the manual
names explicitly gets its own logged record, written as JSON lines to
<out_dir>/t1r_run_<timestamp>.jsonl so a reviewer can grep exactly which
band edges / epoch window / split / seed produced a given figure without
reading code.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunRecord:
    """One line per stage. Every §8.4-mandated field is a named attribute
    (not a free-form dict) so a missing field is a loud KeyError/TypeError
    at construction time, not a silent gap in the log."""
    stage: str
    timestamp: float = field(default_factory=time.time)

    # Dataset
    dataset_version_hash: str | None = None
    data_access_date: str | None = None

    # Preprocessing
    band_edges_hz: list | None = None
    filter_order: int | None = None
    epoch_window_s: list | None = None
    baseline_window_s: list | None = None

    # Split
    split_policy: str | None = None
    fold_index: int | None = None
    subject_id: str | None = None
    held_out_session_or_block: str | None = None

    # Model / search
    hyperparam_space: dict | None = None
    hyperparam_budget: int | None = None
    selected_hyperparams: dict | None = None

    # Reproducibility
    seed: int | None = None
    git_commit: str | None = None

    # Timing
    runtime_seconds: float | None = None

    # Free-form extras for anything stage-specific (kept last, never a
    # substitute for the named fields above)
    extra: dict = field(default_factory=dict)


class T1RLogger:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        self.path = self.out_dir / f"t1r_run_{ts}.jsonl"

    def log(self, record: RunRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")

    def log_missing_param_error(self, stage: str, param_name: str) -> None:
        """Per §8.4: 'a parameter that is not logged does not exist.' This
        is the enforcement mechanism — refuse to proceed rather than run
        with a silently-defaulted value that would make a figure
        withdrawable on review."""
        raise ValueError(
            f"[{stage}] Required parameter '{param_name}' is unset "
            f"(null in config). Per §8.4 this parameter must be logged "
            f"explicitly before the run proceeds — fill it from the "
            f"signed pre-registration, do not default it silently."
        )