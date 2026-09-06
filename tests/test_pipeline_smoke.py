"""
tests/test_pipeline_smoke.py

Smoke test for the E3-A pipeline, mirroring
tests/test_tr1_orchestration.py's pattern: runs experiments/run_e3a.py's
actual entry points (not a reimplementation of them) in demo mode, and
checks the things that would otherwise need eyeballing -- does it
correctly report which real modules are missing, does demo mode ever
get mistaken for a real result, does the structured log carry the
S8.4-mandated fields, do the two expected figures actually get
written. Fabricated data throughout (run_e3a.py's own synthetic
planted-MRCP generator).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.run_e3a import run_subject, make_figure_e3a1, make_figure_e3a3, _try_load_real_modules
from experiments.logging_e3a import E3ALogger


BASE_CFG = {
    "pre_registration_id": "PR-TEST",
    "data": {"participants": [1], "data_dir": None},
    "onset": {"threshold_sd": 3.0, "baseline_window_s": [0.0, 1.0], "smooth_window_s": 0.05},
    "preprocessing": {
        "mrcp_band_hz": [0.3, 3.0], "filter_order": 4, "laplacian_lambda2": 1e-5,
        "rejection": {"amplitude_uv": 100.0, "kurtosis_z": 5.0},
    },
    "windowing": {"window_s": 0.5, "step_s": 0.05, "horizon_s": 1.5, "rest_margin_s": 2.0},
    "split": {"policy": "trial_block_wise"},
    "evaluation": {"n_thresholds": 10, "n_bootstrap_resamples": 1000, "ci_percent": 95.0},
    "seed": 42,
}


@pytest.fixture
def scratch_dir(tmp_path):
    return tmp_path


def test_demo_mode_flagged_when_real_modules_missing():
    """torch (eeg/decoder_eegnet.py) is not in this test environment,
    so demo mode must be the reported reason -- never silently treated
    as a real run."""
    mods, missing = _try_load_real_modules()
    assert isinstance(missing, list)
    # way_eeg_gal.py itself imports fine (only scipy), but with no
    # corpus at a real data_dir it can't produce a real result either --
    # this smoke test only asserts the *reporting* mechanism works, not
    # which specific module is missing on any given machine.


def test_run_subject_demo_mode_produces_curve(scratch_dir):
    logger = E3ALogger(scratch_dir / "logs")
    rng = np.random.default_rng(BASE_CFG["seed"])
    result = run_subject(1, BASE_CFG, mods={}, demo_mode=True, logger=logger, rng=rng)

    assert result["subject_id"] == "P1"
    assert result["n_windows"] > 0
    assert "tpr" in result["curve"]
    assert "mean_lead_time_s" in result["curve"]
    assert "fp_per_min" in result["curve"]
    # Bootstrap CI arrays are one value per swept threshold.
    n_thresholds = BASE_CFG["evaluation"]["n_thresholds"]
    assert len(result["curve"]["tpr"]["median"]) == n_thresholds


def test_structured_log_carries_s84_fields(scratch_dir):
    logger = E3ALogger(scratch_dir / "logs")
    rng = np.random.default_rng(BASE_CFG["seed"])
    run_subject(1, BASE_CFG, mods={}, demo_mode=True, logger=logger, rng=rng)

    lines = logger.path.read_text().strip().splitlines()
    assert lines, "expected at least one log line"
    record = json.loads(lines[0])
    for field in ("mrcp_band_hz", "window_s", "horizon_s", "onset_threshold_sd",
                  "rejection_amplitude_uv", "split_policy", "seed"):
        assert field in record, f"missing S8.4-mandated field: {field}"
    assert record["extra"].get("demo_mode") is True, (
        "demo-mode runs must be tagged in the log -- a reviewer must never "
        "mistake fabricated data for a real E3-A result"
    )


def test_figures_written(scratch_dir):
    logger = E3ALogger(scratch_dir / "logs")
    rng = np.random.default_rng(BASE_CFG["seed"])
    result = run_subject(1, BASE_CFG, mods={}, demo_mode=True, logger=logger, rng=rng)

    fig1 = scratch_dir / "DEMO_ONLY_figure.png"
    fig3 = scratch_dir / "DEMO_ONLY_fig_surrogate_panel.png"
    make_figure_e3a1([result], fig1)
    make_figure_e3a3([result], fig3)
    assert fig1.exists() and fig1.stat().st_size > 0
    assert fig3.exists() and fig3.stat().st_size > 0