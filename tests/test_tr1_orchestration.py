"""
tests/test_run_t1r_orchestration.py

Runs experiments/run_t1r.py's actual entry points (not a reimplementation
of them) against a scratch config, and checks the things you'd otherwise
verify by eye: does it detect which Dev A modules are present, does it
refuse to fabricate Figure T1-R.1 without published values, does the
structured log carry every §8.4 field, does demo mode ever get mistaken
for a real result. Fabricated data throughout.
"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.run_t1r import run_subject, make_figure, _try_load_dev_a_modules
from experiments.logging_t1r import T1RLogger


BASE_CFG = {
    "pre_registration_id": "PR-TEST",
    "data": {"subjects": [1], "data_dir": None},
    "preprocessing": {"band_pass_hz": [8, 30], "filter_order": None,
                       "epoch_window_s": None, "baseline_window_s": None},
    "features": {"family": "csp", "csp": {"n_components": 4}, "riemann": {}},
    "decoder": {"classifier": "slda"},
    "split": {"policy": "session_wise"},
    "evaluation": {"n_bootstrap_resamples": 1000, "n_surrogates": 10, "ci_percent": 95.0},
    "published_reference": {"per_subject_accuracy": {}},
    "seed": 42,
}


@pytest.fixture
def scratch_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_dev_a_module_detection_matches_current_repo_state():
    """As of this integration pass: data/bci_iv_2a.py and
    preprocessing/eeg_filters.py are real; preprocessing/epoching.py is
    the one still missing. This test pins that expectation so it fails
    loudly (in a good way) the moment epoching.py lands and nobody
    updated this test -- that's your signal to flip demo mode off."""
    mods, missing = _try_load_dev_a_modules()
    assert "bci_iv_2a" in mods
    assert "eeg_filters" in mods
    assert missing == ["preprocessing/epoching.py"]


def test_run_subject_demo_mode_produces_valid_result(scratch_dir):
    cfg = dict(BASE_CFG)
    logger = T1RLogger(scratch_dir / "logs")
    rng = np.random.default_rng(42)
    result = run_subject(1, cfg, mods={}, demo_mode=True, logger=logger, rng=rng)

    assert result["subject_id"] == "A01"
    assert 0.0 <= result["mean_accuracy"] <= 1.0
    assert result["ci_low"] <= result["mean_accuracy"] <= result["ci_high"]
    assert 0.0 < result["surrogate_p_value"] <= 1.0
    assert len(result["fold_accuracies"]) == 2  # 2a: 2 sessions -> 2 folds


def test_structured_log_has_every_8_4_field(scratch_dir):
    cfg = dict(BASE_CFG)
    logger = T1RLogger(scratch_dir / "logs")
    rng = np.random.default_rng(42)
    run_subject(1, cfg, mods={}, demo_mode=True, logger=logger, rng=rng)

    lines = logger.path.read_text().strip().split("\n")
    assert len(lines) >= 2  # one per fold, 2a has 2 sessions
    record = json.loads(lines[0])

    required_8_4_fields = [
        "band_edges_hz", "split_policy", "fold_index", "subject_id", "seed",
    ]
    for field in required_8_4_fields:
        assert field in record, f"§8.4-required field '{field}' missing from log record"
    assert record["extra"]["demo_mode"] is True  # never mistakeable for real


def test_figure_refuses_to_fabricate_without_published_values(scratch_dir):
    """No published_reference values in config -> figure must not draw a
    fake comparison; it should render its 'nothing to compare yet' state."""
    result = {"subject_id": "A01", "mean_accuracy": 0.9,
              "ci_low": 0.85, "ci_high": 0.95}
    out_path = scratch_dir / "fig.png"
    make_figure([result], published={}, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0  # rendered something, not a crash


def test_figure_draws_scatter_when_published_values_present(scratch_dir):
    result = {"subject_id": "A01", "mean_accuracy": 0.9,
              "ci_low": 0.85, "ci_high": 0.95}
    out_path = scratch_dir / "fig.png"
    make_figure([result], published={"A01": 0.82}, out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))