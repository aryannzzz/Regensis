from __future__ import annotations

import numpy as np

from regenesis_cmi.information.decoder import estimate_decoder_cmi
from regenesis_cmi.synthetic.causal_generator import generate_causal_dataset


def test_same_seed_reproduces_samples_parameters_and_summary(small_config) -> None:
    first = generate_causal_dataset(small_config, "cortical_plus_artifact", "post", seed=77)
    second = generate_causal_dataset(small_config, "cortical_plus_artifact", "post", seed=77)
    assert np.array_equal(first.g, second.g)
    assert np.array_equal(first.eeg, second.eeg)
    assert np.array_equal(first.emg, second.emg)
    assert first.oracle_model.fingerprint() == second.oracle_model.fingerprint()
    assert first.parameters == second.parameters


def test_different_seed_changes_samples_not_configuration(small_config) -> None:
    first = generate_causal_dataset(small_config, "genuine_cortical", "post", seed=1)
    second = generate_causal_dataset(small_config, "genuine_cortical", "post", seed=2)
    assert not np.array_equal(first.eeg, second.eeg)
    assert first.parameters["config"] == second.parameters["config"]
    assert first.scenario == second.scenario
    assert first.window == second.window


def test_negative_decoder_estimate_is_preserved(small_config) -> None:
    data = generate_causal_dataset(small_config, "null", "post", seed=1)
    result = estimate_decoder_cmi(
        data.g,
        data.eeg,
        data.emg,
        groups=data.metadata["block_uid"],
        direction="I(G;E|M)",
        seed=1,
        c_grid=(0.1, 1.0),
    )
    assert result.estimate_bits < 0
    assert "negative_decoder_estimate_preserved" in result.warnings

