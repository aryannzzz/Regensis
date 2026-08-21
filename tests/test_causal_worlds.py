from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from regenesis_cmi.synthetic.causal_generator import build_oracle_model, generate_causal_dataset
from regenesis_cmi.synthetic.oracle import monte_carlo_cmi


def oracle(model, **kwargs) -> float:
    return monte_carlo_cmi(
        model,
        seed=11,
        precision_bits=1.0,
        min_samples=8_000,
        max_samples=8_000,
        batch_size=4_000,
        **kwargs,
    ).estimate_bits


@pytest.mark.parametrize("world", ["null", "redundant"])
def test_null_and_exact_redundant_worlds_are_conditionally_null(small_config, world: str) -> None:
    model = build_oracle_model(small_config, world, "post", seed=42)
    assert abs(oracle(model)) < 0.02


def test_all_five_scenarios_and_components_exist(small_config) -> None:
    worlds = ["null", "redundant", "genuine_cortical", "artifact_only", "cortical_plus_artifact"]
    for world in worlds:
        data = generate_causal_dataset(small_config, world, "post", seed=3)
        assert set(np.unique(data.g)) == {0, 1, 2, 3}
        assert data.eeg.shape[1] == small_config.eeg_dimensions
        assert data.emg.shape[1] == small_config.emg_channels
        assert data.cranial.shape[1] == small_config.cranial_dimensions
        assert {"central", "peripheral_frontal_temporal"} == set(
            data.eeg_feature_metadata["channel_group"]
        )
        assert set(small_config.band_labels) == set(data.eeg_feature_metadata["band"])


def test_pre_and_post_are_never_pooled(small_config) -> None:
    pre = generate_causal_dataset(small_config, "genuine_cortical", "pre", seed=4)
    post = generate_causal_dataset(small_config, "genuine_cortical", "post", seed=4)
    assert set(pre.metadata["window"]) == {"pre"}
    assert set(post.metadata["window"]) == {"post"}
    assert set(pre.metadata["window_id"]) == {0}
    assert set(post.metadata["window_id"]) == {1}
    assert oracle(pre.oracle_model) > 0.2
    assert oracle(post.oracle_model) > 0.1


def test_spatial_and_band_profiles_make_artifact_diagnosable() -> None:
    from regenesis_cmi.synthetic.schema import GeneratorConfig

    config = GeneratorConfig(
        n_subjects=1,
        trials_per_subject=64,
        blocks_per_subject=4,
        eeg_channels=6,
        emg_channels=3,
    )
    data = generate_causal_dataset(config, "artifact_only", "post", seed=42)
    full = oracle(data.oracle_model, eeg_indices=data.eeg_indices())
    central = oracle(data.oracle_model, eeg_indices=data.eeg_indices(montage="central"))
    high = oracle(data.oracle_model, eeg_indices=data.eeg_indices(band="high"))
    low = oracle(data.oracle_model, eeg_indices=data.eeg_indices(band="low"))
    assert full > central + 0.1
    assert high > low + 0.1


def test_effect_strengths_are_configuration_overrides(small_config) -> None:
    zero_effect = replace(
        small_config, cortical_strength=0.0, shared_eeg_strength=0.0
    )
    model = build_oracle_model(zero_effect, "genuine_cortical", "post", seed=42)
    assert abs(oracle(model)) < 0.02
    assert model.parameter_manifest["scenario_strengths"]["cortical"] == 0.0
