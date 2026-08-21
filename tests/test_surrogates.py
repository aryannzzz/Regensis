from __future__ import annotations

import numpy as np

from regenesis_cmi.information.mixed import estimate_mixed_cmi
from regenesis_cmi.synthetic.causal_generator import generate_causal_dataset
from regenesis_cmi.synthetic.schema import GeneratorConfig
from regenesis_cmi.synthetic.surrogates import (
    discriminating_cortical_surrogate,
    grouped_label_shuffle,
)


def _large_config() -> GeneratorConfig:
    return GeneratorConfig(
        n_subjects=1,
        trials_per_subject=512,
        blocks_per_subject=8,
        eeg_channels=6,
        emg_channels=5,
    )


def estimate(data, eeg, labels=None, seed=1) -> float:
    return estimate_mixed_cmi(
        data.g if labels is None else labels,
        eeg,
        data.emg,
        direction="I(G;E|M)",
        seed=seed,
    ).estimate_bits


def test_discriminating_surrogate_preserves_non_cortical_components() -> None:
    data = generate_causal_dataset(_large_config(), "genuine_cortical", "post", seed=101)
    g_before, m_before, c_before = data.g.copy(), data.emg.copy(), data.cranial.copy()
    surrogate = discriminating_cortical_surrogate(data, seed=9)
    assert np.array_equal(data.g, g_before)
    assert np.array_equal(data.emg, m_before)
    assert np.array_equal(data.cranial, c_before)
    assert not np.array_equal(surrogate, data.eeg)


def test_surrogate_preserves_artifact_but_removes_cortical_target_structure() -> None:
    config = _large_config()
    artifact = generate_causal_dataset(config, "artifact_only", "post", seed=101)
    genuine = generate_causal_dataset(config, "genuine_cortical", "post", seed=101)
    artifact_surrogate = discriminating_cortical_surrogate(artifact, seed=9)
    genuine_surrogate = discriminating_cortical_surrogate(genuine, seed=9)
    artifact_raw = estimate(artifact, artifact.eeg)
    artifact_intervened = estimate(artifact, artifact_surrogate, seed=2)
    genuine_raw = estimate(genuine, genuine.eeg)
    genuine_intervened = estimate(genuine, genuine_surrogate, seed=2)
    assert abs(artifact_raw - artifact_intervened) < 0.25
    assert genuine_raw > genuine_intervened + 0.4


def test_label_shuffle_can_make_artifact_only_effect_look_significant() -> None:
    data = generate_causal_dataset(_large_config(), "artifact_only", "post", seed=101)
    observed = estimate(data, data.eeg)
    null = []
    for permutation in range(9):
        labels = grouped_label_shuffle(
            data.g, data.metadata, seed=100 + permutation
        )
        null.append(estimate(data, data.eeg, labels=labels, seed=permutation))
    assert observed > np.quantile(null, 0.95)

