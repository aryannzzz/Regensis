from __future__ import annotations

import numpy as np

from regenesis_cmi.synthetic.causal_generator import generate_causal_dataset
from regenesis_cmi.synthetic.degradation import degrade
from regenesis_cmi.synthetic.oracle import monte_carlo_cmi


def oracle(model, **kwargs) -> float:
    return monte_carlo_cmi(
        model,
        seed=23,
        precision_bits=1.0,
        min_samples=8_000,
        max_samples=8_000,
        **kwargs,
    ).estimate_bits


def test_emg_degradation_scales_signal_not_observed_m(small_config) -> None:
    data = generate_causal_dataset(small_config, "artifact_only", "post", seed=8)
    half = degrade(data, 0.5)
    zero = degrade(data, 0.0)
    assert np.allclose(zero.emg, data.emg_noise)
    assert not np.allclose(half.emg, 0.5 * data.emg)
    assert half.realized_snr_db < degrade(data, 1.0).realized_snr_db


def test_degradation_reduces_emg_information(small_config) -> None:
    data = generate_causal_dataset(small_config, "genuine_cortical", "post", seed=9)
    healthy = oracle(data.oracle_model, direction="I(G;M|E)", alpha=1.0)
    absent = oracle(data.oracle_model, direction="I(G;M|E)", alpha=0.0)
    assert healthy > absent + 0.1


def test_matched_myogenic_degradation_reduces_artifact_rise(small_config) -> None:
    data = generate_causal_dataset(small_config, "artifact_only", "post", seed=42)
    raw_healthy = oracle(data.oracle_model, alpha=1.0)
    raw_absent = oracle(data.oracle_model, alpha=0.0)
    matched_absent = oracle(data.oracle_model, alpha=0.0, eeg_variant="matched")
    assert raw_absent > raw_healthy + 0.1
    assert matched_absent < raw_absent - 0.1

