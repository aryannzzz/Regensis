from __future__ import annotations

import pytest
from dataclasses import replace

from regenesis_cmi.information.gaussian import (
    gaussian_benchmark_covariance,
    gaussian_cmi_from_covariance,
    gaussian_cmi_from_partial_correlation,
    gaussian_cmi_monte_carlo,
    partial_correlation,
)
from regenesis_cmi.synthetic.causal_generator import build_oracle_model
from regenesis_cmi.synthetic.oracle import monte_carlo_cmi


def test_gaussian_partial_correlation_and_logdet_agree() -> None:
    covariance = gaussian_benchmark_covariance()
    partial = partial_correlation(covariance[0, 1], covariance[0, 2], covariance[1, 2])
    scalar = gaussian_cmi_from_partial_correlation(partial)
    general = gaussian_cmi_from_covariance(covariance, [0], [1], [2])
    assert general == pytest.approx(scalar, abs=1e-12)


def test_gaussian_monte_carlo_oracle_matches_analytic() -> None:
    covariance = gaussian_benchmark_covariance()
    truth = gaussian_cmi_from_covariance(covariance, [0], [1], [2])
    estimate = gaussian_cmi_monte_carlo(covariance, n_samples=80_000, seed=71)
    tolerance = max(0.012, 4 * float(estimate["standard_error_bits"]))
    assert float(estimate["estimate_bits"]) == pytest.approx(truth, abs=tolerance)


def test_categorical_gaussian_oracle_reports_precision(small_config) -> None:
    model = build_oracle_model(small_config, "genuine_cortical", "post", seed=7)
    estimate = monte_carlo_cmi(
        model,
        seed=9,
        precision_bits=0.03,
        min_samples=4_000,
        max_samples=12_000,
        batch_size=4_000,
    )
    assert estimate.standard_error_bits <= 0.03
    assert estimate.precision_achieved
    assert estimate.sample_size >= 4_000
    assert estimate.direction == "I(G;E|M)"


def test_stress_oracle_can_report_one_subject_without_mixing_keys(small_config) -> None:
    stress = replace(
        small_config, mode="stress", subject_variability_scale=0.12
    )
    model = build_oracle_model(stress, "genuine_cortical", "post", seed=7)
    subject_keys = tuple(key for key in model.keys if key.startswith("subject_00_"))
    estimate = monte_carlo_cmi(
        model,
        seed=9,
        precision_bits=1.0,
        min_samples=2_000,
        max_samples=2_000,
        oracle_keys=subject_keys,
    )
    assert set(estimate.diagnostics["oracle_keys"]) == set(subject_keys)
    assert all(key.startswith("subject_00_") for key in estimate.diagnostics["oracle_keys"])
