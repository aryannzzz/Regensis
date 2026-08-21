from __future__ import annotations

import pytest

from regenesis_cmi.information.discrete import conditional_mutual_information, entropy, mutual_information
from regenesis_cmi.information.mixed import estimate_mixed_cmi
from regenesis_cmi.synthetic.exact_worlds import (
    asymmetric_exact,
    binary_entropy,
    complementary_exact,
    noisy_complementary_exact,
    redundant_markov_exact,
)
from regenesis_cmi.synthetic.oracle import discrete_pmf_cmi_oracle


def test_exact_complementary_world_is_one_bit() -> None:
    world = complementary_exact(20)
    assert entropy(world.g) == pytest.approx(2.0)
    assert conditional_mutual_information(world.g, world.eeg, world.emg) == pytest.approx(1.0)


@pytest.mark.parametrize("q", [0.00, 0.05, 0.11, 0.20, 0.30, 0.40, 0.50])
def test_noisy_complementary_matches_binary_channel(q: float) -> None:
    world = noisy_complementary_exact(q)
    estimate = conditional_mutual_information(world.g, world.eeg, world.emg)
    assert estimate == pytest.approx(1.0 - binary_entropy(q), abs=1e-12)


def test_redundant_markov_chain_has_mi_but_zero_cmi() -> None:
    world = redundant_markov_exact()
    assert mutual_information(world.g, world.eeg) > 0
    assert conditional_mutual_information(world.g, world.eeg, world.emg) == 0.0


def test_asymmetric_directions_are_calculated_independently() -> None:
    world = asymmetric_exact()
    assert conditional_mutual_information(world.g, world.eeg, world.emg) == pytest.approx(1.0)
    assert conditional_mutual_information(world.g, world.emg, world.eeg) == pytest.approx(2.0)


def test_discrete_likelihood_ratio_oracle_matches_exact_answer() -> None:
    world = noisy_complementary_exact(0.11)
    oracle = discrete_pmf_cmi_oracle(
        world.g, world.eeg, world.emg, n_samples=50_000, seed=22
    )
    tolerance = max(0.01, 4 * oracle.standard_error_bits)
    assert oracle.estimate_bits == pytest.approx(world.truth_forward_bits, abs=tolerance)


@pytest.mark.parametrize(
    "world_factory,truth",
    [(complementary_exact, 1.0), (redundant_markov_exact, 0.0)],
)
def test_published_mixed_estimator_recovers_discrete_generator_a(
    world_factory, truth: float
) -> None:
    world = world_factory()
    result = estimate_mixed_cmi(
        world.g,
        world.eeg,
        world.emg,
        direction="I(G;E|M)",
        added_discrete_mask=[True],
        conditioned_discrete_mask=[True],
    )
    assert result.estimate_bits == pytest.approx(truth, abs=1e-12)
