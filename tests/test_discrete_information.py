from __future__ import annotations

import numpy as np
import pytest

from regenesis_cmi.information.discrete import (
    cmi_identities,
    conditional_entropy,
    conditional_mutual_information,
    entropy,
    joint_entropy,
)
from regenesis_cmi.synthetic.exact_worlds import complementary_exact
from regenesis_cmi.synthetic.schema import GeneratorConfig


@pytest.mark.parametrize("cardinality,truth", [(1, 0.0), (2, 1.0), (4, 2.0), (8, 3.0)])
def test_balanced_entropy(cardinality: int, truth: float) -> None:
    values = np.tile(np.arange(cardinality), 20)
    assert entropy(values) == pytest.approx(truth, abs=1e-12)


def test_both_discrete_cmi_identities_agree() -> None:
    world = complementary_exact(10)
    identities = cmi_identities(world.g, world.eeg, world.emg)
    assert identities["four_entropy_bits"] == pytest.approx(1.0)
    assert identities["conditional_entropy_difference_bits"] == pytest.approx(1.0)
    assert identities["absolute_identity_gap_bits"] < 1e-12
    assert conditional_entropy(world.g, world.emg) == pytest.approx(1.0)
    assert conditional_entropy(world.g, np.column_stack((world.emg, world.eeg))) == 0.0


def test_joint_categorical_values_are_not_treated_as_continuous() -> None:
    x = np.array([[0, 10], [0, 10], [1, -500], [1, -500]])
    assert joint_entropy(x) == pytest.approx(1.0)


def test_constant_target_is_rejected_by_generator_configuration() -> None:
    config = GeneratorConfig(n_grasp_types=1)
    with pytest.raises(ValueError, match="at least two"):
        config.validate()

