from __future__ import annotations

import pytest

from regenesis_cmi.synthetic.schema import GeneratorConfig


@pytest.fixture(scope="session")
def small_config() -> GeneratorConfig:
    return GeneratorConfig(
        n_subjects=2,
        trials_per_subject=64,
        blocks_per_subject=4,
        eeg_channels=4,
        emg_channels=3,
        mode="debug",
    )

