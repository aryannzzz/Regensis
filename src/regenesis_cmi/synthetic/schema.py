"""Typed configuration and data schema for Generator B."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .oracle import LinearGaussianOracle


SCENARIOS = (
    "null",
    "redundant",
    "genuine_cortical",
    "artifact_only",
    "cortical_plus_artifact",
)
WINDOWS = ("pre", "post")


@dataclass(frozen=True)
class GeneratorConfig:
    """Experiment-shaped defaults, with the target ontology isolated here."""

    target_name: str = "intended_grasp_type"
    n_grasp_types: int = 4
    n_subjects: int = 12
    trials_per_subject: int = 328
    blocks_per_subject: int = 8
    eeg_channels: int = 32
    emg_channels: int = 5
    band_labels: tuple[str, ...] = ("low", "mu", "beta", "high")
    cranial_dimensions: int = 2
    mode: str = "debug"
    mild_class_imbalance: float = 0.0
    block_drift_scale: float = 0.0
    subject_variability_scale: float = 0.0
    default_proxy_quality: float = 0.7
    pre_emg_fraction: float = 0.05
    pre_artifact_fraction: float = 0.10
    cortical_noise: float = 0.75
    emg_sensor_noise: float = 0.75
    eeg_sensor_noise: float = 0.80
    cranial_noise: float = 0.55
    # Optional named-world overrides support targeted zero/very-small/small/
    # medium/large effect matrices without changing causal code.
    cortical_strength: float | None = None
    artifact_strength: float | None = None
    shared_eeg_strength: float | None = None
    shared_motor_strength: float = 0.75
    emg_unique_strength: float = 1.15

    @property
    def eeg_dimensions(self) -> int:
        return self.eeg_channels * len(self.band_labels)

    def validate(self) -> None:
        if self.n_grasp_types < 2:
            raise ValueError("G must have at least two possible intended grasp types")
        if self.n_subjects < 1 or self.trials_per_subject < self.n_grasp_types:
            raise ValueError("The generator needs subjects and enough trials for all classes")
        if self.blocks_per_subject < 2:
            raise ValueError("At least two blocks are required for leakage-safe evaluation")
        if self.mode not in {"debug", "stress"}:
            raise ValueError("mode must be 'debug' or 'stress'")
        if not 0.0 <= self.default_proxy_quality <= 1.0:
            raise ValueError("Proxy quality must lie in [0, 1]")
        nonnegative = {
            "cortical_strength": self.cortical_strength,
            "artifact_strength": self.artifact_strength,
            "shared_eeg_strength": self.shared_eeg_strength,
            "shared_motor_strength": self.shared_motor_strength,
            "emg_unique_strength": self.emg_unique_strength,
        }
        if any(value is not None and value < 0 for value in nonnegative.values()):
            raise ValueError("Configured causal strengths must be nonnegative")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "GeneratorConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in mapping.items() if key in known}
        if "band_labels" in values:
            values["band_labels"] = tuple(values["band_labels"])
        config = cls(**values)
        config.validate()
        return config


@dataclass
class SyntheticDataset:
    """One independently generated window class for a causal world."""

    g: NDArray[np.int64]
    eeg: NDArray[np.float64]
    emg: NDArray[np.float64]
    emg_signal: NDArray[np.float64]
    emg_noise: NDArray[np.float64]
    cortical: NDArray[np.float64]
    cranial: NDArray[np.float64]
    cranial_signal: NDArray[np.float64]
    cranial_noise: NDArray[np.float64]
    eeg_noise: NDArray[np.float64]
    artifact_signal_eeg: NDArray[np.float64]
    artifact_noise_eeg: NDArray[np.float64]
    myogenic_proxy: NDArray[np.float64]
    metadata: pd.DataFrame
    eeg_feature_metadata: pd.DataFrame
    emg_projections: dict[str, NDArray[np.float64]]
    oracle_model: "LinearGaussianOracle"
    scenario: str
    window: str
    seed: int
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.g)
        sample_arrays = (
            self.eeg,
            self.emg,
            self.emg_signal,
            self.emg_noise,
            self.cortical,
            self.cranial,
            self.cranial_signal,
            self.cranial_noise,
            self.eeg_noise,
            self.artifact_signal_eeg,
            self.artifact_noise_eeg,
            self.myogenic_proxy,
        )
        if any(len(array) != n for array in sample_arrays) or len(self.metadata) != n:
            raise ValueError("Every dataset field must use the same sample axis")
        unique_targets = (
            self.g.nunique() if isinstance(self.g, pd.Series) else len(np.unique(self.g))
        )
        if unique_targets < 2:
            raise ValueError("Generated G is constant; the CMI experiment is meaningless")

    @property
    def eeg_clean(self) -> NDArray[np.float64]:
        """Ideal synthetic removal E - W_C C (not a real-data procedure)."""

        if self.scenario == "redundant":
            return self.eeg.copy()
        return self.cortical + self.eeg_noise

    def emg_representation(self, richness: str) -> NDArray[np.float64]:
        if richness not in self.emg_projections:
            raise KeyError(f"Unknown EMG representation: {richness}")
        return self.emg @ self.emg_projections[richness].T

    def eeg_indices(
        self, *, montage: str = "full", band: str = "all"
    ) -> NDArray[np.int64]:
        mask = np.ones(len(self.eeg_feature_metadata), dtype=bool)
        if montage == "central":
            mask &= self.eeg_feature_metadata["channel_group"].to_numpy() == "central"
        elif montage != "full":
            raise ValueError("montage must be 'full' or 'central'")
        if band != "all":
            if band not in set(self.eeg_feature_metadata["band"]):
                raise ValueError(f"Unknown band: {band}")
            mask &= self.eeg_feature_metadata["band"].to_numpy() == band
        return np.flatnonzero(mask).astype(np.int64)
