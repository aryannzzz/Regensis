"""Generic and origin-discriminating synthetic null interventions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .schema import SyntheticDataset


def grouped_label_shuffle(
    labels: NDArray[np.int64],
    metadata: pd.DataFrame,
    *,
    seed: int,
    group_column: str = "trial_uid",
    strata_columns: Sequence[str] = ("subject_id",),
) -> NDArray[np.int64]:
    """Shuffle whole trial labels within declared strata, never individual windows."""

    if len(labels) != len(metadata):
        raise ValueError("labels and metadata must have the same length")
    frame = metadata.copy()
    frame["_label"] = np.asarray(labels)
    unique_per_group = frame.groupby(group_column, sort=False)["_label"].nunique()
    if (unique_per_group != 1).any():
        raise ValueError("Every correlated group must have one consistent target label")
    group_rows = frame.drop_duplicates(group_column).copy()
    rng = np.random.default_rng(seed)
    shuffled_map: dict[object, int] = {}
    grouper: str | list[str]
    grouper = list(strata_columns) if len(strata_columns) > 1 else strata_columns[0]
    for _, stratum in group_rows.groupby(grouper, sort=False, dropna=False):
        values = stratum["_label"].to_numpy().copy()
        rng.shuffle(values)
        shuffled_map.update(zip(stratum[group_column], values, strict=True))
    return frame[group_column].map(shuffled_map).to_numpy(dtype=np.int64)


def discriminating_cortical_surrogate(
    dataset: SyntheticDataset,
    *,
    seed: int,
    strata_columns: Sequence[str] = ("subject_id", "window"),
) -> NDArray[np.float64]:
    """Destroy target-related K while preserving G, M, C, effort, and artifact."""

    if dataset.scenario == "redundant":
        # The redundant world's E is generated from observed M, not from K. There
        # is no genuine cortical component to intervene on.
        return dataset.eeg.copy()
    rng = np.random.default_rng(seed)
    surrogate_k = dataset.cortical.copy()
    frame = dataset.metadata.copy()
    grouper = list(strata_columns) if len(strata_columns) > 1 else strata_columns[0]
    for _, stratum in frame.groupby(grouper, sort=False, dropna=False):
        positions = stratum.index.to_numpy()
        permuted = positions.copy()
        rng.shuffle(permuted)
        surrogate_k[positions] = dataset.cortical[permuted]
    return (
        surrogate_k
        + dataset.artifact_signal_eeg
        + dataset.artifact_noise_eeg
        + dataset.eeg_noise
    )

