"""Reusable leakage-safe group split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


@dataclass(frozen=True)
class FoldIndices:
    train: NDArray[np.int64]
    test: NDArray[np.int64]
    mode: str


GROUP_COLUMNS = {
    "trial": "trial_uid",
    "block": "block_uid",
    "subject": "subject_id",
}


def make_group_splits(
    metadata: pd.DataFrame,
    *,
    labels: NDArray[np.int64] | None = None,
    mode: str = "block",
    n_splits: int = 3,
    seed: int = 0,
) -> list[FoldIndices]:
    if mode not in GROUP_COLUMNS:
        raise ValueError(f"mode must be one of {tuple(GROUP_COLUMNS)}")
    group_column = GROUP_COLUMNS[mode]
    groups = metadata[group_column].to_numpy()
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough independent groups for the requested folds")
    indices = np.arange(len(metadata))
    if labels is None:
        splitter = GroupKFold(n_splits=n_splits)
        iterator = splitter.split(indices, groups=groups)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(indices, np.asarray(labels), groups)
    folds = [
        FoldIndices(train=train.astype(np.int64), test=test.astype(np.int64), mode=mode)
        for train, test in iterator
    ]
    assert_no_leakage(folds, metadata, mode=mode)
    return folds


def assert_no_leakage(
    folds: list[FoldIndices], metadata: pd.DataFrame, *, mode: str
) -> None:
    for fold in folds:
        train_trials = set(metadata.iloc[fold.train]["trial_uid"])
        test_trials = set(metadata.iloc[fold.test]["trial_uid"])
        if train_trials & test_trials:
            raise AssertionError("A trial appears in both train and test")
        if mode == "block":
            train_blocks = set(metadata.iloc[fold.train]["block_uid"])
            test_blocks = set(metadata.iloc[fold.test]["block_uid"])
            if train_blocks & test_blocks:
                raise AssertionError("A block appears in both train and test")
        if mode == "subject":
            train_subjects = set(metadata.iloc[fold.train]["subject_id"])
            test_subjects = set(metadata.iloc[fold.test]["subject_id"])
            if train_subjects & test_subjects:
                raise AssertionError("A subject appears in both train and test")

