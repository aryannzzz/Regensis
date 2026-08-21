from __future__ import annotations

import numpy as np
import pandas as pd

from regenesis_cmi.evaluation.splits import make_group_splits
from regenesis_cmi.information.decoder import estimate_decoder_cmi
from regenesis_cmi.synthetic.causal_generator import generate_causal_dataset


def test_trial_split_keeps_correlated_windows_together(small_config) -> None:
    data = generate_causal_dataset(small_config, "genuine_cortical", "post", seed=3)
    first = data.metadata.copy()
    second = data.metadata.copy()
    first["window"] = "pre"
    first["window_id"] = 0
    second["window"] = "post"
    second["window_id"] = 1
    metadata = pd.concat((first, second), ignore_index=True)
    labels = np.tile(data.g, 2)
    folds = make_group_splits(metadata, labels=labels, mode="trial", n_splits=3, seed=1)
    for fold in folds:
        train_trials = set(metadata.iloc[fold.train]["trial_uid"])
        test_trials = set(metadata.iloc[fold.test]["trial_uid"])
        assert not train_trials & test_trials


def test_block_split_has_no_block_or_trial_leakage(small_config) -> None:
    data = generate_causal_dataset(small_config, "artifact_only", "post", seed=4)
    folds = make_group_splits(
        data.metadata, labels=data.g, mode="block", n_splits=3, seed=2
    )
    assert len(folds) == 3


def test_decoder_fits_preprocessing_inside_training_fold(small_config) -> None:
    data = generate_causal_dataset(small_config, "genuine_cortical", "post", seed=5)
    result = estimate_decoder_cmi(
        data.g,
        data.eeg,
        data.emg,
        groups=data.metadata["block_uid"],
        direction="I(G;E|M)",
        seed=5,
        c_grid=(0.1, 1.0),
    )
    assert result.diagnostics["equal_search_budget"]
    for fold in result.diagnostics["folds"]:
        assert fold["base_scaler_seen"] == fold["train_samples"]
        assert fold["augmented_scaler_seen"] == fold["train_samples"]
        assert fold["base_search_candidates"] == fold["augmented_search_candidates"]
        assert fold["preprocessing_fit_on_training_fold_only"]

