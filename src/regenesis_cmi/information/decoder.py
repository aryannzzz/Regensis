"""Nested, group-aware decoder/log-loss CMI estimate."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .interface import EstimatorResult


def _matrix(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("Features must be a 1-D or 2-D sample array")
    return array


def _pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs", max_iter=1_000, random_state=seed
                ),
            ),
        ]
    )


def _multiclass_brier(
    labels: NDArray[np.int64], probabilities: NDArray[np.float64], classes: NDArray
) -> float:
    one_hot = np.zeros_like(probabilities)
    lookup = {value: index for index, value in enumerate(classes)}
    for row, value in enumerate(labels):
        one_hot[row, lookup[value]] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _expected_calibration_error(
    labels: NDArray[np.int64], probabilities: NDArray[np.float64], classes: NDArray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = classes[probabilities.argmax(axis=1)]
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def estimate_decoder_cmi(
    g: ArrayLike,
    added: ArrayLike,
    conditioned: ArrayLike,
    *,
    groups: ArrayLike,
    direction: str,
    seed: int = 0,
    outer_splits: int = 3,
    inner_splits: int = 2,
    c_grid: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> EstimatorResult:
    """CE(G|conditioned)-CE(G|conditioned,added) on untouched folds."""

    started = time.perf_counter()
    labels = np.asarray(g, dtype=np.int64)
    added_matrix = _matrix(added)
    conditioned_matrix = _matrix(conditioned)
    group_array = np.asarray(groups)
    if len({len(labels), len(added_matrix), len(conditioned_matrix), len(group_array)}) != 1:
        raise ValueError("Labels, features, and groups must have equal sample counts")
    if len(np.unique(group_array)) < outer_splits:
        raise ValueError("Not enough groups for outer cross-validation")
    classes = np.unique(labels)
    augmented_matrix = np.column_stack((conditioned_matrix, added_matrix))
    outer = GroupKFold(n_splits=outer_splits)
    fold_records: list[dict[str, Any]] = []
    all_labels: list[NDArray[np.int64]] = []
    base_probabilities: list[NDArray[np.float64]] = []
    augmented_probabilities: list[NDArray[np.float64]] = []

    for fold_index, (train, test) in enumerate(
        outer.split(conditioned_matrix, labels, group_array)
    ):
        train_groups = group_array[train]
        test_groups = group_array[test]
        if set(train_groups) & set(test_groups):
            raise AssertionError("Outer evaluation groups overlap")
        available_inner = min(inner_splits, len(np.unique(train_groups)))
        if available_inner < 2:
            raise ValueError("At least two independent inner groups are required")
        inner = GroupKFold(n_splits=available_inner)
        parameter_grid = {"classifier__C": list(c_grid)}

        searches = []
        for matrix in (conditioned_matrix, augmented_matrix):
            search = GridSearchCV(
                _pipeline(seed + fold_index),
                parameter_grid,
                scoring="neg_log_loss",
                cv=inner,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(matrix[train], labels[train], groups=train_groups)
            searches.append(search)
        base_search, augmented_search = searches
        base_probability = base_search.predict_proba(conditioned_matrix[test])
        augmented_probability = augmented_search.predict_proba(augmented_matrix[test])
        if not np.array_equal(base_search.classes_, classes) or not np.array_equal(
            augmented_search.classes_, classes
        ):
            raise ValueError("A training fold omitted a target class")
        all_labels.append(labels[test])
        base_probabilities.append(base_probability)
        augmented_probabilities.append(augmented_probability)
        fold_records.append(
            {
                "fold": fold_index,
                "train_samples": len(train),
                "test_samples": len(test),
                "train_groups": len(np.unique(train_groups)),
                "test_groups": len(np.unique(test_groups)),
                "base_selected_c": base_search.best_params_["classifier__C"],
                "augmented_selected_c": augmented_search.best_params_["classifier__C"],
                "base_search_candidates": len(base_search.cv_results_["params"]),
                "augmented_search_candidates": len(
                    augmented_search.cv_results_["params"]
                ),
                "base_scaler_seen": int(
                    base_search.best_estimator_.named_steps["scale"].n_samples_seen_
                ),
                "augmented_scaler_seen": int(
                    augmented_search.best_estimator_.named_steps["scale"].n_samples_seen_
                ),
                "preprocessing_fit_on_training_fold_only": True,
            }
        )

    held_out_labels = np.concatenate(all_labels)
    base_probability = np.vstack(base_probabilities)
    augmented_probability = np.vstack(augmented_probabilities)
    base_cross_entropy_nats = log_loss(
        held_out_labels, base_probability, labels=classes
    )
    augmented_cross_entropy_nats = log_loss(
        held_out_labels, augmented_probability, labels=classes
    )
    estimate_bits = float(
        (base_cross_entropy_nats - augmented_cross_entropy_nats) / math.log(2.0)
    )
    warnings = (
        ("negative_decoder_estimate_preserved",) if estimate_bits < 0 else ()
    )
    return EstimatorResult(
        estimate_bits=estimate_bits,
        direction=direction,
        estimator_name="nested_logistic_log_loss",
        sample_size=len(labels),
        seed=seed,
        runtime_seconds=time.perf_counter() - started,
        diagnostics={
            "base_cross_entropy_bits": float(base_cross_entropy_nats / math.log(2.0)),
            "augmented_cross_entropy_bits": float(
                augmented_cross_entropy_nats / math.log(2.0)
            ),
            "base_accuracy": float(
                accuracy_score(held_out_labels, classes[base_probability.argmax(axis=1)])
            ),
            "augmented_accuracy": float(
                accuracy_score(
                    held_out_labels, classes[augmented_probability.argmax(axis=1)]
                )
            ),
            "base_brier": _multiclass_brier(
                held_out_labels, base_probability, classes
            ),
            "augmented_brier": _multiclass_brier(
                held_out_labels, augmented_probability, classes
            ),
            "base_ece": _expected_calibration_error(
                held_out_labels, base_probability, classes
            ),
            "augmented_ece": _expected_calibration_error(
                held_out_labels, augmented_probability, classes
            ),
            "classifier_family": "regularized_multinomial_logistic_regression",
            "equal_search_budget": True,
            "c_grid": list(c_grid),
            "outer_split": "group_k_fold",
            "inner_split": "group_k_fold",
            "folds": fold_records,
            "interpretation": "decoder-dependent held-out log-loss difference",
        },
        warnings=warnings,
    )

