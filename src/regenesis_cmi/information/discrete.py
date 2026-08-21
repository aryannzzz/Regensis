"""Exact empirical information identities for genuinely discrete variables."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .interface import EstimatorResult


def _rows(values: ArrayLike) -> NDArray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ValueError("A discrete variable must be a 1-D or 2-D sample array")
    if len(array) == 0:
        raise ValueError("Information is undefined for an empty sample")
    return array


def _join(*variables: ArrayLike) -> NDArray:
    arrays = [_rows(variable) for variable in variables]
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("All variables must contain the same number of samples")
    return np.column_stack(arrays)


def entropy(values: ArrayLike) -> float:
    """Empirical plug-in entropy in bits."""

    array = _rows(values)
    _, counts = np.unique(array, axis=0, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def joint_entropy(*variables: ArrayLike) -> float:
    return entropy(_join(*variables))


def conditional_entropy(values: ArrayLike, given: ArrayLike) -> float:
    return joint_entropy(values, given) - entropy(given)


def mutual_information(x: ArrayLike, y: ArrayLike) -> float:
    return entropy(x) + entropy(y) - joint_entropy(x, y)


def conditional_mutual_information(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> float:
    """Compute I(X;Y|Z) through the four-entropy identity, in bits."""

    value = (
        joint_entropy(x, z)
        + joint_entropy(y, z)
        - entropy(z)
        - joint_entropy(x, y, z)
    )
    # Round only numerical cancellation around exact zero, never a negative estimate.
    return 0.0 if abs(value) < 1e-14 else float(value)


def cmi_identities(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> dict[str, float]:
    """Return both mandated discrete CMI identities for an audit/test."""

    four_entropy = conditional_mutual_information(x, y, z)
    conditional_difference = conditional_entropy(x, z) - conditional_entropy(
        x, _join(z, y)
    )
    return {
        "four_entropy_bits": four_entropy,
        "conditional_entropy_difference_bits": float(conditional_difference),
        "absolute_identity_gap_bits": abs(four_entropy - conditional_difference),
    }


def estimate_discrete_cmi(
    g: ArrayLike,
    added: ArrayLike,
    conditioned: ArrayLike,
    *,
    direction: str,
    seed: int = 0,
) -> EstimatorResult:
    started = time.perf_counter()
    estimate = conditional_mutual_information(g, added, conditioned)
    identities = cmi_identities(g, added, conditioned)
    return EstimatorResult(
        estimate_bits=estimate,
        direction=direction,
        estimator_name="exact_discrete_plugin",
        sample_size=len(_rows(g)),
        seed=seed,
        runtime_seconds=time.perf_counter() - started,
        diagnostics=identities,
    )


def validate_entropy_sanity() -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for cardinality in (1, 2, 4, 8):
        values = np.arange(cardinality, dtype=int)
        rows.append(
            {
                "cardinality": cardinality,
                "estimate_bits": entropy(values),
                "truth_bits": float(np.log2(cardinality)),
            }
        )
    return rows

