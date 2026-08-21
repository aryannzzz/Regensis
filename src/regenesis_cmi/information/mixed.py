"""Direct hybrid CMI estimator for categorical/continuous mixed data.

This is a clean implementation of the entropy decomposition in Equation 9 of
Zan et al. (2022), combined with their conditional kNN entropy construction.
The authors' reference repository is MIT licensed. No source code is vendored.
"""

from __future__ import annotations

import math
import time
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import digamma
from scipy.spatial import cKDTree

from .interface import EstimatorResult

PAPER_URL = "https://doi.org/10.3390/e24091234"
REFERENCE_CODE_URL = "https://github.com/leizan/CMIh2022"


def _matrix(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("Each variable must be a one- or two-dimensional sample array")
    return array.astype(float, copy=False)


def _discrete_entropy_nats(values: NDArray[np.float64]) -> float:
    if values.shape[1] == 0:
        return 0.0
    _, counts = np.unique(values, axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def _continuous_entropy_knn_nats(
    values: NDArray[np.float64], k_fraction: float
) -> float:
    n_samples, dimensions = values.shape
    if dimensions == 0 or n_samples <= 1:
        return 0.0
    k = max(1, int(round(k_fraction * n_samples)))
    k = min(k, n_samples - 1)
    tree = cKDTree(values)
    distances = tree.query(values, k=k + 1, p=np.inf, workers=1)[0][:, -1]
    positive = distances > 0
    if not positive.any():
        return 0.0
    usable = distances[positive]
    # Maximum-norm unit-ball volume is 2**d, hence d*log(2*epsilon).
    return float(
        digamma(len(usable))
        - digamma(k)
        + dimensions * np.mean(np.log(2.0 * usable))
    )


def mixed_entropy_nats(
    values: ArrayLike,
    discrete_mask: ArrayLike,
    *,
    k_fraction: float = 0.10,
) -> float:
    """Estimate H(discrete, continuous) as H(D)+E_D[H(C|D)]."""

    array = _matrix(values)
    mask = np.asarray(discrete_mask, dtype=bool)
    if mask.shape != (array.shape[1],):
        raise ValueError("discrete_mask must identify every feature column")
    discrete = array[:, mask]
    continuous = array[:, ~mask]
    discrete_entropy = _discrete_entropy_nats(discrete)
    if continuous.shape[1] == 0:
        return discrete_entropy
    if discrete.shape[1] == 0:
        return _continuous_entropy_knn_nats(continuous, k_fraction)
    unique, inverse, counts = np.unique(
        discrete, axis=0, return_inverse=True, return_counts=True
    )
    conditional = 0.0
    for cluster_index, count in enumerate(counts):
        cluster = continuous[inverse == cluster_index]
        conditional += (count / len(array)) * _continuous_entropy_knn_nats(
            cluster, k_fraction
        )
    return float(discrete_entropy + conditional)


def estimate_mixed_cmi(
    g: ArrayLike,
    added: ArrayLike,
    conditioned: ArrayLike,
    *,
    direction: str,
    seed: int = 0,
    k_fraction: float = 0.10,
    added_discrete_mask: ArrayLike | None = None,
    conditioned_discrete_mask: ArrayLike | None = None,
) -> EstimatorResult:
    """Estimate I(G;added|conditioned) using the published CMIh identity."""

    started = time.perf_counter()
    g_matrix = _matrix(g)
    added_matrix = _matrix(added)
    conditioned_matrix = _matrix(conditioned)
    lengths = {len(g_matrix), len(added_matrix), len(conditioned_matrix)}
    if len(lengths) != 1:
        raise ValueError("All variables must contain the same number of samples")
    if not 0.0 < k_fraction < 1.0:
        raise ValueError("k_fraction must lie strictly between zero and one")

    added_mask = (
        np.zeros(added_matrix.shape[1], dtype=bool)
        if added_discrete_mask is None
        else np.asarray(added_discrete_mask, dtype=bool)
    )
    conditioned_mask = (
        np.zeros(conditioned_matrix.shape[1], dtype=bool)
        if conditioned_discrete_mask is None
        else np.asarray(conditioned_discrete_mask, dtype=bool)
    )
    if added_mask.shape != (added_matrix.shape[1],) or conditioned_mask.shape != (
        conditioned_matrix.shape[1],
    ):
        raise ValueError("Discrete masks must identify every feature column")

    # Scale continuous dimensions only. No discrete variable is jittered or cast
    # as a metric coordinate. Masks make the implementation usable for Generator
    # A while real E/M default to continuous.
    combined = np.column_stack((added_matrix, conditioned_matrix)).astype(float)
    combined_mask = np.concatenate((added_mask, conditioned_mask))
    continuous_indices = np.flatnonzero(~combined_mask)
    standardized = combined.copy()
    if len(continuous_indices):
        continuous = combined[:, continuous_indices]
        mean = continuous.mean(axis=0)
        std = continuous.std(axis=0)
        continuous = (continuous - mean) / np.where(std > 0, std, 1.0)
        continuous += 1e-12 * np.random.default_rng(seed).standard_normal(
            continuous.shape
        )
        standardized[:, continuous_indices] = continuous
    added_width = added_matrix.shape[1]
    added_standardized = standardized[:, :added_width]
    conditioned_standardized = standardized[:, added_width:]

    gz = np.column_stack((g_matrix, conditioned_standardized))
    yz = np.column_stack((added_standardized, conditioned_standardized))
    gyz = np.column_stack((g_matrix, added_standardized, conditioned_standardized))
    h_gz = mixed_entropy_nats(
        gz,
        np.r_[np.ones(g_matrix.shape[1], dtype=bool), conditioned_mask],
        k_fraction=k_fraction,
    )
    h_yz = mixed_entropy_nats(
        yz, np.r_[added_mask, conditioned_mask], k_fraction=k_fraction
    )
    h_z = mixed_entropy_nats(
        conditioned_standardized,
        conditioned_mask,
        k_fraction=k_fraction,
    )
    h_gyz = mixed_entropy_nats(
        gyz,
        np.r_[np.ones(g_matrix.shape[1], dtype=bool), added_mask, conditioned_mask],
        k_fraction=k_fraction,
    )
    estimate_bits = float((h_gz + h_yz - h_z - h_gyz) / math.log(2.0))
    warnings: list[str] = []
    total_dimension = g_matrix.shape[1] + added_matrix.shape[1] + conditioned_matrix.shape[1]
    if len(g_matrix) < 20 * total_dimension:
        warnings.append("sample_size_is_small_relative_to_mixed_dimension")
    if estimate_bits < 0:
        warnings.append("negative_finite_sample_estimate_preserved")
    return EstimatorResult(
        estimate_bits=estimate_bits,
        direction=direction,
        estimator_name="zmadg_cmi_hybrid_knn",
        sample_size=len(g_matrix),
        seed=seed,
        runtime_seconds=time.perf_counter() - started,
        diagnostics={
            "k_fraction": k_fraction,
            "categorical_dimensions": g_matrix.shape[1],
            "added_continuous_dimensions": added_matrix.shape[1],
            "conditioning_continuous_dimensions": conditioned_matrix.shape[1],
            "added_discrete_dimensions": int(added_mask.sum()),
            "conditioning_discrete_dimensions": int(conditioned_mask.sum()),
            "paper": PAPER_URL,
            "reference_code": REFERENCE_CODE_URL,
            "reference_code_license": "MIT",
        },
        warnings=tuple(warnings),
    )


def estimate_continuous_cmi(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    seed: int = 0,
    k_fraction: float = 0.10,
) -> EstimatorResult:
    """CMIh's all-continuous limiting case for the generic X,Y,Z benchmark."""

    started = time.perf_counter()
    x_matrix, y_matrix, z_matrix = _matrix(x), _matrix(y), _matrix(z)
    if len({len(x_matrix), len(y_matrix), len(z_matrix)}) != 1:
        raise ValueError("X, Y, and Z must have equal sample counts")
    combined = np.column_stack((x_matrix, y_matrix, z_matrix))
    combined = (combined - combined.mean(axis=0)) / np.where(
        combined.std(axis=0) > 0, combined.std(axis=0), 1.0
    )
    combined += 1e-12 * np.random.default_rng(seed).standard_normal(combined.shape)
    nx, ny = x_matrix.shape[1], y_matrix.shape[1]
    x_scaled = combined[:, :nx]
    y_scaled = combined[:, nx : nx + ny]
    z_scaled = combined[:, nx + ny :]
    xz = np.column_stack((x_scaled, z_scaled))
    yz = np.column_stack((y_scaled, z_scaled))
    xyz = np.column_stack((x_scaled, y_scaled, z_scaled))
    h_xz = mixed_entropy_nats(
        xz, np.zeros(xz.shape[1], dtype=bool), k_fraction=k_fraction
    )
    h_yz = mixed_entropy_nats(
        yz, np.zeros(yz.shape[1], dtype=bool), k_fraction=k_fraction
    )
    h_z = mixed_entropy_nats(
        z_scaled, np.zeros(z_scaled.shape[1], dtype=bool), k_fraction=k_fraction
    )
    h_xyz = mixed_entropy_nats(
        xyz, np.zeros(xyz.shape[1], dtype=bool), k_fraction=k_fraction
    )
    estimate = float((h_xz + h_yz - h_z - h_xyz) / math.log(2.0))
    return EstimatorResult(
        estimate_bits=estimate,
        direction="generic",
        estimator_name="zmadg_cmi_hybrid_knn_continuous",
        sample_size=len(combined),
        seed=seed,
        runtime_seconds=time.perf_counter() - started,
        diagnostics={
            "k_fraction": k_fraction,
            "x_dimensions": nx,
            "y_dimensions": ny,
            "z_dimensions": z_matrix.shape[1],
            "paper": PAPER_URL,
        },
        warnings=("negative_finite_sample_estimate_preserved",)
        if estimate < 0
        else (),
    )
