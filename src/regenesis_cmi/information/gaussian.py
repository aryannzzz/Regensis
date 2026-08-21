"""Gaussian reference calculations; never used to recast categorical G."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import multivariate_normal


def partial_correlation(
    rho_xy: float, rho_xz: float, rho_yz: float
) -> float:
    denominator = math.sqrt((1.0 - rho_xz**2) * (1.0 - rho_yz**2))
    if denominator <= 0:
        raise ValueError("Correlations imply a degenerate conditioning problem")
    return (rho_xy - rho_xz * rho_yz) / denominator


def gaussian_cmi_from_partial_correlation(rho_xy_given_z: float) -> float:
    squared = float(rho_xy_given_z) ** 2
    if squared >= 1:
        return math.inf
    return float(-0.5 * math.log2(1.0 - squared))


def _logdet(matrix: NDArray[np.floating]) -> float:
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("Covariance submatrix must be positive definite")
    return float(value)


def gaussian_cmi_from_covariance(
    covariance: ArrayLike,
    x_indices: Sequence[int],
    y_indices: Sequence[int],
    z_indices: Sequence[int],
) -> float:
    """Analytic multivariate Gaussian I(X;Y|Z), returned in bits."""

    covariance_array = np.asarray(covariance, dtype=float)
    x = np.asarray(tuple(x_indices), dtype=int)
    y = np.asarray(tuple(y_indices), dtype=int)
    z = np.asarray(tuple(z_indices), dtype=int)
    if x.size == 0 or y.size == 0:
        raise ValueError("X and Y must each contain at least one dimension")

    def sub(indices: NDArray[np.integer]) -> NDArray[np.floating]:
        return covariance_array[np.ix_(indices, indices)]

    xz = np.concatenate((x, z))
    yz = np.concatenate((y, z))
    xyz = np.concatenate((x, y, z))
    z_logdet = 0.0 if z.size == 0 else _logdet(sub(z))
    nats = 0.5 * (
        _logdet(sub(xz)) + _logdet(sub(yz)) - z_logdet - _logdet(sub(xyz))
    )
    return float(nats / math.log(2.0))


def gaussian_benchmark_covariance(
    rho_xy: float = 0.65,
    rho_xz: float = 0.35,
    rho_yz: float = 0.25,
) -> NDArray[np.float64]:
    covariance = np.array(
        [[1.0, rho_xy, rho_xz], [rho_xy, 1.0, rho_yz], [rho_xz, rho_yz, 1.0]],
        dtype=float,
    )
    if np.min(np.linalg.eigvalsh(covariance)) <= 0:
        raise ValueError("Requested correlations do not define a covariance matrix")
    return covariance


def sample_gaussian_benchmark(
    n_samples: int,
    *,
    seed: int = 0,
    covariance: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    covariance_array = (
        gaussian_benchmark_covariance()
        if covariance is None
        else np.asarray(covariance, dtype=float)
    )
    samples = np.random.default_rng(seed).multivariate_normal(
        np.zeros(3), covariance_array, size=n_samples
    )
    return samples[:, [0]], samples[:, [1]], samples[:, [2]]


def gaussian_cmi_monte_carlo(
    covariance: ArrayLike,
    *,
    n_samples: int = 100_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Independent likelihood-ratio Monte Carlo check for scalar X,Y,Z."""

    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.shape != (3, 3):
        raise ValueError("The generic scalar benchmark expects a 3x3 covariance")
    samples = np.random.default_rng(seed).multivariate_normal(
        np.zeros(3), covariance_array, size=n_samples
    )
    xyz = multivariate_normal(mean=np.zeros(3), cov=covariance_array).logpdf(samples)
    z = multivariate_normal(mean=np.zeros(1), cov=covariance_array[np.ix_([2], [2])]).logpdf(
        samples[:, [2]]
    )
    xz = multivariate_normal(
        mean=np.zeros(2), cov=covariance_array[np.ix_([0, 2], [0, 2])]
    ).logpdf(samples[:, [0, 2]])
    yz = multivariate_normal(
        mean=np.zeros(2), cov=covariance_array[np.ix_([1, 2], [1, 2])]
    ).logpdf(samples[:, [1, 2]])
    information = (xyz + z - xz - yz) / math.log(2.0)
    return {
        "estimate_bits": float(np.mean(information)),
        "standard_error_bits": float(
            np.std(information, ddof=1) / math.sqrt(n_samples)
        ),
        "sample_size": n_samples,
    }
