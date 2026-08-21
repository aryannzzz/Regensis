"""Independent Bayes oracle for known class-conditional Gaussian worlds."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


@dataclass(frozen=True)
class ObservedMoments:
    means: dict[str, NDArray[np.float64]]
    covariances: dict[str, NDArray[np.float64]]
    priors: dict[str, NDArray[np.float64]]
    key_weights: dict[str, float]
    slices: dict[str, NDArray[np.int64]]


@dataclass(frozen=True)
class OracleEstimate:
    estimate_bits: float
    standard_error_bits: float
    sample_size: int
    precision_target_bits: float
    precision_achieved: bool
    direction: str
    control: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate_bits": self.estimate_bits,
            "standard_error_bits": self.standard_error_bits,
            "sample_size": self.sample_size,
            "precision_target_bits": self.precision_target_bits,
            "precision_achieved": self.precision_achieved,
            "direction": self.direction,
            "control": self.control,
            "diagnostics": self.diagnostics,
        }


@dataclass
class LinearGaussianOracle:
    """Known latent Gaussian model; truth is not defined by a tested estimator."""

    scenario: str
    window: str
    n_classes: int
    keys: tuple[str, ...]
    key_weights: dict[str, float]
    priors: dict[str, NDArray[np.float64]]
    base_means: dict[str, NDArray[np.float64]]
    base_covariances: dict[str, NDArray[np.float64]]
    base_slices: dict[str, slice]
    artifact_weights: dict[str, NDArray[np.float64]]
    redundant_weights: dict[str, NDArray[np.float64] | None]
    emg_projections: dict[str, NDArray[np.float64]]
    eeg_dimensions: int
    emg_dimensions: int
    cranial_dimensions: int
    parameter_manifest: dict[str, Any]

    def _cranial_standardization(
        self, key: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        means = self.base_means[key]
        covariances = self.base_covariances[key]
        priors = self.priors[key]
        sc = self.base_slices["cranial_signal"]
        nc = self.base_slices["cranial_noise"]
        c_means = means[:, sc] + means[:, nc]
        c_covariances = (
            covariances[:, sc, :][:, :, sc]
            + covariances[:, nc, :][:, :, nc]
            + covariances[:, sc, :][:, :, nc]
            + covariances[:, nc, :][:, :, sc]
        )
        global_mean = np.sum(priors[:, None] * c_means, axis=0)
        second = np.sum(
            priors[:, None]
            * (np.diagonal(c_covariances, axis1=1, axis2=2) + c_means**2),
            axis=0,
        )
        std = np.sqrt(np.maximum(second - global_mean**2, 1e-12))
        return global_mean, std

    def observed_moments(
        self,
        *,
        alpha: float = 1.0,
        eeg_variant: str = "raw",
        control: str = "none",
        proxy_quality: float = 1.0,
        emg_richness: str = "rich",
        eeg_indices: NDArray[np.int64] | None = None,
    ) -> ObservedMoments:
        """Construct exact moments for E, degraded M, and an optional control."""

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if not 0.0 <= proxy_quality <= 1.0:
            raise ValueError("proxy_quality must lie in [0, 1]")
        if eeg_variant not in {"raw", "matched", "clean"}:
            raise ValueError("eeg_variant must be raw, matched, or clean")
        if control not in {"none", "true_c", "proxy"}:
            raise ValueError("control must be none, true_c, or proxy")
        if emg_richness not in self.emg_projections:
            raise KeyError(f"Unknown EMG richness: {emg_richness}")

        selected_eeg = (
            np.arange(self.eeg_dimensions, dtype=int)
            if eeg_indices is None
            else np.asarray(eeg_indices, dtype=int)
        )
        projection = self.emg_projections[emg_richness]
        observed_means: dict[str, NDArray[np.float64]] = {}
        observed_covariances: dict[str, NDArray[np.float64]] = {}
        output_slices: dict[str, NDArray[np.int64]] = {}

        for key in self.keys:
            base_dim = self.base_means[key].shape[1]
            k_slice = self.base_slices["cortical"]
            sm_slice = self.base_slices["emg_signal"]
            nm_slice = self.base_slices["emg_noise"]
            sc_slice = self.base_slices["cranial_signal"]
            nc_slice = self.base_slices["cranial_noise"]
            ne_slice = self.base_slices["eeg_noise"]

            eeg_transform = np.zeros((self.eeg_dimensions, base_dim), dtype=float)
            redundant = self.redundant_weights[key]
            if redundant is not None:
                eeg_transform[:, sm_slice] = redundant
                eeg_transform[:, nm_slice] = redundant
                eeg_transform[:, ne_slice] = np.eye(self.eeg_dimensions)
            else:
                eeg_transform[:, k_slice] = np.eye(self.eeg_dimensions)
                eeg_transform[:, ne_slice] = np.eye(self.eeg_dimensions)
                if eeg_variant != "clean":
                    signal_factor = alpha if eeg_variant == "matched" else 1.0
                    eeg_transform[:, sc_slice] = (
                        signal_factor * self.artifact_weights[key]
                    )
                    eeg_transform[:, nc_slice] = self.artifact_weights[key]
            eeg_transform = eeg_transform[selected_eeg]

            emg_transform = np.zeros((self.emg_dimensions, base_dim), dtype=float)
            emg_transform[:, sm_slice] = alpha * np.eye(self.emg_dimensions)
            emg_transform[:, nm_slice] = np.eye(self.emg_dimensions)
            emg_transform = projection @ emg_transform

            transforms = [eeg_transform, emg_transform]
            offsets = [np.zeros(len(selected_eeg)), np.zeros(projection.shape[0])]
            extra_variance_sizes = [len(selected_eeg), projection.shape[0]]
            output_slices = {
                "eeg": np.arange(len(selected_eeg), dtype=int),
                "emg": np.arange(
                    len(selected_eeg), len(selected_eeg) + projection.shape[0], dtype=int
                ),
            }

            if control in {"true_c", "proxy"}:
                c_transform = np.zeros((self.cranial_dimensions, base_dim), dtype=float)
                c_transform[:, sc_slice] = np.eye(self.cranial_dimensions)
                c_transform[:, nc_slice] = np.eye(self.cranial_dimensions)
                if control == "proxy":
                    global_mean, c_std = self._cranial_standardization(key)
                    c_transform = (
                        proxy_quality * np.diag(1.0 / c_std) @ c_transform
                    )
                    c_offset = -proxy_quality * global_mean / c_std
                else:
                    c_offset = np.zeros(self.cranial_dimensions)
                start = len(selected_eeg) + projection.shape[0]
                output_slices["control"] = np.arange(
                    start, start + self.cranial_dimensions, dtype=int
                )
                transforms.append(c_transform)
                offsets.append(c_offset)
                extra_variance_sizes.append(self.cranial_dimensions)

            transform = np.vstack(transforms)
            offset = np.concatenate(offsets)
            means = self.base_means[key] @ transform.T + offset
            covariances = np.einsum(
                "ab,kbc,dc->kad",
                transform,
                self.base_covariances[key],
                transform,
                optimize=True,
            )
            if control == "proxy":
                control_indices = output_slices["control"]
                covariances[:, control_indices[:, None], control_indices] += (
                    1.0 - proxy_quality**2
                ) * np.eye(self.cranial_dimensions)[None, :, :]
            # A tiny numerical ridge does not change the generating distribution in
            # any meaningful way; it stabilizes Cholesky at r=1 proxy controls.
            covariances += 1e-10 * np.eye(covariances.shape[-1])[None, :, :]
            observed_means[key] = means
            observed_covariances[key] = covariances

        return ObservedMoments(
            means=observed_means,
            covariances=observed_covariances,
            priors=self.priors,
            key_weights=self.key_weights,
            slices=output_slices,
        )

    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        for key in self.keys:
            hasher.update(self.base_means[key].tobytes())
            hasher.update(self.base_covariances[key].tobytes())
            hasher.update(self.artifact_weights[key].tobytes())
        return hasher.hexdigest()


def discrete_pmf_cmi_oracle(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    *,
    n_samples: int = 20_000,
    seed: int = 0,
    direction: str = "generic",
) -> OracleEstimate:
    """Likelihood-ratio Monte Carlo from a declared exact discrete PMF."""

    def rows(values: NDArray) -> NDArray:
        array = np.asarray(values)
        return array[:, None] if array.ndim == 1 else array

    x_rows, y_rows, z_rows = rows(x), rows(y), rows(z)
    if len({len(x_rows), len(y_rows), len(z_rows)}) != 1:
        raise ValueError("Discrete oracle variables must have equal lengths")
    joint = np.column_stack((x_rows, y_rows, z_rows))
    states, counts = np.unique(joint, axis=0, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    nx, ny = x_rows.shape[1], y_rows.shape[1]
    information = np.empty(len(states), dtype=float)
    for index, state in enumerate(states):
        x_value = state[:nx]
        y_value = state[nx : nx + ny]
        z_value = state[nx + ny :]
        z_mask = np.all(states[:, nx + ny :] == z_value, axis=1)
        xz_mask = z_mask & np.all(states[:, :nx] == x_value, axis=1)
        yz_mask = z_mask & np.all(states[:, nx : nx + ny] == y_value, axis=1)
        p_xyz = probabilities[index]
        p_z = probabilities[z_mask].sum()
        p_xz = probabilities[xz_mask].sum()
        p_yz = probabilities[yz_mask].sum()
        information[index] = math.log2((p_xyz * p_z) / (p_xz * p_yz))
    sampled = np.random.default_rng(seed).choice(
        len(states), size=n_samples, p=probabilities
    )
    values = information[sampled]
    standard_error = float(np.std(values, ddof=1) / math.sqrt(n_samples))
    return OracleEstimate(
        estimate_bits=float(np.mean(values)),
        standard_error_bits=standard_error,
        sample_size=n_samples,
        precision_target_bits=standard_error,
        precision_achieved=True,
        direction=direction,
        control="none",
        diagnostics={"source": "declared exact discrete probability mass function"},
    )


def _prepare_logpdf(
    means: NDArray[np.float64],
    covariances: NDArray[np.float64],
    dimensions: NDArray[np.int64],
) -> list[tuple[NDArray[np.float64], NDArray[np.float64], float]]:
    prepared = []
    d = len(dimensions)
    for class_index in range(len(means)):
        mean = means[class_index, dimensions]
        covariance = covariances[class_index][np.ix_(dimensions, dimensions)]
        chol = np.linalg.cholesky(covariance)
        log_normalizer = d * math.log(2.0 * math.pi) + 2.0 * np.log(
            np.diag(chol)
        ).sum()
        prepared.append((mean, chol, float(log_normalizer)))
    return prepared


def _posterior_log_probability(
    samples: NDArray[np.float64],
    labels: NDArray[np.int64],
    priors: NDArray[np.float64],
    prepared: list[tuple[NDArray[np.float64], NDArray[np.float64], float]],
) -> NDArray[np.float64]:
    scores = np.empty((len(samples), len(prepared)), dtype=float)
    for class_index, (mean, chol, log_normalizer) in enumerate(prepared):
        centered = (samples - mean).T
        solved = np.linalg.solve(chol, centered)
        scores[:, class_index] = (
            math.log(priors[class_index])
            - 0.5 * (np.sum(solved**2, axis=0) + log_normalizer)
        )
    return scores[np.arange(len(labels)), labels] - logsumexp(scores, axis=1)


def monte_carlo_cmi(
    model: LinearGaussianOracle,
    *,
    direction: str = "I(G;E|M)",
    control: str = "none",
    proxy_quality: float = 1.0,
    alpha: float = 1.0,
    eeg_variant: str = "raw",
    emg_richness: str = "rich",
    eeg_indices: NDArray[np.int64] | None = None,
    seed: int = 0,
    precision_bits: float = 1e-3,
    min_samples: int = 10_000,
    max_samples: int = 200_000,
    batch_size: int = 5_000,
    oracle_keys: tuple[str, ...] | None = None,
) -> OracleEstimate:
    """Adaptive independent Monte Carlo expectation of the Bayes log ratio."""

    if direction not in {"I(G;E|M)", "I(G;M|E)"}:
        raise ValueError("The biological oracle reports one of the two CMI directions")
    moments = model.observed_moments(
        alpha=alpha,
        eeg_variant=eeg_variant,
        control=control,
        proxy_quality=proxy_quality,
        emg_richness=emg_richness,
        eeg_indices=eeg_indices,
    )
    selected_keys = model.keys if oracle_keys is None else tuple(oracle_keys)
    if not selected_keys or not set(selected_keys).issubset(model.keys):
        raise ValueError("oracle_keys must be a non-empty subset of model keys")
    eeg_dims = moments.slices["eeg"]
    emg_dims = moments.slices["emg"]
    control_dims = moments.slices.get("control", np.array([], dtype=int))
    if direction == "I(G;E|M)":
        baseline_dims = np.concatenate((emg_dims, control_dims))
        augmented_dims = np.concatenate((emg_dims, eeg_dims, control_dims))
    else:
        baseline_dims = np.concatenate((eeg_dims, control_dims))
        augmented_dims = np.concatenate((eeg_dims, emg_dims, control_dims))

    prepared: dict[str, dict[str, Any]] = {}
    for key in selected_keys:
        prepared[key] = {
            "full_chol": [
                np.linalg.cholesky(covariance)
                for covariance in moments.covariances[key]
            ],
            "baseline": _prepare_logpdf(
                moments.means[key], moments.covariances[key], baseline_dims
            ),
            "augmented": _prepare_logpdf(
                moments.means[key], moments.covariances[key], augmented_dims
            ),
        }

    rng = np.random.default_rng(seed)
    key_probabilities = np.asarray([model.key_weights[key] for key in selected_keys])
    key_probabilities = key_probabilities / key_probabilities.sum()
    information_values: list[NDArray[np.float64]] = []
    total = 0
    standard_error = math.inf

    while total < max_samples and (total < min_samples or standard_error > precision_bits):
        current = min(batch_size, max_samples - total)
        key_indices = rng.choice(len(selected_keys), size=current, p=key_probabilities)
        batch_information = np.empty(current, dtype=float)
        for key_index, key in enumerate(selected_keys):
            positions = np.flatnonzero(key_indices == key_index)
            if len(positions) == 0:
                continue
            priors = moments.priors[key]
            labels = rng.choice(model.n_classes, size=len(positions), p=priors).astype(
                np.int64
            )
            samples = np.empty(
                (len(positions), moments.means[key].shape[1]), dtype=float
            )
            for class_index in range(model.n_classes):
                class_positions = np.flatnonzero(labels == class_index)
                if len(class_positions) == 0:
                    continue
                noise = rng.standard_normal(
                    (len(class_positions), moments.means[key].shape[1])
                )
                samples[class_positions] = (
                    noise @ prepared[key]["full_chol"][class_index].T
                    + moments.means[key][class_index]
                )
            log_base = _posterior_log_probability(
                samples[:, baseline_dims],
                labels,
                priors,
                prepared[key]["baseline"],
            )
            log_augmented = _posterior_log_probability(
                samples[:, augmented_dims],
                labels,
                priors,
                prepared[key]["augmented"],
            )
            batch_information[positions] = (log_augmented - log_base) / math.log(2.0)
        information_values.append(batch_information)
        total += current
        joined = np.concatenate(information_values)
        standard_error = float(np.std(joined, ddof=1) / math.sqrt(total))

    joined = np.concatenate(information_values)
    estimate = float(np.mean(joined))
    return OracleEstimate(
        estimate_bits=estimate,
        standard_error_bits=standard_error,
        sample_size=total,
        precision_target_bits=precision_bits,
        precision_achieved=standard_error <= precision_bits,
        direction=direction,
        control=control,
        diagnostics={
            "alpha": alpha,
            "eeg_variant": eeg_variant,
            "emg_richness": emg_richness,
            "proxy_quality": proxy_quality if control == "proxy" else None,
            "conditioned_on_subject_block_metadata": len(model.keys) > 1,
            "oracle_keys": list(selected_keys),
            "model_fingerprint": model.fingerprint(),
        },
    )
