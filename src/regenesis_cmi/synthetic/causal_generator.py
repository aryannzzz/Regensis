"""Generator B: causal, feature-level EEG--EMG statistical abstraction."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from regenesis_cmi.utils.seeds import derive_seed

from .oracle import LinearGaussianOracle
from .schema import GeneratorConfig, SCENARIOS, WINDOWS, SyntheticDataset


SCENARIO_STRENGTHS: dict[str, dict[str, float]] = {
    "null": {"cortical": 0.0, "artifact": 0.0, "shared_eeg": 0.0},
    "redundant": {"cortical": 0.0, "artifact": 0.0, "shared_eeg": 0.0},
    "genuine_cortical": {"cortical": 1.35, "artifact": 0.0, "shared_eeg": 0.30},
    "artifact_only": {"cortical": 0.0, "artifact": 1.45, "shared_eeg": 0.0},
    "cortical_plus_artifact": {
        "cortical": 1.00,
        "artifact": 1.15,
        "shared_eeg": 0.25,
    },
}


def _class_codes(n_classes: int) -> NDArray[np.float64]:
    if n_classes == 4:
        return np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    angles = 2.0 * np.pi * np.arange(n_classes) / n_classes
    return np.column_stack((np.cos(angles), np.sin(angles)))


def eeg_feature_metadata(config: GeneratorConfig) -> pd.DataFrame:
    central_count = max(1, config.eeg_channels // 2)
    rows: list[dict[str, Any]] = []
    feature_index = 0
    for channel_index in range(config.eeg_channels):
        group = (
            "central"
            if channel_index < central_count
            else "peripheral_frontal_temporal"
        )
        for band in config.band_labels:
            rows.append(
                {
                    "feature_index": feature_index,
                    "channel": f"CH{channel_index + 1:02d}",
                    "channel_index": channel_index,
                    "channel_group": group,
                    "band": band,
                }
            )
            feature_index += 1
    return pd.DataFrame(rows)


def _profiles(metadata: pd.DataFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cortical_band = {"low": 0.85, "mu": 1.30, "beta": 1.15, "high": 0.40}
    artifact_band = {"low": 0.35, "mu": 0.55, "beta": 1.05, "high": 1.70}
    cortical = np.array(
        [
            (1.35 if row.channel_group == "central" else 0.55)
            * cortical_band.get(row.band, 0.8)
            for row in metadata.itertuples()
        ],
        dtype=float,
    )
    artifact = np.array(
        [
            (0.35 if row.channel_group == "central" else 1.45)
            * artifact_band.get(row.band, 1.0)
            for row in metadata.itertuples()
        ],
        dtype=float,
    )
    return cortical, artifact


def _profiled_matrix(
    rng: np.random.Generator,
    profile: NDArray[np.float64],
    columns: int,
    *,
    gain: float = 1.0,
) -> NDArray[np.float64]:
    matrix = rng.normal(size=(len(profile), columns)) * profile[:, None]
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return gain * matrix / np.maximum(norms, 1e-12)


def _emg_projections(dimensions: int) -> dict[str, NDArray[np.float64]]:
    rich = np.eye(dimensions)
    intermediate_dimensions = min(3, dimensions)
    intermediate = np.zeros((intermediate_dimensions, dimensions), dtype=float)
    for channel in range(dimensions):
        intermediate[channel % intermediate_dimensions, channel] = 1.0
    intermediate /= np.maximum(intermediate.sum(axis=1, keepdims=True), 1.0)
    sparse_weights = np.linspace(1.0, -0.35, dimensions, dtype=float)
    sparse_weights /= np.linalg.norm(sparse_weights)
    sparse = sparse_weights.reshape(1, -1)
    return {"rich": rich, "intermediate": intermediate, "sparse": sparse}


def _base_slices(config: GeneratorConfig) -> tuple[dict[str, slice], int]:
    start = 0
    slices: dict[str, slice] = {}
    for name, width in (
        ("cortical", config.eeg_dimensions),
        ("emg_signal", config.emg_channels),
        ("emg_noise", config.emg_channels),
        ("cranial_signal", config.cranial_dimensions),
        ("cranial_noise", config.cranial_dimensions),
        ("eeg_noise", config.eeg_dimensions),
    ):
        slices[name] = slice(start, start + width)
        start += width
    return slices, start


def _model_keys(config: GeneratorConfig) -> tuple[str, ...]:
    if config.mode == "debug" and config.subject_variability_scale == 0 and config.block_drift_scale == 0:
        return ("shared",)
    return tuple(
        f"subject_{subject:02d}_block_{block:02d}"
        for subject in range(config.n_subjects)
        for block in range(config.blocks_per_subject)
    )


def build_oracle_model(
    config: GeneratorConfig,
    scenario: str,
    window: str,
    *,
    seed: int,
) -> LinearGaussianOracle:
    """Build the exact class-conditional law before drawing evaluation samples."""

    config.validate()
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose one of {SCENARIOS}")
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}")

    metadata = eeg_feature_metadata(config)
    cortical_profile, artifact_profile = _profiles(metadata)
    projections = _emg_projections(config.emg_channels)
    slices, base_dim = _base_slices(config)
    codes = _class_codes(config.n_grasp_types)
    strengths = dict(SCENARIO_STRENGTHS[scenario])
    for field_name, strength_name in (
        ("cortical_strength", "cortical"),
        ("artifact_strength", "artifact"),
        ("shared_eeg_strength", "shared_eeg"),
    ):
        override = getattr(config, field_name)
        if override is not None:
            strengths[strength_name] = float(override)
    emg_window_factor = config.pre_emg_fraction if window == "pre" else 1.0
    artifact_window_factor = config.pre_artifact_fraction if window == "pre" else 1.0
    shared_window_factor = 0.0 if window == "pre" else 1.0

    keys = _model_keys(config)
    base_means: dict[str, NDArray[np.float64]] = {}
    base_covariances: dict[str, NDArray[np.float64]] = {}
    artifact_weights: dict[str, NDArray[np.float64]] = {}
    redundant_weights: dict[str, NDArray[np.float64] | None] = {}
    priors: dict[str, NDArray[np.float64]] = {}
    key_weights = {key: 1.0 / len(keys) for key in keys}
    key_seeds: dict[str, int] = {}

    for key in keys:
        key_seed = derive_seed(seed, "oracle-parameters", scenario, window, key)
        key_seeds[key] = key_seed
        rng = np.random.default_rng(key_seed)
        variability = config.subject_variability_scale if config.mode == "stress" else 0.0
        gain = float(np.exp(rng.normal(0.0, variability))) if variability else 1.0

        w_shared_eeg = _profiled_matrix(rng, cortical_profile, 2, gain=gain)
        w_unique_eeg = _profiled_matrix(rng, cortical_profile, 2, gain=gain)
        w_artifact = _profiled_matrix(rng, artifact_profile, config.cranial_dimensions, gain=gain)
        v_shared_emg = _profiled_matrix(rng, np.ones(config.emg_channels), 2, gain=gain)
        v_unique_emg = _profiled_matrix(rng, np.ones(config.emg_channels), 2, gain=gain)
        w_redundant = _profiled_matrix(
            rng, np.ones(config.eeg_dimensions), config.emg_channels, gain=0.85
        )

        sigma_l = 0.55
        sigma_ue = 0.55
        sigma_um = 0.60
        sigma_r = 0.50
        emg_process_noise = 0.18 * gain
        cortical_noise = config.cortical_noise * gain
        emg_noise = config.emg_sensor_noise * gain
        eeg_noise = config.eeg_sensor_noise * gain
        cranial_noise = config.cranial_noise * gain

        class_means = np.zeros((config.n_grasp_types, base_dim), dtype=float)
        class_covariances = np.zeros(
            (config.n_grasp_types, base_dim, base_dim), dtype=float
        )

        shared_eeg_gain = strengths["shared_eeg"] * shared_window_factor
        for class_index, code in enumerate(codes):
            mu_l = config.shared_motor_strength * shared_window_factor * code
            mu_ue = strengths["cortical"] * code
            mu_um = config.emg_unique_strength * emg_window_factor * code
            mu_r = strengths["artifact"] * artifact_window_factor * code[
                : config.cranial_dimensions
            ]
            if config.cranial_dimensions > 2:
                mu_r = np.pad(mu_r, (0, config.cranial_dimensions - 2))

            cortical_mean = shared_eeg_gain * (w_shared_eeg @ mu_l) + (
                w_unique_eeg @ mu_ue
            )
            emg_signal_mean = v_shared_emg @ mu_l + v_unique_emg @ mu_um
            cranial_signal_mean = mu_r

            if config.block_drift_scale and key != "shared":
                drift_rng = np.random.default_rng(derive_seed(key_seed, "drift"))
                cortical_mean = cortical_mean + drift_rng.normal(
                    0.0, config.block_drift_scale, config.eeg_dimensions
                )
                emg_signal_mean = emg_signal_mean + drift_rng.normal(
                    0.0, config.block_drift_scale, config.emg_channels
                )

            class_means[class_index, slices["cortical"]] = cortical_mean
            class_means[class_index, slices["emg_signal"]] = emg_signal_mean
            class_means[class_index, slices["cranial_signal"]] = cranial_signal_mean

            covariance = np.zeros((base_dim, base_dim), dtype=float)
            covariance[slices["cortical"], slices["cortical"]] = (
                (shared_eeg_gain**2 * sigma_l**2)
                * (w_shared_eeg @ w_shared_eeg.T)
                + sigma_ue**2 * (w_unique_eeg @ w_unique_eeg.T)
                + cortical_noise**2 * np.eye(config.eeg_dimensions)
            )
            covariance[slices["emg_signal"], slices["emg_signal"]] = (
                sigma_l**2 * (v_shared_emg @ v_shared_emg.T)
                + sigma_um**2 * (v_unique_emg @ v_unique_emg.T)
                + emg_process_noise**2 * np.eye(config.emg_channels)
            )
            cross = (
                shared_eeg_gain
                * sigma_l**2
                * (w_shared_eeg @ v_shared_emg.T)
            )
            covariance[slices["cortical"], slices["emg_signal"]] = cross
            covariance[slices["emg_signal"], slices["cortical"]] = cross.T
            covariance[slices["emg_noise"], slices["emg_noise"]] = (
                emg_noise**2 * np.eye(config.emg_channels)
            )
            covariance[slices["cranial_signal"], slices["cranial_signal"]] = (
                sigma_r**2 * np.eye(config.cranial_dimensions)
            )
            covariance[slices["cranial_noise"], slices["cranial_noise"]] = (
                cranial_noise**2 * np.eye(config.cranial_dimensions)
            )
            covariance[slices["eeg_noise"], slices["eeg_noise"]] = (
                eeg_noise**2 * np.eye(config.eeg_dimensions)
            )
            class_covariances[class_index] = covariance

        if scenario in {"null", "artifact_only"}:
            # Prevent noise-cancellation complementarity from a shared latent: in
            # these worlds K is deliberately independent of observed M.
            for class_index in range(config.n_grasp_types):
                cross = class_covariances[
                    class_index, slices["cortical"], slices["emg_signal"]
                ]
                cross[...] = 0.0
                class_covariances[
                    class_index, slices["emg_signal"], slices["cortical"]
                ] = 0.0

        if config.mild_class_imbalance > 0 and config.mode == "stress":
            logits = rng.normal(0.0, config.mild_class_imbalance, config.n_grasp_types)
            probability = np.exp(logits - logits.max())
            probability /= probability.sum()
        else:
            probability = np.full(config.n_grasp_types, 1.0 / config.n_grasp_types)

        base_means[key] = class_means
        base_covariances[key] = class_covariances
        artifact_weights[key] = w_artifact
        redundant_weights[key] = w_redundant if scenario == "redundant" else None
        priors[key] = probability

    manifest = {
        "generator": "linear_gaussian_feature_level_v1",
        "seed": seed,
        "scenario": scenario,
        "window": window,
        "config": asdict(config),
        "scenario_strengths": strengths,
        "emg_window_factor": emg_window_factor,
        "artifact_window_factor": artifact_window_factor,
        "shared_window_factor": shared_window_factor,
        "key_parameter_seeds": key_seeds,
        "cortical_spatial_band_profile": cortical_profile.tolist(),
        "myogenic_spatial_band_profile": artifact_profile.tolist(),
        "emg_projection_matrices": {
            name: matrix.tolist() for name, matrix in projections.items()
        },
        "oracle_quantity": (
            "conditioned on known subject/block metadata"
            if len(keys) > 1
            else "debug distribution without subject/block drift"
        ),
    }
    return LinearGaussianOracle(
        scenario=scenario,
        window=window,
        n_classes=config.n_grasp_types,
        keys=keys,
        key_weights=key_weights,
        priors=priors,
        base_means=base_means,
        base_covariances=base_covariances,
        base_slices=slices,
        artifact_weights=artifact_weights,
        redundant_weights=redundant_weights,
        emg_projections=projections,
        eeg_dimensions=config.eeg_dimensions,
        emg_dimensions=config.emg_channels,
        cranial_dimensions=config.cranial_dimensions,
        parameter_manifest=manifest,
    )


def _balanced_labels(n_trials: int, n_classes: int, rng: np.random.Generator) -> NDArray[np.int64]:
    labels = np.resize(np.arange(n_classes, dtype=np.int64), n_trials)
    rng.shuffle(labels)
    return labels


def generate_causal_dataset(
    config: GeneratorConfig,
    scenario: str,
    window: str,
    *,
    seed: int = 0,
) -> SyntheticDataset:
    """Draw a small estimator-evaluation sample and retain every true component."""

    model = build_oracle_model(config, scenario, window, seed=seed)
    rng = np.random.default_rng(derive_seed(seed, "evaluation-sample", scenario, window))
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    oracle_keys: list[str] = []

    for subject in range(config.n_subjects):
        subject_rng = np.random.default_rng(derive_seed(seed, "labels", subject, window))
        if config.mode == "debug" or config.mild_class_imbalance == 0:
            subject_labels = _balanced_labels(
                config.trials_per_subject, config.n_grasp_types, subject_rng
            )
        else:
            subject_labels = np.empty(config.trials_per_subject, dtype=np.int64)
            for trial in range(config.trials_per_subject):
                block = trial % config.blocks_per_subject
                key = f"subject_{subject:02d}_block_{block:02d}"
                subject_labels[trial] = subject_rng.choice(
                    config.n_grasp_types, p=model.priors[key]
                )
        for trial in range(config.trials_per_subject):
            block = trial % config.blocks_per_subject
            key = (
                "shared"
                if model.keys == ("shared",)
                else f"subject_{subject:02d}_block_{block:02d}"
            )
            rows.append(
                {
                    "subject_id": subject,
                    "block_id": block,
                    "trial_id": trial,
                    "trial_uid": f"s{subject:02d}_t{trial:04d}",
                    "block_uid": f"s{subject:02d}_b{block:02d}",
                    "window_id": 0 if window == "pre" else 1,
                    "window": window,
                    "oracle_key": key,
                    "target_name": config.target_name,
                }
            )
            labels.append(int(subject_labels[trial]))
            oracle_keys.append(key)

    metadata = pd.DataFrame(rows)
    g = np.asarray(labels, dtype=np.int64)
    base_dim = next(iter(model.base_means.values())).shape[1]
    base = np.empty((len(g), base_dim), dtype=float)
    oracle_key_array = np.asarray(oracle_keys)
    for key in model.keys:
        key_positions = np.flatnonzero(oracle_key_array == key)
        for class_index in range(config.n_grasp_types):
            positions = key_positions[g[key_positions] == class_index]
            if len(positions) == 0:
                continue
            chol = np.linalg.cholesky(model.base_covariances[key][class_index])
            noise = rng.standard_normal((len(positions), base_dim))
            base[positions] = noise @ chol.T + model.base_means[key][class_index]

    cortical = base[:, model.base_slices["cortical"]]
    emg_signal = base[:, model.base_slices["emg_signal"]]
    emg_noise = base[:, model.base_slices["emg_noise"]]
    cranial_signal = base[:, model.base_slices["cranial_signal"]]
    cranial_noise = base[:, model.base_slices["cranial_noise"]]
    eeg_noise = base[:, model.base_slices["eeg_noise"]]
    emg = emg_signal + emg_noise
    cranial = cranial_signal + cranial_noise

    artifact_signal_eeg = np.empty_like(cortical)
    artifact_noise_eeg = np.empty_like(cortical)
    eeg = np.empty_like(cortical)
    for key in model.keys:
        positions = np.flatnonzero(oracle_key_array == key)
        weights = model.artifact_weights[key]
        artifact_signal_eeg[positions] = cranial_signal[positions] @ weights.T
        artifact_noise_eeg[positions] = cranial_noise[positions] @ weights.T
        redundant = model.redundant_weights[key]
        if redundant is not None:
            eeg[positions] = emg[positions] @ redundant.T + eeg_noise[positions]
        else:
            eeg[positions] = (
                cortical[positions]
                + artifact_signal_eeg[positions]
                + artifact_noise_eeg[positions]
                + eeg_noise[positions]
            )

    proxy_quality = config.default_proxy_quality
    proxy = np.empty_like(cranial)
    for key in model.keys:
        positions = np.flatnonzero(oracle_key_array == key)
        global_mean, std = model._cranial_standardization(key)
        standardized = (cranial[positions] - global_mean) / std
        proxy[positions] = proxy_quality * standardized + np.sqrt(
            1.0 - proxy_quality**2
        ) * rng.standard_normal(standardized.shape)

    parameters = dict(model.parameter_manifest)
    parameters["oracle_model_fingerprint"] = model.fingerprint()
    parameters["evaluation_sample_seed"] = derive_seed(
        seed, "evaluation-sample", scenario, window
    )
    return SyntheticDataset(
        g=g,
        eeg=eeg,
        emg=emg,
        emg_signal=emg_signal,
        emg_noise=emg_noise,
        cortical=cortical,
        cranial=cranial,
        cranial_signal=cranial_signal,
        cranial_noise=cranial_noise,
        eeg_noise=eeg_noise,
        artifact_signal_eeg=artifact_signal_eeg,
        artifact_noise_eeg=artifact_noise_eeg,
        myogenic_proxy=proxy,
        metadata=metadata,
        eeg_feature_metadata=eeg_feature_metadata(config),
        emg_projections=model.emg_projections,
        oracle_model=model,
        scenario=scenario,
        window=window,
        seed=seed,
        parameters=parameters,
    )
