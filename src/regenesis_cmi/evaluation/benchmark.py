"""End-to-end synthetic benchmark runner and auditable result writer."""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from regenesis_cmi import __version__
from regenesis_cmi.information.decoder import estimate_decoder_cmi
from regenesis_cmi.information.discrete import (
    cmi_identities,
    conditional_mutual_information,
    estimate_discrete_cmi,
    mutual_information,
    validate_entropy_sanity,
)
from regenesis_cmi.information.gaussian import (
    gaussian_benchmark_covariance,
    gaussian_cmi_from_covariance,
    gaussian_cmi_monte_carlo,
    sample_gaussian_benchmark,
)
from regenesis_cmi.information.interface import EstimatorResult
from regenesis_cmi.information.mixed import (
    estimate_continuous_cmi,
    estimate_mixed_cmi,
)
from regenesis_cmi.synthetic.causal_generator import generate_causal_dataset
from regenesis_cmi.synthetic.degradation import degrade
from regenesis_cmi.synthetic.exact_worlds import (
    ExactWorld,
    asymmetric_exact,
    complementary_exact,
    complementary_sampled,
    noisy_complementary_exact,
    redundant_markov_exact,
)
from regenesis_cmi.synthetic.oracle import (
    LinearGaussianOracle,
    OracleEstimate,
    discrete_pmf_cmi_oracle,
    monte_carlo_cmi,
)
from regenesis_cmi.synthetic.schema import GeneratorConfig, SCENARIOS
from regenesis_cmi.synthetic.surrogates import (
    discriminating_cortical_surrogate,
    grouped_label_shuffle,
)
from regenesis_cmi.utils.seeds import derive_seed

from .metrics import add_error_columns, estimator_scorecard
from .plots import generate_required_figures


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Benchmark configuration must be a YAML mapping")
    return config


def expand_seeds(specification: Any) -> list[int]:
    if isinstance(specification, int):
        return list(range(specification))
    if isinstance(specification, list):
        return [int(value) for value in specification]
    if isinstance(specification, dict):
        start = int(specification.get("start", 0))
        count = int(specification["count"])
        step = int(specification.get("step", 1))
        return [start + step * index for index in range(count)]
    raise ValueError("Seeds must be a list, count, or {start,count,step} mapping")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unborn-or-unavailable"
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _package_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "PyYAML",
        "pytest",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _estimator_row(
    result: EstimatorResult,
    *,
    world: str,
    oracle_bits: float,
    window: str = "not_applicable",
    subject_id: int | str = "pooled",
    sample_size: int | None = None,
    benchmark_cell: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "world": world,
        "window": window,
        "direction": result.direction,
        "estimator_name": result.estimator_name,
        "estimate_bits": result.estimate_bits,
        "oracle_cmi_bits": oracle_bits,
        "sample_size": result.sample_size if sample_size is None else sample_size,
        "seed": result.seed,
        "subject_id": subject_id,
        "runtime_seconds": result.runtime_seconds,
        "warnings": ";".join(result.warnings),
        "diagnostics": json.dumps(result.diagnostics, default=_json_default, sort_keys=True),
        "units": result.units,
        "benchmark_cell": benchmark_cell
        or f"{world}|{window}|{result.direction}|n{result.sample_size}",
    }
    row.update(extra)
    return row


def _oracle_row(
    result: OracleEstimate,
    *,
    world: str,
    window: str,
    seed: int,
    sample_size: int,
    subject_id: int | str = "pooled",
    runtime_seconds: float = 0.0,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "world": world,
        "window": window,
        "direction": result.direction,
        "estimator_name": "bayes_oracle",
        "estimate_bits": result.estimate_bits,
        "oracle_cmi_bits": result.estimate_bits,
        "oracle_standard_error_bits": result.standard_error_bits,
        "oracle_sample_size": result.sample_size,
        "oracle_precision_achieved": result.precision_achieved,
        "sample_size": sample_size,
        "seed": seed,
        "subject_id": subject_id,
        "runtime_seconds": runtime_seconds,
        "warnings": "" if result.precision_achieved else "oracle_precision_cap_reached",
        "diagnostics": json.dumps(result.diagnostics, default=_json_default, sort_keys=True),
        "units": "bits per independent trial",
        "benchmark_cell": f"{world}|{window}|{result.direction}|n{sample_size}",
    }
    row.update(extra)
    return row


def _run_oracle(
    model: LinearGaussianOracle,
    settings: dict[str, Any],
    *,
    seed: int,
    **kwargs: Any,
) -> tuple[OracleEstimate, float]:
    started = time.perf_counter()
    result = monte_carlo_cmi(
        model,
        seed=seed,
        precision_bits=float(settings.get("precision_bits", 1e-3)),
        min_samples=int(settings.get("min_samples", 10_000)),
        max_samples=int(settings.get("max_samples", 200_000)),
        batch_size=int(settings.get("batch_size", 5_000)),
        **kwargs,
    )
    return result, time.perf_counter() - started


def validate_exact_worlds() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entropy = pd.DataFrame(validate_entropy_sanity())
    rows: list[dict[str, Any]] = []
    worlds = [
        complementary_exact(25),
        *(noisy_complementary_exact(q) for q in (0.00, 0.05, 0.11, 0.20, 0.30, 0.40, 0.50)),
        redundant_markov_exact(),
        asymmetric_exact(),
    ]
    for world in worlds:
        for direction, added, conditioned, truth in (
            ("I(G;E|M)", world.eeg, world.emg, world.truth_forward_bits),
            ("I(G;M|E)", world.emg, world.eeg, world.truth_reverse_bits),
        ):
            if np.isnan(truth):
                truth = conditional_mutual_information(world.g, added, conditioned)
            result = estimate_discrete_cmi(
                world.g, added, conditioned, direction=direction, seed=0
            )
            pmf_oracle = discrete_pmf_cmi_oracle(
                world.g,
                added,
                conditioned,
                n_samples=20_000,
                seed=derive_seed(71, world.name, direction),
                direction=direction,
            )
            rows.append(
                _estimator_row(
                    result,
                    world=world.name,
                    oracle_bits=float(truth),
                    benchmark_cell=f"{world.name}|{direction}|deterministic",
                    ordinary_mi_bits=mutual_information(world.g, added),
                    pmf_oracle_monte_carlo_bits=pmf_oracle.estimate_bits,
                    pmf_oracle_standard_error_bits=pmf_oracle.standard_error_bits,
                    expected_behavior=world.description,
                )
            )
    exact = add_error_columns(pd.DataFrame(rows))
    covariance = gaussian_benchmark_covariance()
    analytic = gaussian_cmi_from_covariance(covariance, [0], [1], [2])
    monte_carlo = gaussian_cmi_monte_carlo(covariance, n_samples=100_000, seed=71)
    gaussian = pd.DataFrame(
        [
            {
                "benchmark": "generic_scalar_X_Y_given_Z",
                "analytic_bits": analytic,
                "monte_carlo_bits": monte_carlo["estimate_bits"],
                "monte_carlo_standard_error_bits": monte_carlo[
                    "standard_error_bits"
                ],
                "sample_size": monte_carlo["sample_size"],
                "absolute_error_bits": abs(float(monte_carlo["estimate_bits"]) - analytic),
                "units": "bits per independent sample",
            }
        ]
    )
    return entropy, exact, gaussian


def _sample_exact(world: ExactWorld, n_samples: int, seed: int) -> ExactWorld:
    indices = np.random.default_rng(seed).choice(len(world.g), size=n_samples, replace=True)
    return ExactWorld(
        name=world.name,
        g=world.g[indices],
        eeg=world.eeg[indices],
        emg=world.emg[indices],
        truth_forward_bits=world.truth_forward_bits,
        truth_reverse_bits=world.truth_reverse_bits,
        description=world.description,
    )


def _run_exact_samples(config: dict[str, Any], seeds: list[int]) -> pd.DataFrame:
    sample_sizes = [int(value) for value in config["exact"]["sample_sizes"]]
    worlds = [
        complementary_exact(25),
        noisy_complementary_exact(0.11),
        noisy_complementary_exact(0.50),
        redundant_markov_exact(),
        asymmetric_exact(),
    ]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for sample_size in sample_sizes:
            for world in worlds:
                sampled = _sample_exact(
                    world, sample_size, derive_seed(seed, world.name, sample_size)
                )
                for direction, added, conditioned, truth in (
                    ("I(G;E|M)", sampled.eeg, sampled.emg, world.truth_forward_bits),
                    ("I(G;M|E)", sampled.emg, sampled.eeg, world.truth_reverse_bits),
                ):
                    if np.isnan(truth):
                        truth = conditional_mutual_information(
                            world.g,
                            world.emg if direction == "I(G;M|E)" else world.eeg,
                            world.eeg if direction == "I(G;M|E)" else world.emg,
                        )
                    result = estimate_discrete_cmi(
                        sampled.g,
                        added,
                        conditioned,
                        direction=direction,
                        seed=seed,
                    )
                    rows.append(
                        _estimator_row(
                            result,
                            world=world.name,
                            oracle_bits=float(truth),
                            sample_size=sample_size,
                            benchmark_cell=f"{world.name}|{direction}|n{sample_size}",
                        )
                    )
    return add_error_columns(pd.DataFrame(rows))


def _run_dimension_benchmark(config: dict[str, Any], seeds: list[int]) -> pd.DataFrame:
    dimensions = [int(value) for value in config["dimension_benchmark"]["dimensions"]]
    sample_size = int(config["dimension_benchmark"].get("sample_size", 1_000))
    covariance = gaussian_benchmark_covariance()
    truth = gaussian_cmi_from_covariance(covariance, [0], [1], [2])
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        x_signal, y_signal, z_signal = sample_gaussian_benchmark(
            sample_size, seed=derive_seed(seed, "dimension"), covariance=covariance
        )
        rng = np.random.default_rng(derive_seed(seed, "dimension-noise"))
        for dimension in dimensions:
            extra = dimension - 1
            x = np.column_stack((x_signal, rng.standard_normal((sample_size, extra))))
            y = np.column_stack((y_signal, rng.standard_normal((sample_size, extra))))
            z = np.column_stack((z_signal, rng.standard_normal((sample_size, extra))))
            result = estimate_continuous_cmi(
                x, y, z, seed=seed, k_fraction=float(config["dimension_benchmark"].get("k_fraction", 0.1))
            )
            row = _estimator_row(
                result,
                world="generic_gaussian_X_Y_given_Z",
                oracle_bits=truth,
                sample_size=sample_size,
                benchmark_cell=f"gaussian|dimension{dimension}|n{sample_size}",
                x_dimension=dimension,
                y_dimension=dimension,
                conditioning_dimension=dimension,
                total_dimension=3 * dimension,
            )
            rows.append(row)
    return add_error_columns(pd.DataFrame(rows))


def _run_causal_sample_sizes(
    config: dict[str, Any],
    generator_config: GeneratorConfig,
    seeds: list[int],
) -> pd.DataFrame:
    settings = config.get("sample_size_benchmark", {})
    if not settings:
        return pd.DataFrame()
    sample_sizes = [int(value) for value in settings["sample_sizes"]]
    worlds = [str(value) for value in settings.get("worlds", ["null", "genuine_cortical"])]
    window = str(settings.get("window", "post"))
    n_subjects = int(settings.get("n_subjects", 1))
    causal = config["causal"]
    decoder_settings = causal["decoder"]
    oracle_settings = causal["oracle"]
    direct_eeg_dim = int(causal.get("direct_eeg_dimensions", 6))
    direct_emg_dim = int(causal.get("direct_emg_dimensions", 3))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for sample_size in sample_sizes:
            per_subject = int(math.ceil(sample_size / n_subjects))
            benchmark_config = replace(
                generator_config,
                n_subjects=n_subjects,
                trials_per_subject=per_subject,
                blocks_per_subject=min(
                    generator_config.blocks_per_subject,
                    max(4, per_subject // (2 * generator_config.n_grasp_types)),
                ),
            )
            for world in worlds:
                dataset = generate_causal_dataset(
                    benchmark_config,
                    world,
                    window,
                    seed=derive_seed(seed, "sample-size", world),
                )
                actual_size = len(dataset.g)
                oracle, _ = _run_oracle(
                    dataset.oracle_model,
                    oracle_settings,
                    seed=derive_seed(seed, "sample-size-oracle", world),
                    direction="I(G;E|M)",
                )
                decoder = estimate_decoder_cmi(
                    dataset.g,
                    dataset.eeg,
                    dataset.emg,
                    groups=dataset.metadata["block_uid"],
                    direction="I(G;E|M)",
                    seed=seed,
                    outer_splits=int(decoder_settings.get("outer_splits", 3)),
                    inner_splits=int(decoder_settings.get("inner_splits", 2)),
                    c_grid=tuple(
                        float(value)
                        for value in decoder_settings.get("c_grid", [0.1, 1.0])
                    ),
                )
                mixed = estimate_mixed_cmi(
                    dataset.g,
                    dataset.eeg[:, : min(direct_eeg_dim, dataset.eeg.shape[1])],
                    dataset.emg[:, : min(direct_emg_dim, dataset.emg.shape[1])],
                    direction="I(G;E|M)",
                    seed=seed,
                    k_fraction=float(causal.get("mixed_k_fraction", 0.1)),
                )
                for estimator in (decoder, mixed):
                    rows.append(
                        _estimator_row(
                            estimator,
                            world=world,
                            oracle_bits=oracle.estimate_bits,
                            window=window,
                            sample_size=actual_size,
                            benchmark_cell=f"sample-size|{world}|n{actual_size}",
                            oracle_standard_error_bits=oracle.standard_error_bits,
                            benchmark_family="causal_sample_size",
                        )
                    )
    return add_error_columns(pd.DataFrame(rows))


def _run_causal_core(
    config: dict[str, Any],
    generator_config: GeneratorConfig,
    seeds: list[int],
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], Any]]:
    causal = config["causal"]
    worlds = causal.get("worlds", list(SCENARIOS))
    windows = causal.get("windows", ["pre", "post"])
    oracle_settings = causal["oracle"]
    decoder_settings = causal["decoder"]
    direct_eeg_dim = int(causal.get("direct_eeg_dimensions", 6))
    direct_emg_dim = int(causal.get("direct_emg_dimensions", 3))
    rows: list[dict[str, Any]] = []
    datasets: dict[tuple[str, str, int], Any] = {}

    for seed in seeds:
        for world in worlds:
            for window in windows:
                dataset = generate_causal_dataset(
                    generator_config, world, window, seed=seed
                )
                datasets[(world, window, seed)] = dataset
                groups = dataset.metadata["block_uid"].to_numpy()
                eeg_direct = dataset.eeg[:, : min(direct_eeg_dim, dataset.eeg.shape[1])]
                emg_direct = dataset.emg[:, : min(direct_emg_dim, dataset.emg.shape[1])]
                for direction in ("I(G;E|M)", "I(G;M|E)"):
                    oracle, oracle_runtime = _run_oracle(
                        dataset.oracle_model,
                        oracle_settings,
                        seed=derive_seed(seed, "oracle", world, window, direction),
                        direction=direction,
                    )
                    rows.append(
                        _oracle_row(
                            oracle,
                            world=world,
                            window=window,
                            seed=seed,
                            sample_size=len(dataset.g),
                            runtime_seconds=oracle_runtime,
                            montage="full",
                            band="all",
                            richness="rich",
                            control="none",
                        )
                    )
                    if direction == "I(G;E|M)":
                        added, conditioned = dataset.eeg, dataset.emg
                        added_direct, conditioned_direct = eeg_direct, emg_direct
                    else:
                        added, conditioned = dataset.emg, dataset.eeg
                        added_direct, conditioned_direct = emg_direct, eeg_direct
                    decoder = estimate_decoder_cmi(
                        dataset.g,
                        added,
                        conditioned,
                        groups=groups,
                        direction=direction,
                        seed=seed,
                        outer_splits=int(decoder_settings.get("outer_splits", 3)),
                        inner_splits=int(decoder_settings.get("inner_splits", 2)),
                        c_grid=tuple(float(value) for value in decoder_settings.get("c_grid", [0.1, 1.0, 10.0])),
                    )
                    mixed = estimate_mixed_cmi(
                        dataset.g,
                        added_direct,
                        conditioned_direct,
                        direction=direction,
                        seed=seed,
                        k_fraction=float(causal.get("mixed_k_fraction", 0.1)),
                    )
                    for estimator in (decoder, mixed):
                        rows.append(
                            _estimator_row(
                                estimator,
                                world=world,
                                oracle_bits=oracle.estimate_bits,
                                window=window,
                                montage="full",
                                band="all",
                                richness="rich",
                                control="none",
                                oracle_standard_error_bits=oracle.standard_error_bits,
                                benchmark_cell=f"{world}|{window}|{direction}|n{len(dataset.g)}",
                            )
                        )
    return add_error_columns(pd.DataFrame(rows)), datasets


def _subject_rows(
    result: OracleEstimate,
    *,
    generator_config: GeneratorConfig,
    world: str,
    window: str,
    seed: int,
    model: LinearGaussianOracle | None = None,
    oracle_settings: dict[str, Any] | None = None,
    oracle_kwargs: dict[str, Any] | None = None,
    **extra: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identical = model is None or model.keys == ("shared",)
    for subject in range(generator_config.n_subjects):
        subject_result = result
        if not identical:
            subject_keys = tuple(
                key
                for key in model.keys
                if key.startswith(f"subject_{subject:02d}_")
            )
            subject_result, _ = _run_oracle(
                model,
                oracle_settings or {},
                seed=derive_seed(seed, "subject-oracle", subject, *extra.values()),
                oracle_keys=subject_keys,
                **(oracle_kwargs or {}),
            )
        rows.append(
            {
            "world": world,
            "window": window,
            "direction": subject_result.direction,
            "estimate_bits": subject_result.estimate_bits,
            "oracle_standard_error_bits": subject_result.standard_error_bits,
            "subject_id": subject,
            "seed": seed,
            "subject_parameters_identical": identical,
            "units": "bits per independent trial",
            **extra,
        }
        )
    return rows


def _run_controls(
    config: dict[str, Any],
    generator_config: GeneratorConfig,
    datasets: dict[tuple[str, str, int], Any],
    seed: int,
) -> dict[str, pd.DataFrame]:
    causal = config["causal"]
    settings = causal["oracle"]
    controls = config["controls"]
    honesty_rows: list[dict[str, Any]] = []
    degradation_rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    richness_rows: list[dict[str, Any]] = []
    montage_rows: list[dict[str, Any]] = []
    surrogate_rows: list[dict[str, Any]] = []

    def get_dataset(world: str, window: str = "post"):
        key = (world, window, seed)
        if key not in datasets:
            datasets[key] = generate_causal_dataset(
                generator_config, world, window, seed=seed
            )
        return datasets[key]

    for world in controls["honesty_worlds"]:
        dataset = get_dataset(world)
        conditions = [
            ("raw", {"control": "none", "eeg_variant": "raw"}),
            ("conditioned_true_C", {"control": "true_c", "eeg_variant": "raw"}),
            (
                f"conditioned_proxy_r{controls['default_proxy_quality']}",
                {
                    "control": "proxy",
                    "proxy_quality": float(controls["default_proxy_quality"]),
                    "eeg_variant": "raw",
                },
            ),
            ("ideal_E_minus_WC_C", {"control": "none", "eeg_variant": "clean"}),
        ]
        for label, kwargs in conditions:
            result, _ = _run_oracle(
                dataset.oracle_model,
                settings,
                seed=derive_seed(seed, "honesty", world, label),
                direction="I(G;E|M)",
                **kwargs,
            )
            honesty_rows.extend(
                _subject_rows(
                    result,
                    generator_config=generator_config,
                    world=world,
                    window="post",
                    seed=seed,
                    model=dataset.oracle_model,
                    oracle_settings=settings,
                    oracle_kwargs={"direction": "I(G;E|M)", **kwargs},
                    control=label,
                )
            )

    for world in controls["degradation_worlds"]:
        dataset = get_dataset(world)
        for alpha in (float(value) for value in controls["degradation_levels"]):
            snr = degrade(dataset, alpha).realized_snr_db
            for condition, kwargs in (
                ("raw_EEG", {"control": "none", "eeg_variant": "raw"}),
                ("matched_myogenic", {"control": "none", "eeg_variant": "matched"}),
                ("conditioned_true_C", {"control": "true_c", "eeg_variant": "raw"}),
            ):
                result, _ = _run_oracle(
                    dataset.oracle_model,
                    settings,
                    seed=derive_seed(seed, "degradation", world, alpha, condition),
                    direction="I(G;E|M)",
                    alpha=alpha,
                    **kwargs,
                )
                degradation_rows.extend(
                    _subject_rows(
                        result,
                        generator_config=generator_config,
                        world=world,
                        window="post",
                        seed=seed,
                        model=dataset.oracle_model,
                        oracle_settings=settings,
                        oracle_kwargs={
                            "direction": "I(G;E|M)",
                            "alpha": alpha,
                            **kwargs,
                        },
                        alpha=alpha,
                        condition=condition,
                        realized_emg_snr_db=snr,
                    )
                )

    for world in controls["proxy_worlds"]:
        dataset = get_dataset(world)
        for quality in (float(value) for value in controls["proxy_qualities"]):
            result, _ = _run_oracle(
                dataset.oracle_model,
                settings,
                seed=derive_seed(seed, "proxy", world, quality),
                direction="I(G;E|M)",
                control="proxy",
                proxy_quality=quality,
            )
            proxy_rows.extend(
                _subject_rows(
                    result,
                    generator_config=generator_config,
                    world=world,
                    window="post",
                    seed=seed,
                    model=dataset.oracle_model,
                    oracle_settings=settings,
                    oracle_kwargs={
                        "direction": "I(G;E|M)",
                        "control": "proxy",
                        "proxy_quality": quality,
                    },
                    proxy_quality=quality,
                )
            )

    for world in controls["richness_worlds"]:
        dataset = get_dataset(world)
        for richness in ("sparse", "intermediate", "rich"):
            result, _ = _run_oracle(
                dataset.oracle_model,
                settings,
                seed=derive_seed(seed, "richness", world, richness),
                direction="I(G;E|M)",
                emg_richness=richness,
            )
            richness_rows.extend(
                _subject_rows(
                    result,
                    generator_config=generator_config,
                    world=world,
                    window="post",
                    seed=seed,
                    model=dataset.oracle_model,
                    oracle_settings=settings,
                    oracle_kwargs={
                        "direction": "I(G;E|M)",
                        "emg_richness": richness,
                    },
                    richness=richness,
                )
            )

    for world in controls["montage_band_worlds"]:
        dataset = get_dataset(world)
        conditions = [("full", "all", dataset.eeg_indices(montage="full"))]
        conditions.append(("central", "all", dataset.eeg_indices(montage="central")))
        conditions.extend(
            ("full", band, dataset.eeg_indices(montage="full", band=band))
            for band in generator_config.band_labels
        )
        for montage, band, indices in conditions:
            result, _ = _run_oracle(
                dataset.oracle_model,
                settings,
                seed=derive_seed(seed, "montage-band", world, montage, band),
                direction="I(G;E|M)",
                eeg_indices=indices,
            )
            montage_rows.extend(
                _subject_rows(
                    result,
                    generator_config=generator_config,
                    world=world,
                    window="post",
                    seed=seed,
                    model=dataset.oracle_model,
                    oracle_settings=settings,
                    oracle_kwargs={
                        "direction": "I(G;E|M)",
                        "eeg_indices": indices,
                    },
                    montage=montage,
                    band=band,
                )
            )

    permutations = int(controls.get("surrogate_label_permutations", 9))
    surrogate_trials = int(
        controls.get("surrogate_trials_per_subject", generator_config.trials_per_subject)
    )
    surrogate_config = replace(
        generator_config,
        trials_per_subject=max(generator_config.trials_per_subject, surrogate_trials),
        blocks_per_subject=max(generator_config.blocks_per_subject, 8),
    )
    for world in controls["surrogate_worlds"]:
        if surrogate_config == generator_config:
            dataset = get_dataset(world)
        else:
            dataset = generate_causal_dataset(
                surrogate_config, world, "post", seed=seed
            )
            datasets[(f"{world}_surrogate_diagnostic", "post", seed)] = dataset
        surrogate_eeg = discriminating_cortical_surrogate(
            dataset, seed=derive_seed(seed, "cortical-surrogate", world)
        )
        shuffled_labels = [
            grouped_label_shuffle(
                dataset.g,
                dataset.metadata,
                seed=derive_seed(seed, "label-shuffle", world, permutation),
            )
            for permutation in range(permutations)
        ]
        for subject in range(surrogate_config.n_subjects):
            positions = np.flatnonzero(
                dataset.metadata["subject_id"].to_numpy() == subject
            )
            emg = dataset.emg[positions]
            for label, eeg_values, labels_values, replicate in [
                ("observed", dataset.eeg, dataset.g, 0),
                ("discriminating_cortical_intervention", surrogate_eeg, dataset.g, 0),
                *[
                    ("label_shuffle", dataset.eeg, shuffled, index)
                    for index, shuffled in enumerate(shuffled_labels)
                ],
            ]:
                estimate = estimate_mixed_cmi(
                    labels_values[positions],
                    eeg_values[positions],
                    emg,
                    direction="I(G;E|M)",
                    seed=derive_seed(seed, world, subject, label, replicate),
                    k_fraction=float(causal.get("mixed_k_fraction", 0.1)),
                )
                surrogate_rows.append(
                    {
                        "world": world,
                        "window": "post",
                        "direction": "I(G;E|M)",
                        "surrogate": label,
                        "replicate": replicate,
                        "estimate_bits": estimate.estimate_bits,
                        "estimator_name": estimate.estimator_name,
                        "subject_id": subject,
                        "seed": seed,
                        "units": "bits per independent trial",
                        "conditioning_features": "rich",
                        "eeg_features": "full",
                    }
                )

    return {
        "honesty": pd.DataFrame(honesty_rows),
        "degradation": pd.DataFrame(degradation_rows),
        "proxy": pd.DataFrame(proxy_rows),
        "richness": pd.DataFrame(richness_rows),
        "montage_band": pd.DataFrame(montage_rows),
        "surrogates": pd.DataFrame(surrogate_rows),
    }


def _run_null_calibration(
    config: dict[str, Any], generator_config: GeneratorConfig, seeds: list[int]
) -> pd.DataFrame:
    causal = config["causal"]
    decoder_settings = causal["decoder"]
    rows: list[dict[str, Any]] = []
    direct_eeg_dim = int(causal.get("direct_eeg_dimensions", 6))
    direct_emg_dim = int(causal.get("direct_emg_dimensions", 3))
    for seed in seeds:
        dataset = generate_causal_dataset(
            generator_config, "null", "post", seed=derive_seed(seed, "null-calibration")
        )
        groups = dataset.metadata["block_uid"].to_numpy()
        decoder = estimate_decoder_cmi(
            dataset.g,
            dataset.eeg,
            dataset.emg,
            groups=groups,
            direction="I(G;E|M)",
            seed=seed,
            outer_splits=int(decoder_settings.get("outer_splits", 3)),
            inner_splits=int(decoder_settings.get("inner_splits", 2)),
            c_grid=tuple(float(value) for value in decoder_settings.get("c_grid", [0.1, 1.0])),
        )
        mixed = estimate_mixed_cmi(
            dataset.g,
            dataset.eeg[:, :direct_eeg_dim],
            dataset.emg[:, :direct_emg_dim],
            direction="I(G;E|M)",
            seed=seed,
            k_fraction=float(causal.get("mixed_k_fraction", 0.1)),
        )
        for result in (decoder, mixed):
            rows.append(
                {
                    "world": "null",
                    "estimate_bits": result.estimate_bits,
                    "estimator_name": result.estimator_name,
                    "seed": seed,
                    "sample_size": len(dataset.g),
                    "direction": result.direction,
                    "runtime_seconds": result.runtime_seconds,
                    "units": result.units,
                }
            )
        exact_null = _sample_exact(
            redundant_markov_exact(), len(dataset.g), derive_seed(seed, "exact-null")
        )
        exact_result = estimate_discrete_cmi(
            exact_null.g,
            exact_null.eeg,
            exact_null.emg,
            direction="I(G;E|M)",
            seed=seed,
        )
        rows.append(
            {
                "world": "redundant_markov_chain",
                "estimate_bits": exact_result.estimate_bits,
                "estimator_name": exact_result.estimator_name,
                "seed": seed,
                "sample_size": len(dataset.g),
                "direction": exact_result.direction,
                "runtime_seconds": exact_result.runtime_seconds,
                "units": exact_result.units,
            }
        )
    return pd.DataFrame(rows)


def _scientific_checks(
    exact: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(name: str, observed: str, expected: str, passed: bool) -> None:
        rows.append(
            {
                "check": name,
                "observed": observed,
                "expected": expected,
                "pass": bool(passed),
            }
        )

    complementary = exact[
        (exact["world"] == "complementary_four_class")
        & (exact["direction"] == "I(G;E|M)")
    ].iloc[0]
    add(
        "exact_complementary",
        f"{complementary.estimate_bits:.6f} bits",
        "1 bit",
        abs(complementary.estimate_bits - 1.0) < 1e-12,
    )
    redundant = exact[
        (exact["world"] == "redundant_markov_chain")
        & (exact["direction"] == "I(G;E|M)")
    ].iloc[0]
    add(
        "redundant_markov",
        f"MI={redundant.ordinary_mi_bits:.4f}, CMI={redundant.estimate_bits:.4g}",
        "ordinary MI > 0 and CMI = 0",
        redundant.ordinary_mi_bits > 0 and abs(redundant.estimate_bits) < 1e-12,
    )

    honesty = controls["honesty"].groupby(["world", "control"])["estimate_bits"].mean()
    artifact_raw = float(honesty.loc[("artifact_only", "raw")])
    artifact_controlled = float(honesty.loc[("artifact_only", "conditioned_true_C")])
    genuine_controlled = float(
        honesty.loc[("genuine_cortical", "conditioned_true_C")]
    )
    mixed_raw = float(honesty.loc[("cortical_plus_artifact", "raw")])
    mixed_controlled = float(
        honesty.loc[("cortical_plus_artifact", "conditioned_true_C")]
    )
    add(
        "artifact_only_control",
        f"raw={artifact_raw:.3f}, true-C={artifact_controlled:.3f}",
        "raw positive; true-C near zero",
        artifact_raw > 0.05 and abs(artifact_controlled) < 0.03,
    )
    add(
        "genuine_cortical_control",
        f"true-C={genuine_controlled:.3f}",
        "controlled CMI remains positive",
        genuine_controlled > 0.05,
    )
    add(
        "mixed_control",
        f"raw={mixed_raw:.3f}, true-C={mixed_controlled:.3f}",
        "raw > controlled > 0",
        mixed_raw > mixed_controlled > 0.03,
    )

    degradation = controls["degradation"]
    artifact = degradation[degradation["world"] == "artifact_only"].groupby(
        ["alpha", "condition"]
    )["estimate_bits"].mean()
    levels = sorted(degradation["alpha"].unique())
    low, high = levels[0], levels[-1]
    raw_low = float(artifact.loc[(low, "raw_EEG")])
    raw_high = float(artifact.loc[(high, "raw_EEG")])
    matched_low = float(artifact.loc[(low, "matched_myogenic")])
    add(
        "artifact_degradation_rise",
        f"raw alpha={high:g}: {raw_high:.3f}; alpha={low:g}: {raw_low:.3f}",
        "raw CMI rises as EMG signal is removed",
        raw_low > raw_high + 0.05,
    )
    add(
        "matched_myogenic_degradation",
        f"raw={raw_low:.3f}, matched={matched_low:.3f} at alpha={low:g}",
        "matched control reduces artifact-driven rise",
        matched_low < raw_low - 0.05,
    )

    proxy = controls["proxy"]
    artifact_proxy = proxy[proxy["world"] == "artifact_only"].groupby(
        "proxy_quality"
    )["estimate_bits"].mean()
    add(
        "proxy_quality",
        f"r=1: {artifact_proxy.loc[1.0]:.3f}; r=0: {artifact_proxy.loc[0.0]:.3f}",
        "residual artifact increases as proxy quality falls",
        artifact_proxy.loc[1.0] < artifact_proxy.loc[0.0] - 0.05,
    )

    surrogate = controls["surrogates"].groupby(["world", "surrogate"])[
        "estimate_bits"
    ].mean()
    artifact_observed = float(surrogate.loc[("artifact_only", "observed")])
    artifact_intervention = float(
        surrogate.loc[("artifact_only", "discriminating_cortical_intervention")]
    )
    genuine_observed = float(surrogate.loc[("genuine_cortical", "observed")])
    genuine_intervention = float(
        surrogate.loc[("genuine_cortical", "discriminating_cortical_intervention")]
    )
    add(
        "discriminating_surrogate",
        (
            f"artifact observed/intervened={artifact_observed:.3f}/{artifact_intervention:.3f}; "
            f"genuine={genuine_observed:.3f}/{genuine_intervention:.3f}"
        ),
        "artifact preserved; genuine cortical contribution reduced",
        abs(artifact_observed - artifact_intervention) < 0.25
        and genuine_observed > genuine_intervention + 0.03,
    )
    artifact_shuffle = controls["surrogates"][
        (controls["surrogates"]["world"] == "artifact_only")
        & (controls["surrogates"]["surrogate"] == "label_shuffle")
    ]["estimate_bits"]
    shuffle_threshold = float(np.quantile(artifact_shuffle, 0.95))
    add(
        "label_shuffle_not_origin_discriminating",
        f"artifact observed={artifact_observed:.3f}; label-shuffle q95={shuffle_threshold:.3f}",
        "artifact-only effect exceeds generic label-shuffle null",
        artifact_observed > shuffle_threshold,
    )
    return pd.DataFrame(rows)


def estimate_runtime(config: dict[str, Any]) -> dict[str, Any]:
    seeds = expand_seeds(config.get("seeds", [0]))
    causal_seeds = expand_seeds(config.get("causal", {}).get("seeds", seeds[:1]))
    worlds = len(config.get("causal", {}).get("worlds", SCENARIOS))
    windows = len(config.get("causal", {}).get("windows", ["pre", "post"]))
    core_decoder_evaluations = len(causal_seeds) * worlds * windows * 2
    sample_settings = config.get("sample_size_benchmark", {})
    sample_decoder_evaluations = (
        len(causal_seeds)
        * len(sample_settings.get("sample_sizes", []))
        * len(sample_settings.get("worlds", []))
    )
    null_decoder_evaluations = len(
        expand_seeds(config.get("null_calibration_seeds", []))
    )
    decoder_evaluations = (
        core_decoder_evaluations
        + sample_decoder_evaluations
        + null_decoder_evaluations
    )
    oracle_evaluations = core_decoder_evaluations + sample_decoder_evaluations
    generator = config.get("causal", {}).get("generator", {})
    trials = int(generator.get("n_subjects", 1)) * int(
        generator.get("trials_per_subject", 328)
    )
    dimensions = int(generator.get("eeg_channels", 32)) * len(
        generator.get("band_labels", ["low", "mu", "beta", "high"])
    ) + int(generator.get("emg_channels", 5))
    # Calibrated against one 3,936-trial, 133-dimensional nested decoder and
    # conservatively accounts for the separate full-dimensional Bayes oracles.
    # A small floor avoids implausibly tiny estimates for smoke runs.
    workload_scale = max(0.05, (trials * dimensions) / (3_936 * 133))
    decoder_minutes = decoder_evaluations * 0.15 * workload_scale
    oracle_minutes = oracle_evaluations * 0.12 * workload_scale
    estimated_minutes = decoder_minutes + oracle_minutes
    return {
        "estimated_cpu_minutes": estimated_minutes,
        "decoder_cmi_evaluations": decoder_evaluations,
        "decoder_evaluation_breakdown": {
            "core_both_directions": core_decoder_evaluations,
            "sample_size_sweep": sample_decoder_evaluations,
            "disjoint_seed_null_calibration": null_decoder_evaluations,
        },
        "oracle_evaluations": oracle_evaluations,
        "workload_scale_vs_3936x133_reference": workload_scale,
        "estimated_decoder_minutes": decoder_minutes,
        "estimated_oracle_minutes": oracle_minutes,
        "gpu_required": False,
        "run_recommended_automatically": estimated_minutes <= 30.0,
        "basis": (
            "dimension/sample-scaled nested decoder pilot plus separate adaptive "
            "full-dimensional Bayes-oracle allowance"
        ),
    }


def run_benchmark(
    config_path: str | Path,
    *,
    output_override: str | Path | None = None,
    core_only: bool = False,
) -> Path:
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    root = Path.cwd().resolve()
    output = Path(output_override or config.get("output_dir", f"results/{config.get('name', 'run')}"))
    if not output.is_absolute():
        output = root / output
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seeds = expand_seeds(config.get("seeds", [0]))
    causal_seeds = expand_seeds(config["causal"].get("seeds", seeds[:1]))
    generator_config = GeneratorConfig.from_mapping(config["causal"]["generator"])
    entropy, exact_validation, gaussian_validation = validate_exact_worlds()
    exact_samples = _run_exact_samples(config, seeds)
    dimension = _run_dimension_benchmark(config, seeds)
    sample_size_results = _run_causal_sample_sizes(
        config, generator_config, causal_seeds
    )
    causal_core, datasets = _run_causal_core(
        config, generator_config, causal_seeds
    )
    null_seeds = expand_seeds(config.get("null_calibration_seeds", {"start": 9000, "count": 10}))
    null_calibration = _run_null_calibration(config, generator_config, null_seeds)

    base_tables = {
        "entropy_sanity": entropy,
        "exact_validation": exact_validation,
        "gaussian_validation": gaussian_validation,
        "exact_samples": exact_samples,
        "dimension": dimension,
        "causal_sample_sizes": sample_size_results,
        "causal_core": causal_core,
        "null_calibration": null_calibration,
    }
    for name, frame in base_tables.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False)

    combined_estimates = pd.concat(
        [
            exact_samples,
            sample_size_results,
            causal_core[causal_core["estimator_name"] != "bayes_oracle"],
        ],
        ignore_index=True,
        sort=False,
    )
    scorecard = estimator_scorecard(
        combined_estimates, null_calibration=null_calibration
    )
    scorecard.to_csv(tables_dir / "estimator_scorecard.csv", index=False)

    controls: dict[str, pd.DataFrame] = {}
    if not core_only:
        control_seed = causal_seeds[0]
        controls = _run_controls(
            config, generator_config, datasets, control_seed
        )
        for name, frame in controls.items():
            frame.to_csv(tables_dir / f"{name}.csv", index=False)
        checks = _scientific_checks(exact_validation, controls)
        checks.to_csv(tables_dir / "scientific_checks.csv", index=False)

        oracle_subject_rows: list[dict[str, Any]] = []
        for row in causal_core[causal_core["estimator_name"] == "bayes_oracle"].itertuples():
            dataset = datasets[(row.world, row.window, row.seed)]
            identical = dataset.oracle_model.keys == ("shared",)
            for subject in range(generator_config.n_subjects):
                estimate_bits = row.estimate_bits
                standard_error_bits = row.oracle_standard_error_bits
                if not identical:
                    subject_keys = tuple(
                        key
                        for key in dataset.oracle_model.keys
                        if key.startswith(f"subject_{subject:02d}_")
                    )
                    subject_result, _ = _run_oracle(
                        dataset.oracle_model,
                        config["causal"]["oracle"],
                        seed=derive_seed(
                            row.seed,
                            "subject-core-oracle",
                            row.world,
                            row.window,
                            row.direction,
                            subject,
                        ),
                        direction=row.direction,
                        oracle_keys=subject_keys,
                    )
                    estimate_bits = subject_result.estimate_bits
                    standard_error_bits = subject_result.standard_error_bits
                oracle_subject_rows.append(
                    {
                        "world": row.world,
                        "window": row.window,
                        "direction": row.direction,
                        "estimate_bits": estimate_bits,
                        "oracle_standard_error_bits": standard_error_bits,
                        "subject_id": subject,
                        "subject_parameters_identical": identical,
                        "seed": row.seed,
                        "units": "bits per independent trial",
                    }
                )
        subject_oracles = pd.DataFrame(oracle_subject_rows)
        subject_oracles.to_csv(tables_dir / "subject_oracles.csv", index=False)
        plot_tables = {
            "estimated_vs_true": combined_estimates,
            "bias_sample": pd.concat(
                [exact_samples, sample_size_results], ignore_index=True, sort=False
            ),
            "null": null_calibration,
            "dimension": dimension,
            "directions": subject_oracles[subject_oracles["window"] == "post"],
            "honesty": controls["honesty"],
            "degradation": controls["degradation"],
            "proxy": controls["proxy"],
            "richness": controls["richness"],
            "pre_post": subject_oracles[
                subject_oracles["direction"] == "I(G;E|M)"
            ],
            "montage_band": controls["montage_band"],
            "surrogates": controls["surrogates"],
            "scorecard": scorecard,
        }
        generate_required_figures(plot_tables, figures_dir)
    else:
        checks = pd.DataFrame(
            [
                {
                    "check": "core_only_run",
                    "observed": "controls and full figure suite intentionally skipped",
                    "expected": "use full smoke run for completion evidence",
                    "pass": True,
                }
            ]
        )
        checks.to_csv(tables_dir / "scientific_checks.csv", index=False)

    # Save compact exact generator parameters, never the raw synthetic arrays.
    manifests = {
        f"{world}|{window}|{seed}": dataset.parameters
        for (world, window, seed), dataset in datasets.items()
    }
    with (output / "generator_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(manifests, handle, indent=2, default=_json_default, sort_keys=True)

    run_metadata = {
        "framework_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "config_path": str(config_path),
        "seeds": seeds,
        "causal_seeds": causal_seeds,
        "null_calibration_seeds": null_seeds,
        "package_versions": _package_versions(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git_state(root),
        "core_only": core_only,
        "real_data_used": False,
        "units": "bits per independent trial",
        "oracle_precision_failures": int(
            (~causal_core.loc[
                causal_core["estimator_name"] == "bayes_oracle",
                "oracle_precision_achieved",
            ].astype(bool)).sum()
        ),
        "warnings": sorted(
            {
                warning
                for value in combined_estimates["warnings"].fillna("")
                for warning in str(value).split(";")
                if warning
            }
        ),
        "estimator_names": sorted(
            set(combined_estimates["estimator_name"]) | {"bayes_oracle"}
        ),
        "estimator_failures": {
            row.estimator_name: float(row.failure_rate)
            for row in scorecard.itertuples()
        },
        "scientific_checks_passed": bool(checks["pass"].all()),
    }
    with (output / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, default=_json_default, sort_keys=True)
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    with (output / "seed.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(str(seed) for seed in seeds) + "\n")
    lock_source = root / "requirements.lock"
    with (output / "env.lock").open("w", encoding="utf-8") as handle:
        if lock_source.exists():
            handle.write(lock_source.read_text(encoding="utf-8"))
        else:
            for package, package_version in sorted(_package_versions().items()):
                handle.write(f"{package}=={package_version}\n")
    with (output / "data.sha256").open("w", encoding="utf-8") as handle:
        handle.write("NO_EXTERNAL_DATA -- deterministic synthetic generation only\n")

    post = causal_core[
        (causal_core["window"] == "post")
        & (causal_core["estimator_name"] != "bayes_oracle")
    ]
    lines = [
        f"# {config.get('name', 'benchmark')} synthetic validation",
        "",
        f"Generated: {run_metadata['timestamp_utc']}",
        f"Runtime: {run_metadata['runtime_seconds']:.2f} seconds",
        f"Scientific checks passed: {run_metadata['scientific_checks_passed']}",
        f"Oracle precision failures: {run_metadata['oracle_precision_failures']}",
        "",
        "## Post-onset estimator summary",
        "",
        "| World | Direction | Oracle | Decoder | Direct CMIh |",
        "|---|---|---:|---:|---:|",
    ]
    for (world, direction), subset in post.groupby(["world", "direction"]):
        oracle_value = float(subset["oracle_cmi_bits"].mean())
        decoder_values = subset[
            subset["estimator_name"] == "nested_logistic_log_loss"
        ]["estimate_bits"]
        direct_values = subset[
            subset["estimator_name"] == "zmadg_cmi_hybrid_knn"
        ]["estimate_bits"]
        lines.append(
            f"| {world} | `{direction}` | {oracle_value:.3f} | "
            f"{decoder_values.mean():.3f} | {direct_values.mean():.3f} |"
        )
    lines.extend(
        [
            "",
            "## Scientific checks",
            "",
            "| Check | Observed | Pass |",
            "|---|---|---:|",
        ]
    )
    for _, row in checks.iterrows():
        observed = str(row["observed"]).replace("|", "/")
        lines.append(f"| {row['check']} | {observed} | {row['pass']} |")
    lines.extend(
        [
            "",
            "All values are bits per independent trial. This validates the synthetic measuring apparatus only; it is not evidence for a biological hypothesis.",
            "",
        ]
    )
    with (output / "RESULTS.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output


def regenerate_report(result_directory: str | Path) -> Path:
    result = Path(result_directory).resolve()
    tables = result / "tables"

    def read_table(name: str) -> pd.DataFrame:
        # Pandas' default NA vocabulary includes the literal label "null".
        # Preserve scientific scenario names while still parsing empty CSV cells.
        return pd.read_csv(
            tables / name, keep_default_na=False, na_values=[""]
        )

    required = {
        "estimated_vs_true": pd.concat(
            [
                read_table("exact_samples.csv"),
                read_table("causal_sample_sizes.csv"),
                read_table("causal_core.csv").query(
                    "estimator_name != 'bayes_oracle'"
                ),
            ],
            ignore_index=True,
            sort=False,
        ),
        "bias_sample": pd.concat(
            [
                read_table("exact_samples.csv"),
                read_table("causal_sample_sizes.csv"),
            ],
            ignore_index=True,
            sort=False,
        ),
        "null": read_table("null_calibration.csv"),
        "dimension": read_table("dimension.csv"),
        "honesty": read_table("honesty.csv"),
        "degradation": read_table("degradation.csv"),
        "proxy": read_table("proxy.csv"),
        "richness": read_table("richness.csv"),
        "montage_band": read_table("montage_band.csv"),
        "surrogates": read_table("surrogates.csv"),
        "scorecard": read_table("estimator_scorecard.csv"),
    }
    subject_path = tables / "subject_oracles.csv"
    if subject_path.exists():
        subject_frame = read_table("subject_oracles.csv")
    else:
        causal = read_table("causal_core.csv")
        oracle = causal[causal["estimator_name"] == "bayes_oracle"]
        subjects = []
        n_subjects = int(
            load_config(result / "config.yaml")["causal"]["generator"]["n_subjects"]
        )
        for row in oracle.itertuples():
            for subject in range(n_subjects):
                subjects.append(
                    {
                        "world": row.world,
                        "window": row.window,
                        "direction": row.direction,
                        "estimate_bits": row.estimate_bits,
                        "subject_id": subject,
                        "seed": row.seed,
                        "subject_parameters_identical": True,
                    }
                )
        subject_frame = pd.DataFrame(subjects)
    required["directions"] = subject_frame[subject_frame["window"] == "post"]
    required["pre_post"] = subject_frame[
        subject_frame["direction"] == "I(G;E|M)"
    ]
    generate_required_figures(required, result / "figures")
    return result / "figures"
