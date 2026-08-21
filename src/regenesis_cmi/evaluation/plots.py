"""Publication-oriented figures, each paired with its exact source table."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _finish(
    figure: plt.Figure,
    frame: pd.DataFrame,
    output: Path,
    name: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"{name}.csv", index=False)
    figure.tight_layout()
    figure.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def _point_overlay(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
) -> None:
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        dodge=True,
        alpha=0.38,
        size=3,
        palette="colorblind",
        legend=False,
        ax=ax,
    )
    sns.pointplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        dodge=0.35,
        errorbar=None,
        markers="o",
        linestyles="none",
        palette="colorblind",
        ax=ax,
    )


def plot_estimated_vs_true(frame: pd.DataFrame, output: Path) -> None:
    data = frame[frame["estimator_name"] != "bayes_oracle"].copy()
    figure, ax = plt.subplots(figsize=(7.2, 5.4))
    sns.scatterplot(
        data=data,
        x="oracle_cmi_bits",
        y="estimate_bits",
        hue="estimator_name",
        style="sample_size",
        palette="colorblind",
        alpha=0.8,
        ax=ax,
    )
    low = min(0.0, data[["oracle_cmi_bits", "estimate_bits"]].min().min())
    high = data[["oracle_cmi_bits", "estimate_bits"]].max().max()
    ax.plot([low, high], [low, high], "--", color="black", linewidth=1, label="y=x")
    ax.set(xlabel="Oracle CMI (bits/trial)", ylabel="Estimated CMI (bits/trial)")
    ax.set_title("Estimated versus known conditional mutual information")
    _finish(figure, data, output, "estimated_vs_true_cmi")


def plot_bias_sample(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    exact = data[data["estimator_name"] == "exact_discrete_plugin"]
    continuous = data[data["estimator_name"] != "exact_discrete_plugin"]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    sns.lineplot(
        data=exact,
        x="sample_size",
        y="bias_bits",
        hue="world",
        markers=True,
        dashes=False,
        errorbar="sd",
        palette="colorblind",
        ax=axes[0],
    )
    sns.lineplot(
        data=continuous,
        x="sample_size",
        y="bias_bits",
        hue="estimator_name",
        style="world",
        markers=True,
        dashes=False,
        errorbar="sd",
        palette="colorblind",
        ax=axes[1],
    )
    for ax in axes:
        ax.axhline(0, color="black", linewidth=1)
        ax.set(xscale="log", xlabel="Independent trials", ylabel="Bias (bits/trial)")
    axes[0].set_title("Exact discrete plug-in")
    axes[1].set_title("Continuous biological candidates")
    figure.suptitle("Finite-sample bias, including true-zero worlds", y=1.02)
    _finish(figure, data, output, "bias_vs_sample_size")


def plot_null_distribution(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    figure, ax = plt.subplots(figsize=(7.2, 5.0))
    sns.histplot(
        data=data,
        x="estimate_bits",
        hue="estimator_name",
        element="step",
        stat="density",
        common_norm=False,
        palette="colorblind",
        ax=ax,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Estimated CMI under known null (bits/trial)")
    ax.set_title("Independent synthetic-null calibration distribution")
    _finish(figure, data, output, "null_distribution")


def plot_error_dimension(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    figure, ax = plt.subplots(figsize=(7.2, 5.0))
    sns.lineplot(
        data=data,
        x="total_dimension",
        y="absolute_error_bits",
        hue="estimator_name",
        marker="o",
        errorbar="sd",
        palette="colorblind",
        ax=ax,
    )
    ax.set(xlabel="Total continuous dimensions", ylabel="Absolute error (bits/trial)")
    ax.set_title("Estimator error versus dimension")
    _finish(figure, data, output, "error_vs_dimension")


def plot_both_directions(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    figure, ax = plt.subplots(figsize=(9.0, 5.2))
    _point_overlay(ax, data, x="world", y="estimate_bits", hue="direction")
    ax.set(ylabel="CMI (bits/trial)", xlabel="Synthetic world")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Both conditional-information directions (per-subject points)")
    _finish(figure, data, output, "both_directions")


def plot_honesty(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    figure, ax = plt.subplots(figsize=(9.0, 5.2))
    _point_overlay(ax, data, x="world", y="estimate_bits", hue="control")
    ax.set(ylabel="I(G;E|M, control) (bits/trial)", xlabel="Synthetic world")
    ax.tick_params(axis="x", rotation=18)
    ax.set_title("Myogenic honesty control")
    _finish(figure, data, output, "myogenic_honesty")


def plot_degradation(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    figure, axes = plt.subplots(1, data["world"].nunique(), figsize=(12.0, 4.7), sharey=True)
    axes_array = np.atleast_1d(axes)
    for ax, (world, subset) in zip(axes_array, data.groupby("world"), strict=False):
        sns.lineplot(
            data=subset,
            x="alpha",
            y="estimate_bits",
            hue="condition",
            marker="o",
            errorbar=None,
            palette="colorblind",
            ax=ax,
        )
        sns.scatterplot(
            data=subset,
            x="alpha",
            y="estimate_bits",
            hue="condition",
            legend=False,
            alpha=0.3,
            palette="colorblind",
            ax=ax,
        )
        ax.invert_xaxis()
        ax.set_title(world.replace("_", " "))
        ax.set(xlabel="EMG signal factor α", ylabel="I(G;E|Mα) (bits/trial)")
    _finish(figure, data, output, "degradation_sweep")


def plot_proxy(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    figure, ax = plt.subplots(figsize=(7.2, 5.0))
    sns.lineplot(
        data=data,
        x="proxy_quality",
        y="estimate_bits",
        hue="world",
        marker="o",
        errorbar=None,
        palette="colorblind",
        ax=ax,
    )
    sns.scatterplot(
        data=data,
        x="proxy_quality",
        y="estimate_bits",
        hue="world",
        legend=False,
        alpha=0.35,
        palette="colorblind",
        ax=ax,
    )
    ax.set(xlabel="Correlation parameter r", ylabel="Residual CMI (bits/trial)")
    ax.set_title("Imperfect myogenic proxies leave residual artifact")
    _finish(figure, data, output, "proxy_quality")


def plot_richness(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    order = [value for value in ("sparse", "intermediate", "rich") if value in set(data["richness"])]
    figure, ax = plt.subplots(figsize=(7.2, 5.0))
    _point_overlay(ax, data, x="richness", y="estimate_bits", hue="world")
    ax.set(xlabel="EMG conditioning representation", ylabel="I(G;E|Mrep) (bits/trial)")
    ax.set_title("Information lost from M can reappear as apparently unique E")
    _finish(figure, data, output, "conditioning_richness")


def plot_pre_post(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    figure, ax = plt.subplots(figsize=(9.0, 5.2))
    _point_overlay(ax, data, x="world", y="estimate_bits", hue="window")
    ax.set(ylabel="I(G;E|M) (bits/trial)", xlabel="Synthetic world")
    ax.tick_params(axis="x", rotation=18)
    ax.set_title("Pre-onset anticipation and post-onset complementarity are separate")
    _finish(figure, data, output, "pre_vs_post")


def plot_montage_band(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    montage = data[data["band"] == "all"]
    band = data[data["montage"] == "full"]
    _point_overlay(axes[0], montage, x="world", y="estimate_bits", hue="montage")
    _point_overlay(axes[1], band, x="band", y="estimate_bits", hue="world")
    axes[0].set_title("Full versus central-only montage")
    axes[1].set_title("Band-resolved feature groups")
    for ax in axes:
        ax.set(ylabel="I(G;E|M) (bits/trial)")
        ax.tick_params(axis="x", rotation=20)
    _finish(figure, data, output, "montage_and_band")


def plot_surrogates(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    figure, ax = plt.subplots(figsize=(9.0, 5.2))
    sns.boxplot(
        data=data,
        x="world",
        y="estimate_bits",
        hue="surrogate",
        palette="colorblind",
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="world",
        y="estimate_bits",
        hue="surrogate",
        dodge=True,
        alpha=0.4,
        size=3,
        palette="colorblind",
        legend=False,
        ax=ax,
    )
    ax.set(ylabel="Estimated I(G;E|M) (bits/trial)", xlabel="Synthetic world")
    ax.set_title("Label shuffle cannot localize origin; cortical intervention can")
    _finish(figure, data, output, "surrogate_comparison")


def plot_scorecard(frame: pd.DataFrame, output: Path) -> None:
    data = frame.copy()
    columns = [
        "bias_bits",
        "rmse_bits",
        "null_false_positive_rate",
        "mean_runtime_seconds",
        "failure_rate",
    ]
    heat = data.set_index("estimator_name")[columns]
    color = heat.copy()
    color["bias_bits"] = color["bias_bits"].abs()
    for column in color:
        minimum, maximum = color[column].min(), color[column].max()
        color[column] = (
            0.0 if maximum == minimum else (color[column] - minimum) / (maximum - minimum)
        )
    figure, ax = plt.subplots(figsize=(9.0, max(3.2, 0.7 * len(heat))))
    sns.heatmap(
        color,
        annot=heat,
        fmt=".3g",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "relative magnitude within each metric"},
        ax=ax,
    )
    ax.set_title("Estimator scorecard (raw metrics; see source CSV for definitions)")
    _finish(figure, data, output, "estimator_scorecard")


def generate_required_figures(tables: dict[str, pd.DataFrame], output: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
    dispatch: list[tuple[str, Callable[[pd.DataFrame, Path], None]]] = [
        ("estimated_vs_true", plot_estimated_vs_true),
        ("bias_sample", plot_bias_sample),
        ("null", plot_null_distribution),
        ("dimension", plot_error_dimension),
        ("directions", plot_both_directions),
        ("honesty", plot_honesty),
        ("degradation", plot_degradation),
        ("proxy", plot_proxy),
        ("richness", plot_richness),
        ("pre_post", plot_pre_post),
        ("montage_band", plot_montage_band),
        ("surrogates", plot_surrogates),
        ("scorecard", plot_scorecard),
    ]
    for table_name, function in dispatch:
        frame = tables.get(table_name)
        if frame is None or frame.empty:
            raise ValueError(f"Required figure source table {table_name!r} is empty")
        function(frame, output)
