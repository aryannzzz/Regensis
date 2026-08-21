"""Bias, error, detection, and runtime summaries across independent seeds."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def add_error_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["bias_bits"] = result["estimate_bits"] - result["oracle_cmi_bits"]
    result["absolute_error_bits"] = result["bias_bits"].abs()
    result["squared_error_bits2"] = result["bias_bits"] ** 2
    return result


def _coverage_and_detection(rows: pd.DataFrame) -> tuple[float, float, int]:
    if "benchmark_cell" not in rows:
        return float("nan"), float("nan"), 0
    covered: list[bool] = []
    detected: list[bool] = []
    for _, cell in rows.groupby("benchmark_cell", dropna=False):
        values = cell["estimate_bits"].dropna().to_numpy()
        if len(values) < 2:
            continue
        truth = float(cell["oracle_cmi_bits"].iloc[0])
        half_width = 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
        lower, upper = float(np.mean(values) - half_width), float(np.mean(values) + half_width)
        covered.append(lower <= truth <= upper)
        if 0.0 < truth <= 0.25:
            detected.append(lower > 0.0)
    return (
        float(np.mean(covered)) if covered else float("nan"),
        float(np.mean(detected)) if detected else float("nan"),
        len(covered),
    )


def estimator_scorecard(
    estimates: pd.DataFrame,
    *,
    null_calibration: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize candidates; positivity alone is never called significance."""

    candidates = estimates[estimates["estimator_name"] != "bayes_oracle"].copy()
    records: list[dict[str, object]] = []
    for estimator_name, rows in candidates.groupby("estimator_name", dropna=False):
        valid = rows.dropna(subset=["estimate_bits", "oracle_cmi_bits"])
        errors = valid["estimate_bits"] - valid["oracle_cmi_bits"]
        coverage, small_detection, interval_cells = _coverage_and_detection(valid)
        false_positive_rate = float("nan")
        null_threshold = float("nan")
        null_evaluations = 0
        if null_calibration is not None:
            available = null_calibration[
                null_calibration["estimator_name"] == estimator_name
            ].sort_values("seed")
            # Use disjoint simulator seeds for calibration and evaluation. This
            # avoids defining a threshold on the same null realizations used to
            # report its false-positive rate.
            split = len(available) // 2
            calibration = available.iloc[:split]["estimate_bits"].dropna()
            evaluation_nulls = available.iloc[split:]["estimate_bits"].dropna()
            if len(calibration) >= 5 and len(evaluation_nulls) >= 5:
                null_threshold = float(np.quantile(calibration, 0.95))
                false_positive_rate = float(
                    np.mean(evaluation_nulls > null_threshold)
                )
                null_evaluations = len(evaluation_nulls)
        records.append(
            {
                "estimator_name": estimator_name,
                "bias_bits": float(errors.mean()) if len(errors) else float("nan"),
                "mae_bits": float(np.mean(np.abs(errors))) if len(errors) else float("nan"),
                "rmse_bits": float(np.sqrt(np.mean(errors**2)))
                if len(errors)
                else float("nan"),
                "error_sd_bits": float(errors.std(ddof=1))
                if len(errors) > 1
                else 0.0,
                "null_false_positive_rate": false_positive_rate,
                "null_95pct_calibration_threshold_bits": null_threshold,
                "null_evaluation_count": null_evaluations,
                "normal_interval_coverage": coverage,
                "interval_cell_count": interval_cells,
                "small_effect_detection_rate": small_detection,
                "failure_rate": float(1.0 - len(valid) / len(rows)) if len(rows) else 1.0,
                "mean_runtime_seconds": float(valid["runtime_seconds"].mean())
                if len(valid)
                else float("nan"),
                "evaluations": len(rows),
                "units": "bits per independent trial",
                "significance_rule": (
                    "estimate exceeds disjoint-seed synthetic-null 95th percentile"
                ),
                "interval_rule": "normal interval across independent seeds",
            }
        )
    return pd.DataFrame(records)
