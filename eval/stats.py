"""eval/stats.py — Track E deliverable. See repo copy for full docstring."""
from __future__ import annotations
from typing import Sequence
import numpy as np
from scipy import stats as _scipy_stats


def bootstrap_ci(values, n_resamples=1000, ci=95.0, rng=None):
    if n_resamples < 1000:
        raise ValueError(f"n_resamples={n_resamples} is below the manual's floor of 1000 (§5.1).")
    values = np.asarray(values, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    n = len(values)
    resample_medians = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        resample_medians[i] = np.median(sample)
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(resample_medians, [alpha, 100.0 - alpha])
    return {"median": float(np.median(values)), "ci_low": float(lo),
            "ci_high": float(hi), "n_resamples": n_resamples}


def permutation_test(real_stat, null_stats, alternative="greater"):
    null_stats = np.asarray(null_stats, dtype=float)
    n = len(null_stats)
    if alternative == "greater":
        count = np.sum(null_stats >= real_stat)
    elif alternative == "less":
        count = np.sum(null_stats <= real_stat)
    elif alternative == "two-sided":
        centered_real = abs(real_stat - np.median(null_stats))
        centered_null = np.abs(null_stats - np.median(null_stats))
        count = np.sum(centered_null >= centered_real)
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")
    p_value = (count + 1) / (n + 1)
    return {"p_value": float(p_value), "n_surrogates": n, "real_stat": float(real_stat)}


def wilcoxon_holm(comparisons):
    raw_results = {}
    for name, (x, y) in comparisons.items():
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) != len(y):
            raise ValueError(f"Comparison '{name}': paired arrays must be equal length.")
        stat, p = _scipy_stats.wilcoxon(x, y)
        raw_results[name] = {"statistic": float(stat), "p_raw": float(p)}
    names_sorted = sorted(raw_results, key=lambda n: raw_results[n]["p_raw"])
    m = len(names_sorted)
    holm_p = {}
    running_max = 0.0
    for i, name in enumerate(names_sorted):
        adjusted = (m - i) * raw_results[name]["p_raw"]
        running_max = max(running_max, adjusted)
        holm_p[name] = min(running_max, 1.0)
    out = {}
    for name in comparisons:
        p_holm = holm_p[name]
        out[name] = {"statistic": raw_results[name]["statistic"],
                      "p_raw": raw_results[name]["p_raw"], "p_holm": p_holm,
                      "reject_at_0.05": p_holm < 0.05}
    return out