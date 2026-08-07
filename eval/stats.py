"""
eval/stats.py — Track E deliverable, laboratory-wide.

Statistical utilities used across Project Regenesis. Single
implementation (§0.1, §3.2) — no track reimplements these locally.

Covers what T1-R and E2-A need per the manual's stated statistical
tests (§5.1, §5.2):
  - bootstrap_ci      — percentile bootstrap CI on a per-subject metric,
                         >=1000 resamples (§5.1: "Bootstrap percentile
                         CIs on per-subject accuracy (>=1000 resamples)").
  - permutation_test  — real statistic against a label-shuffle (or other
                         surrogate) null distribution.
  - wilcoxon_holm      — paired Wilcoxon signed-rank across subjects for
                         a family of comparisons (e.g. per-band), with
                         Holm correction across the family (§5.2:
                         "Holm correction across the band family, stated
                         in the pre-registration").
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats as _scipy_stats


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 95.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Percentile bootstrap CI on the median of `values` (per-subject
    accuracy, typically).

    Parameters
    ----------
    values : array, shape (n_subjects,)
        One value per subject (or per fold — whatever the unit of
        resampling should be; resample at the level you'd want the CI
        to reflect).
    n_resamples : int, default 1000
        Minimum required by the manual is 1000; higher is fine, never lower.
    ci : float, default 95.0
        Confidence level in percent.
    rng : numpy.random.Generator, optional
        Pass one for reproducibility; logged per §8.4.

    Returns
    -------
    dict with 'median', 'ci_low', 'ci_high', 'n_resamples'.
    """
    if n_resamples < 1000:
        raise ValueError(
            f"n_resamples={n_resamples} is below the manual's floor of 1000 (§5.1)."
        )
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
    return {
        "median": float(np.median(values)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_resamples": n_resamples,
    }


def permutation_test(
    real_stat: float,
    null_stats: np.ndarray,
    alternative: str = "greater",
) -> dict:
    """
    Permutation test of a real statistic against a null distribution
    built from surrogate draws (typically `eval.surrogates.label_shuffle`
    applied many times, per §5.1's "permutation test against the
    label-shuffle null").

    Parameters
    ----------
    real_stat : float
        The statistic computed on real (non-surrogate) data.
    null_stats : array, shape (n_surrogates,)
        The same statistic recomputed on each surrogate draw.
    alternative : {'greater', 'less', 'two-sided'}, default 'greater'
        'greater' is the usual case for accuracy-type metrics, where the
        hypothesis is that the real effect exceeds chance/null.

    Returns
    -------
    dict with 'p_value', 'n_surrogates', 'real_stat'.

    Notes
    -----
    p-value uses the standard "+1" correction (Davison & Hinkley / Phipson
    & Smyth) so p is never reported as exactly zero regardless of
    n_surrogates.
    """
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


def wilcoxon_holm(comparisons: dict[str, tuple[Sequence[float], Sequence[float]]]) -> dict:
    """
    Paired Wilcoxon signed-rank test per comparison in a family, with
    Holm-Bonferroni correction across the family (§5.2: band comparisons,
    full-vs-central-only montage comparison).

    Parameters
    ----------
    comparisons : dict
        Mapping from a comparison name (e.g. 'mrcp_vs_beta',
        'full_vs_central') to a (x, y) pair of equal-length paired
        per-subject arrays.

    Returns
    -------
    dict mapping each comparison name to
        {'statistic', 'p_raw', 'p_holm', 'reject_at_0.05'}
    Ordered by ascending p_raw is used internally for the Holm procedure;
    the returned dict preserves the caller's original key order.
    """
    raw_results = {}
    for name, (x, y) in comparisons.items():
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) != len(y):
            raise ValueError(f"Comparison '{name}': paired arrays must be equal length.")
        stat, p = _scipy_stats.wilcoxon(x, y)
        raw_results[name] = {"statistic": float(stat), "p_raw": float(p)}

    # Holm-Bonferroni: sort ascending by p_raw, step up applying an
    # increasing correction, enforcing monotonicity.
    names_sorted = sorted(raw_results, key=lambda n: raw_results[n]["p_raw"])
    m = len(names_sorted)
    holm_p = {}
    running_max = 0.0
    for i, name in enumerate(names_sorted):
        adjusted = (m - i) * raw_results[name]["p_raw"]
        running_max = max(running_max, adjusted)
        holm_p[name] = min(running_max, 1.0)

    out = {}
    for name in comparisons:  # preserve caller's original order
        p_holm = holm_p[name]
        out[name] = {
            "statistic": raw_results[name]["statistic"],
            "p_raw": raw_results[name]["p_raw"],
            "p_holm": p_holm,
            "reject_at_0.05": p_holm < 0.05,
        }
    return out