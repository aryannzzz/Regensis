"""
eeg/anticipation.py

The E3-A detector: EMG threshold-onset marking, the lead x TPR x FP/min
threshold sweep, and the three mandatory controls (time-reversed
surrogate, shuffled-onset null, pre-baseline-shift check). Per
prereg/PR-2026-02.md and the E3-A protocol doc.

Detector-family-agnostic: takes decision values from either
eeg/decoders_linear.py-style sLDA (on preprocessing/features_eeg.py's
window features) or eeg/decoder_eegnet.py (on raw windowed signal) --
this file only ever sees a decision-value array plus window metadata,
same pattern as run_t1r.py treating csp/riemann interchangeably.

Splits, surrogates and stats used here come from eval/ only (S0.1/S8.1)
EXCEPT the shuffled-onset null, which eval/surrogates.py does not
provide -- see shuffled_onset_null()'s docstring. That gap should be
filed as an issue against eval/ (Track A never carries a permanent
local copy of a shared surrogate); the local implementation here is a
stated, temporary deviation, not a quiet workaround.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from eval import stats as eval_stats
from eval import surrogates as eval_surrogates
from preprocessing.features_eeg import SlidingWindowConfig, sliding_windows, window_features


# ---------------------------------------------------------------------
# 1. EMG threshold-onset marking (protocol doc: "onset is defined by
#    actual muscle activation, not by a mechanical trigger")
# ---------------------------------------------------------------------

@dataclass
class OnsetConfig:
    rectify: bool = True
    smooth_window_s: float = 0.05     # moving-average envelope window
    baseline_window_s: tuple = (0.0, 1.0)   # rest period to estimate baseline
                                              # EMG level from, seconds from
                                              # trial start
    threshold_sd: float = 3.0         # onset = envelope crosses
                                       # baseline_mean + threshold_sd * baseline_sd


def emg_threshold_onset(emg: np.ndarray, fs: float, cfg: OnsetConfig = OnsetConfig()) -> dict:
    """
    Mark true movement onset from the EMG signal via threshold-crossing.

    Args:
        emg: (n_emg_channels, n_times) for one trial, mean-removed
             (WAY-EEG-GAL's own preprocessing; see data/way_eeg_gal.py).
        fs: EMG sampling rate, Hz.
        cfg: rectify/smooth/threshold parameters (S8.4: loggable).

    Returns:
        dict with onset_sample (int or None if no crossing found),
        envelope (n_times,) the channel-summed envelope used, and the
        config.

    Per protocol doc: "Manually check at least 30 trials to confirm
    onset detection is not off" -- this function does the detection;
    the hand-verification is a separate step the caller must perform
    (log it: how many trials verified, how many corrected) before any
    epoching downstream uses these onsets.
    """
    emg = np.asarray(emg, dtype=float)
    sig = np.abs(emg) if cfg.rectify else emg
    channel_sum = sig.sum(axis=0)

    win = max(1, int(round(cfg.smooth_window_s * fs)))
    kernel = np.ones(win) / win
    envelope = np.convolve(channel_sum, kernel, mode="same")

    b0, b1 = int(round(cfg.baseline_window_s[0] * fs)), int(round(cfg.baseline_window_s[1] * fs))
    baseline = envelope[b0:b1]
    if len(baseline) == 0:
        raise ValueError(f"Empty baseline window {cfg.baseline_window_s}s at fs={fs} -- "
                          f"check trial length and baseline_window_s.")
    threshold = baseline.mean() + cfg.threshold_sd * baseline.std()

    crossings = np.where(envelope[b1:] > threshold)[0]
    onset_sample = int(b1 + crossings[0]) if len(crossings) else None

    return {"onset_sample": onset_sample, "envelope": envelope,
            "threshold": float(threshold), "config": cfg}


# ---------------------------------------------------------------------
# 2. sLDA detector (background baseline) on window_features()
# ---------------------------------------------------------------------

@dataclass
class SLDAConfig:
    shrinkage: str | float = "auto"


def build_slda(cfg: SLDAConfig = SLDAConfig()) -> LinearDiscriminantAnalysis:
    """
    Shrinkage LDA on preprocessing/features_eeg.window_features() output
    -- NOT eeg/decoders_linear.py's CSP+sLDA pipeline. That pipeline
    expects raw (n_epochs, n_channels, n_times) trial epochs; E3-A's
    sLDA arm classifies short pre-onset windows from hand-crafted
    mean/slope features (protocol doc: "sLDA (background baseline)"),
    a different input shape entirely. Sharing the LDA solver/shrinkage
    convention with decoders_linear.py (solver="lsqr", shrinkage="auto"
    by default) rather than reinventing a different sLDA config.
    """
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=cfg.shrinkage)


def slda_fit_predict_fold(clf, X_train, y_train, X_test, y_test):
    """Mirrors eeg/decoders_linear.fit_predict_fold's return shape."""
    clf.fit(X_train, y_train)
    return clf.predict(X_test), clf.decision_function(X_test)


# ---------------------------------------------------------------------
# 3. Threshold sweep: lead time x TPR x FP/min
# ---------------------------------------------------------------------

def threshold_sweep(
    decision_values: np.ndarray,
    labels: np.ndarray,
    window_end_sample: np.ndarray,
    onset_sample: int,
    fs: float,
    rest_duration_s: float,
    n_thresholds: int = 50,
) -> dict:
    """
    Sweep the detection threshold across the decision-value range and,
    at each threshold, compute mean detection lead time (over true
    positives), TPR, and false positives per minute (over rest-period
    negatives) -- per protocol doc pipeline step 6: "not a single
    number."

    Args:
        decision_values: (n_windows,) detector output for one trial's
            windows (from slda_fit_predict_fold or
            eeg.decoder_eegnet.fit_predict_fold).
        labels: (n_windows,) 1="movement coming", 0="rest" -- from
            preprocessing.features_eeg.sliding_windows.
        window_end_sample: (n_windows,) sample index of each window's
            right edge, same array sliding_windows() returned.
        onset_sample: EMG-marked onset sample for this trial, for lead
            time = (onset_sample - window_end_sample) / fs at the first
            positive-labeled window whose decision value crosses
            threshold.
        fs: sampling rate, Hz.
        rest_duration_s: total rest-period duration contributing
            negative windows for this trial, for the FP/min denominator.
        n_thresholds: number of threshold points swept.

    Returns:
        dict of arrays, each length n_thresholds: thresholds,
        mean_lead_time_s (nan where no true positive fires),
        tpr, fp_per_min.
    """
    decision_values = np.asarray(decision_values)
    labels = np.asarray(labels)
    window_end_sample = np.asarray(window_end_sample)

    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = pos_mask.sum()

    thresholds = np.linspace(decision_values.min(), decision_values.max(), n_thresholds)
    mean_lead = np.full(n_thresholds, np.nan)
    tpr = np.zeros(n_thresholds)
    fp_per_min = np.zeros(n_thresholds)

    for i, thr in enumerate(thresholds):
        fires = decision_values >= thr
        pos_fires = fires & pos_mask
        if pos_fires.any():
            leads_s = (onset_sample - window_end_sample[pos_fires]) / fs
            mean_lead[i] = float(np.mean(leads_s))
        tpr[i] = float(pos_fires.sum() / n_pos) if n_pos else 0.0

        n_fp = int((fires & neg_mask).sum())
        fp_per_min[i] = n_fp / (rest_duration_s / 60.0) if rest_duration_s > 0 else np.nan

    return {"thresholds": thresholds, "mean_lead_time_s": mean_lead,
            "tpr": tpr, "fp_per_min": fp_per_min}


# ---------------------------------------------------------------------
# 4. Controls
# ---------------------------------------------------------------------

def time_reversed_surrogate(X_windows: np.ndarray) -> np.ndarray:
    """Control 1: reverse each epoch. Uses eval/surrogates.time_reversal
    (S0.1: splitters/surrogates/estimators/metrics come from eval/
    only). A real pre-movement signal should not survive this."""
    return eval_surrogates.time_reversal(X_windows)


def shuffled_onset_null(n_times: int, rng: np.random.Generator,
                         min_onset_s: float, max_onset_s: float, fs: float) -> int:
    """
    Control 2: randomize onset time within the recording.

    NOT sourced from eval/surrogates.py -- that file has time_reversal,
    label_shuffle, phase_randomize and channel_permutation, but nothing
    that generates a randomized *event time* to re-run epoching against.
    Per S0.1/S8.1 ("Track A never writes into eval/ ... file an issue
    against eval/ and Track E implements it"), this function is a
    stated, temporary local implementation -- file an eval/ issue
    requesting an `onset_shuffle` surrogate before this experiment's
    branch merges, and delete this function in favor of the eval/
    version once it lands (this is the kind of local copy S0.1 flags
    as a code-review defect if it's still here at merge time).

    Returns a single randomized onset sample within
    [min_onset_s, max_onset_s] of the recording.
    """
    lo = int(round(min_onset_s * fs))
    hi = min(n_times, int(round(max_onset_s * fs)))
    if hi <= lo:
        raise ValueError(f"Empty randomization range: [{min_onset_s}, {max_onset_s}]s in a "
                          f"{n_times / fs:.1f}s recording.")
    return int(rng.integers(lo, hi))


def pre_baseline_shift_check(X_windows: np.ndarray, fs: float,
                              exclude_pre_onset_s: float, window_end_sample: np.ndarray,
                              onset_sample: int) -> np.ndarray:
    """
    Control 3: re-run with baseline correction applied over a window
    that EXCLUDES the pre-onset period, to confirm the detector is not
    just keying on slow baseline drift.

    Subtracts, from each window, the mean signal from a baseline
    segment drawn from well before the anticipation horizon (i.e. from
    the "rest" region, never from inside [window_start, onset_sample])
    -- so a detector that only tracked a slow drift loses its signal,
    while a genuine MRCP relative to a clean baseline should not.

    Args:
        X_windows: (n_windows, n_channels, window_len), same windows
            threshold_sweep was run on.
        fs: sampling rate, Hz.
        exclude_pre_onset_s: baseline is drawn from samples further
            than this many seconds before onset (i.e. outside the
            anticipation horizon).
        window_end_sample: (n_windows,) as returned by sliding_windows().
        onset_sample: this trial's EMG-marked onset sample.

    Returns:
        (n_windows, n_channels, window_len) baseline-corrected windows.
        Windows whose own baseline segment would fall before recording
        start raise no special-case -- caller is responsible for
        excluding trials too short to support this control, same as
        any other windowing edge case.
    """
    exclude_samples = int(round(exclude_pre_onset_s * fs))
    corrected = np.empty_like(X_windows)
    for i, end in enumerate(window_end_sample):
        lead = onset_sample - end
        if lead < exclude_samples:
            # This window is inside the excluded pre-onset zone -- its
            # own mean is not a valid "excludes the pre-onset period"
            # baseline. Use the earliest available window's own value
            # as the baseline reference instead of silently using an
            # in-horizon baseline (which would defeat the control).
            baseline = X_windows[np.argmax(onset_sample - window_end_sample)].mean(axis=1, keepdims=True)
        else:
            baseline = X_windows[i].mean(axis=1, keepdims=True)
        corrected[i] = X_windows[i] - baseline
    return corrected


# ---------------------------------------------------------------------
# 5. Per-subject curve + stats (bootstrap CI, permutation test --
#    both from eval/, per S0.1)
# ---------------------------------------------------------------------

def per_subject_curve_with_ci(per_trial_sweeps: list[dict], rng: np.random.Generator,
                               n_bootstrap: int = 1000, ci_percent: float = 95.0) -> dict:
    """
    Aggregate threshold_sweep() output across a subject's trials into
    one curve with bootstrap CIs, per protocol doc: "Report per-subject
    curves with bootstrap confidence intervals."

    Args:
        per_trial_sweeps: list of threshold_sweep() dicts, one per
            trial, all computed on the same thresholds grid (share the
            decision-value range across trials before sweeping, or
            interpolate onto a common grid before calling this).
        rng, n_bootstrap, ci_percent: passed to eval.stats.bootstrap_ci
            (S0.1: no local copy of a shared stats primitive; the
            manual's n_resamples floor of 1000 is enforced there).

    Returns:
        dict with thresholds, and per-threshold {median, ci_low,
        ci_high} for lead time, TPR, and FP/min.
    """
    thresholds = per_trial_sweeps[0]["thresholds"]
    out = {"thresholds": thresholds}
    for metric in ("mean_lead_time_s", "tpr", "fp_per_min"):
        stacked = np.array([s[metric] for s in per_trial_sweeps])  # (n_trials, n_thresholds)
        medians, ci_lo, ci_hi = [], [], []
        for t in range(stacked.shape[1]):
            values = stacked[:, t]
            values = values[~np.isnan(values)]
            if len(values) == 0:
                medians.append(np.nan); ci_lo.append(np.nan); ci_hi.append(np.nan)
                continue
            ci = eval_stats.bootstrap_ci(values, n_resamples=n_bootstrap, ci=ci_percent, rng=rng)
            medians.append(ci["median"]); ci_lo.append(ci["ci_low"]); ci_hi.append(ci["ci_high"])
        out[metric] = {"median": np.array(medians), "ci_low": np.array(ci_lo), "ci_high": np.array(ci_hi)}
    return out


def permutation_test_at_operating_points(real_tpr: np.ndarray, null_tpr_draws: np.ndarray,
                                          alternative: str = "greater") -> list[dict]:
    """
    Permutation test against the shuffled-onset null at each operating
    point (protocol doc Evaluation section), via eval.stats.permutation_test
    (S0.1). Holm correction across operating points is the caller's job
    (eval.stats.wilcoxon_holm's pattern generalizes; applying Holm to
    a list of p-values here rather than duplicating that logic).

    Args:
        real_tpr: (n_thresholds,) real-data TPR curve.
        null_tpr_draws: (n_surrogates, n_thresholds) TPR curves computed
            under shuffled_onset_null, same thresholds grid.

    Returns:
        list of eval.stats.permutation_test() results, one per threshold.
    """
    return [
        eval_stats.permutation_test(real_tpr[t], null_tpr_draws[:, t], alternative=alternative)
        for t in range(len(real_tpr))
    ]