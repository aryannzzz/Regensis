"""
experiments/run_e3a.py — orchestrates E3-A end to end (protocol doc; S8.2).

    python -m experiments.run_e3a --config experiments/e3a_anticipation.yaml

Same shape as experiments/run_t1r.py: try to import the real pipeline
modules, fall back to a clearly-labeled SYNTHETIC DEMO MODE if any are
missing (data/way_eeg_gal.py needs a real downloaded corpus;
eeg/decoder_eegnet.py needs torch, not yet in env.lock), so the
orchestration logic itself (onset -> windows -> reject -> detect ->
sweep -> controls -> curve -> figure -> log) is provably correct today
even before both dependencies land. Demo-mode runs never write into
results/<PR>/ as if they were a real result.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import splits as eval_splits
from eval import stats as eval_stats
from eeg import anticipation
from preprocessing.features_eeg import SlidingWindowConfig, sliding_windows, window_features
from preprocessing.eeg_artifacts import RejectionConfig, reject_trials
from experiments.logging_e3a import E3ARunRecord, E3ALogger


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _require(cfg: dict, path: str, logger: E3ALogger, stage: str):
    node = cfg
    for key in path.split("."):
        node = node[key]
    if node is None:
        logger.log_missing_param_error(stage, path)
    return node


def _try_load_real_modules():
    """Import the real data loader and EEGNet detector if present;
    report which are missing rather than failing on the first one."""
    missing = []
    mods = {}
    try:
        from data import way_eeg_gal
        mods["way_eeg_gal"] = way_eeg_gal
    except ImportError:
        missing.append("data/way_eeg_gal.py (mne/scipy) or no corpus at configured data_dir")
    try:
        from eeg import decoder_eegnet
        mods["decoder_eegnet"] = decoder_eegnet
    except ImportError:
        missing.append("eeg/decoder_eegnet.py (torch not installed -- add to env.lock)")
    return mods, missing


def _synthetic_trial(fs: float, n_channels: int, duration_s: float, seed: int,
                      onset_s: float, snr: float = 1.5):
    """
    SYNTHETIC DEMO MODE ONLY. Fabricated single-trial MRCP-band signal
    with a planted negative-going slow shift before a labeled onset, so
    the orchestration logic (window -> reject -> detect -> sweep ->
    controls) has something with a real pre-onset structure to find.
    Not real WAY-EEG-GAL data. Never written into results/<PR>/.
    """
    rng = np.random.default_rng(seed)
    n_times = int(round(duration_s * fs))
    t = np.arange(n_times) / fs
    onset_sample = int(round(onset_s * fs))

    X = rng.normal(scale=1.0, size=(n_channels, n_times))
    # Planted MRCP: a ramping negative shift over the ~1s before onset,
    # concentrated on the first few (central-like) channels.
    ramp_start = max(0, onset_sample - int(round(1.0 * fs)))
    ramp = np.zeros(n_times)
    ramp[ramp_start:onset_sample] = -snr * np.linspace(0, 1, onset_sample - ramp_start)
    X[:4, :] += ramp

    # Synthetic EMG: quiet before onset, bursts after.
    emg = rng.normal(scale=0.1, size=(5, n_times))
    emg[:, onset_sample:onset_sample + int(round(0.3 * fs))] += rng.normal(
        scale=2.0, size=(5, min(int(round(0.3 * fs)), n_times - onset_sample)))

    return X, emg, onset_sample


def run_subject(subject_num: int, cfg: dict, mods: dict, demo_mode: bool,
                 logger: E3ALogger, rng: np.random.Generator):
    subject_id = f"P{subject_num}"
    stage = f"subject_{subject_id}"

    win_cfg = SlidingWindowConfig(
        window_s=cfg["windowing"]["window_s"],
        step_s=cfg["windowing"]["step_s"],
        horizon_s=cfg["windowing"]["horizon_s"],
        rest_margin_s=cfg["windowing"]["rest_margin_s"],
    )

    # --- get onset-anchored, MRCP-filtered, artifact-rejected trials ---
    if demo_mode:
        eeg_fs = 250.0
        n_channels = 8
        n_trials = 20
        trials = []
        for i in range(n_trials):
            X, emg, onset_sample = _synthetic_trial(
                eeg_fs, n_channels, duration_s=6.0, seed=cfg["seed"] + subject_num * 100 + i,
                onset_s=4.0,
            )
            trials.append((X, onset_sample))
    else:
        data_dir = Path(_require(cfg, "data.data_dir", logger, stage))
        way_eeg_gal = mods["way_eeg_gal"]
        eeg_fs = None
        trials = []
        for series_num in range(1, 11):
            series = way_eeg_gal.load_series(data_dir, subject_num, series_num)
            eeg_fs = series.eeg_fs
            gap = way_eeg_gal.measure_sync_gap(series)
            logger.log(E3ARunRecord(
                stage=f"{stage}_series_{series_num}_sync",
                subject_id=subject_id, seed=cfg["seed"], git_commit=_git_commit(),
                measured_sync_gap_s=gap["measured_lag_s"],
                extra={"peak_correlation": gap["peak_correlation"], "demo_mode": False},
            ))
            onset_info = anticipation.emg_threshold_onset(
                series.emg, series.emg_fs,
                cfg=anticipation.OnsetConfig(
                    threshold_sd=cfg["onset"]["threshold_sd"],
                    baseline_window_s=tuple(cfg["onset"]["baseline_window_s"]),
                    smooth_window_s=cfg["onset"]["smooth_window_s"],
                ),
            )
            if onset_info["onset_sample"] is None:
                continue
            # Map EMG-clock onset sample onto the EEG clock via the
            # measured sync gap -- never assume the two share a clock.
            onset_s_emg_clock = onset_info["onset_sample"] / series.emg_fs
            onset_s_eeg_clock = onset_s_emg_clock - gap["measured_lag_s"]
            onset_sample_eeg = int(round(onset_s_eeg_clock * series.eeg_fs))
            trials.append((series.eeg, onset_sample_eeg))

    # --- windowing + rejection per trial ---
    all_windows, all_labels, all_ends, per_trial_onset = [], [], [], []
    n_rejected_total = n_total = 0
    for X, onset_sample in trials:
        sw = sliding_windows(X, eeg_fs, onset_sample, cfg=win_cfg)
        if len(sw["labels"]) == 0:
            continue
        rej = reject_trials(
            sw["X_windows"],
            cfg=RejectionConfig(
                amplitude_uv=cfg["preprocessing"]["rejection"]["amplitude_uv"],
                kurtosis_z=cfg["preprocessing"]["rejection"]["kurtosis_z"],
            ),
        )
        keep = rej["keep_mask"]
        n_rejected_total += rej["n_rejected"]
        n_total += rej["n_total"]
        all_windows.append(sw["X_windows"][keep])
        all_labels.append(sw["labels"][keep])
        all_ends.append(sw["window_end_sample"][keep])
        per_trial_onset.append(onset_sample)

    if not all_windows:
        raise RuntimeError(f"{subject_id}: no usable windows survived rejection -- "
                            f"check onset detection and rejection thresholds before proceeding.")

    X_windows = np.concatenate(all_windows)
    labels = np.concatenate(all_labels)
    window_end_sample = np.concatenate(all_ends)
    feats = window_features(X_windows, eeg_fs)

    logger.log(E3ARunRecord(
        stage=stage, subject_id=subject_id, seed=cfg["seed"], git_commit=_git_commit(),
        mrcp_band_hz=cfg["preprocessing"]["mrcp_band_hz"],
        filter_order=cfg["preprocessing"]["filter_order"],
        laplacian_lambda2=cfg["preprocessing"]["laplacian_lambda2"],
        rejection_amplitude_uv=cfg["preprocessing"]["rejection"]["amplitude_uv"],
        rejection_kurtosis_z=cfg["preprocessing"]["rejection"]["kurtosis_z"],
        n_trials_rejected=n_rejected_total, n_trials_total=n_total,
        window_s=win_cfg.window_s, step_s=win_cfg.step_s,
        horizon_s=win_cfg.horizon_s, rest_margin_s=win_cfg.rest_margin_s,
        onset_threshold_sd=cfg["onset"]["threshold_sd"],
        onset_baseline_window_s=list(cfg["onset"]["baseline_window_s"]),
        split_policy=cfg["split"]["policy"],
        extra={"n_windows": int(len(labels)), "demo_mode": demo_mode},
    ))

    # --- sLDA detector, block-wise split via eval/splits.py only ---
    trial_ids = np.repeat(np.arange(len(all_labels)), [len(l) for l in all_labels])
    sweeps = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        eval_splits.trial_block_split(len(labels), block_ids=trial_ids)
    ):
        clf = anticipation.build_slda()
        _, decision_values = anticipation.slda_fit_predict_fold(
            clf, feats[train_idx], labels[train_idx], feats[test_idx], labels[test_idx]
        )
        # decision_values is aligned with test_idx (same order); build a
        # global-index -> position lookup so per-trial slices can be
        # pulled out without assuming test_idx is sorted/contiguous.
        test_pos = {int(gi): pos for pos, gi in enumerate(test_idx)}
        test_idx_set = set(test_idx.tolist())
        test_trials = np.unique(trial_ids[test_idx])
        for tr in test_trials:
            global_positions = np.array([
                i for i in np.where(trial_ids == tr)[0] if i in test_idx_set
            ])
            if len(global_positions) == 0:
                continue
            local_positions = np.array([test_pos[int(gi)] for gi in global_positions])
            mask = np.zeros(len(labels), dtype=bool)
            mask[global_positions] = True
            sweep = anticipation.threshold_sweep(
                decision_values[local_positions],
                labels[mask], window_end_sample[mask], per_trial_onset[tr], eeg_fs,
                rest_duration_s=win_cfg.rest_margin_s * int((labels[mask] == 0).sum()),
                n_thresholds=cfg["evaluation"]["n_thresholds"],
            )
            sweeps.append(sweep)

        logger.log(E3ARunRecord(
            stage=f"{stage}_fold_{fold_idx}", subject_id=subject_id, seed=cfg["seed"],
            git_commit=_git_commit(), detector_family="slda",
            fold_index=fold_idx, split_policy=cfg["split"]["policy"],
            extra={"demo_mode": demo_mode},
        ))

    curve = anticipation.per_subject_curve_with_ci(sweeps, rng,
                                                     n_bootstrap=cfg["evaluation"]["n_bootstrap_resamples"],
                                                     ci_percent=cfg["evaluation"]["ci_percent"])

    # --- controls: time-reversed surrogate on the same windows ---
    X_rev = anticipation.time_reversed_surrogate(X_windows)
    feats_rev = window_features(X_rev, eeg_fs)
    clf_rev = anticipation.build_slda()
    clf_rev.fit(feats, labels)
    decision_rev = clf_rev.decision_function(feats_rev)
    rev_sweep = anticipation.threshold_sweep(
        decision_rev, labels, window_end_sample, per_trial_onset[0], eeg_fs,
        rest_duration_s=win_cfg.rest_margin_s * int((labels == 0).sum()),
        n_thresholds=cfg["evaluation"]["n_thresholds"],
    )

    return {
        "subject_id": subject_id,
        "curve": curve,
        "surrogate_curve": rev_sweep,
        "n_windows": int(len(labels)),
    }


def make_figure_e3a1(results: list[dict], out_path: Path):
    """Figure E3-A.1: lead time vs. TPR at fixed FP/min, per-subject
    curves plus group median (protocol doc Expected Figures)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    all_tpr = []
    for r in results:
        curve = r["curve"]
        lead = curve["mean_lead_time_s"]["median"]
        tpr = curve["tpr"]["median"]
        ax.plot(tpr, lead, alpha=0.4, label=r["subject_id"])
        all_tpr.append(tpr)
    if all_tpr:
        group_median = np.nanmedian(np.array(all_tpr), axis=0)
        lead_ref = results[0]["curve"]["mean_lead_time_s"]["median"]
        ax.plot(group_median, lead_ref, color="black", linewidth=2.5, label="Group median")
    ax.set_xlabel("True positive rate")
    ax.set_ylabel("Mean detection lead time (s before EMG onset)")
    ax.set_title("Figure E3-A.1: Lead time vs. TPR")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_figure_e3a3(results: list[dict], out_path: Path):
    """Figure E3-A.3: surrogate panel -- curve under time reversal
    (protocol doc: also needs shuffled-onset null, added once that
    control's per-subject aggregation is wired into run_subject)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4 * max(1, len(results)), 4), squeeze=False)
    for ax, r in zip(axes[0], results):
        ax.plot(r["curve"]["tpr"]["median"], label="Real")
        ax.plot(r["surrogate_curve"]["tpr"], label="Time-reversed", linestyle="--")
        ax.set_title(r["subject_id"], fontsize=9)
        ax.legend(fontsize=6)
    fig.suptitle("Figure E3-A.3: Surrogate panel (TPR, time-reversed shown; "
                  "shuffled-onset null pending eval/ onset_shuffle)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = E3ALogger(cfg["logging"]["out_dir"])
    rng = np.random.default_rng(cfg["seed"])

    mods, missing = _try_load_real_modules()
    demo_mode = args.demo or bool(missing)

    if demo_mode:
        print("=" * 70)
        print("SYNTHETIC DEMO MODE.")
        if missing:
            print(f"Missing real modules/data: {missing}")
        print("Results below are NOT an E3-A result. They validate the "
              "orchestration logic only (onset -> windows -> reject -> "
              "detect -> sweep -> controls -> curve -> figure -> log) "
              "against fabricated data with a planted pre-onset shift. "
              "Nothing here is written as if it were a real operating curve.")
        print("=" * 70)

    results = []
    for subject_num in cfg["data"]["participants"]:
        result = run_subject(subject_num, cfg, mods, demo_mode, logger, rng)
        results.append(result)
        print(f"{result['subject_id']}: {result['n_windows']} windows, "
              f"median TPR range [{np.nanmin(result['curve']['tpr']['median']):.2f}, "
              f"{np.nanmax(result['curve']['tpr']['median']):.2f}]")

    fig1_path = Path(cfg["outputs"]["figure_e3a1_path"])
    fig3_path = Path(cfg["outputs"]["figure_e3a3_surrogate_path"])
    if demo_mode:
        fig1_path = fig1_path.parent / "DEMO_ONLY_figure.png"
        fig3_path = fig3_path.parent / "DEMO_ONLY_fig_surrogate_panel.png"

    make_figure_e3a1(results, fig1_path)
    make_figure_e3a3(results, fig3_path)
    print(f"Figures written to {fig1_path} and {fig3_path}")
    print(f"Structured log written to {logger.path}")

    if demo_mode:
        print("\nRe-run without --demo once data/way_eeg_gal.py has been "
              "verified against a real downloaded corpus (see its module "
              "docstring's TODO) and torch is installed for "
              "eeg/decoder_eegnet.py.")

    return results


if __name__ == "__main__":
    main()