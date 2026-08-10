"""
experiments/run_t1r.py — orchestrates T1-R end to end (§5.1, §8.2).

    python -m experiments.run_t1r --config experiments/t1r_reproduction.yaml

INTERFACE CONTRACT this orchestrator expects from Developer A's modules.
CONFIRMED against the real files as of the latest integration pass:

  data.bci_iv_2a.load_subject(data_dir: Path, subject_num: int) -> BCIIV2aData:
      .raw_by_session:    dict["T"|"E" -> mne.io.Raw]  (2 sessions, not 5 -- 2a
                           has one T and one E session per subject, unlike 2b)
      .events_by_session: dict[session_id -> np.ndarray (n_events, 3)],
                           already filtered to left/right-hand trials only
      .labels_by_session: dict[session_id -> np.ndarray (n_events,), 0/1]
      .fs:                float

  data.bci_iv_2a.compute_dataset_hash(data_dir: Path) -> str

  preprocessing.eeg_filters.bandpass_filter(raw, low: float, high: float,
      order: int = 4, picks: list | None = None) -> mne.io.Raw (filtered)
    NOTE: real signature is bandpass_filter(...), not bandpass(...), and
    uses low/high not l_freq/h_freq -- this orchestrator calls it by the
    confirmed real name/kwargs below, not the originally-guessed ones.

  preprocessing.epoching.epoch(raw, events, labels, tmin: float, tmax: float,
      baseline: tuple[float, float] | None) -> (X, y)
      X: np.ndarray (n_epochs, n_channels, n_times)
      y: np.ndarray (n_epochs,)
    STILL NOT RECEIVED as of this integration pass -- this is the one
    remaining piece blocking a real (non-demo) run even with real data
    files present, since bandpass_filter and load_subject alone don't
    produce epoched arrays.

If any of these aren't importable, this script falls back to a
clearly-labeled SYNTHETIC DEMO MODE so the orchestration logic itself
(splits -> decoder -> surrogate -> stats -> figure -> log) is provably
correct today. Demo-mode runs are tagged in the log and refuse to write
into results/<PR>/ as if they were a real result.
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
from eval import surrogates as eval_surrogates
from eval import stats as eval_stats
from eeg import decoders_linear
from eeg import decoders_riemann
from experiments.logging_t1r import RunRecord, T1RLogger


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _require(cfg: dict, path: str, logger: T1RLogger, stage: str):
    """Walk a dotted path in cfg; raise loudly (per §8.4) if the value is
    still the null placeholder — refuses to silently run with a default."""
    node = cfg
    for key in path.split("."):
        node = node[key]
    if node is None:
        logger.log_missing_param_error(stage, path)
    return node


def _try_load_dev_a_modules():
    """Import Developer A's modules if present; report which are missing
    rather than failing on the first one, so the blocker list is complete
    in one pass."""
    missing = []
    mods = {}
    try:
        from data import bci_iv_2a
        mods["bci_iv_2a"] = bci_iv_2a
    except ImportError:
        missing.append("data/bci_iv_2a.py")
    try:
        from preprocessing import eeg_filters
        if not hasattr(eeg_filters, "bandpass_filter"):
            missing.append(
                "preprocessing/eeg_filters.py (found, but no "
                "bandpass_filter function -- interface mismatch)"
            )
        else:
            mods["eeg_filters"] = eeg_filters
    except ImportError:
        missing.append("preprocessing/eeg_filters.py")
    try:
        from preprocessing import epoching
        mods["epoching"] = epoching
    except ImportError:
        missing.append("preprocessing/epoching.py")
    return mods, missing


def _synthetic_subject_epochs(subject_num: int, n_channels: int = 22,
                               n_times: int = 500, seed: int = 0):
    """
    SYNTHETIC DEMO MODE ONLY. Fabricated epoched data shaped like what
    preprocessing/epoching.py will hand this orchestrator once it lands
    (data/bci_iv_2a.py and preprocessing/eeg_filters.py are real now;
    epoching.py is still the missing piece). Matches 2a's real structure
    confirmed from the actual loader: 2 sessions per subject (T, E),
    144 left/right-hand trials per session (72/class, out of 288 total
    4-class trials). Not real BCI-IV-2a data. Never used to produce a
    figure written into results/<PR>/ (see main()'s demo-mode guard).
    """
    rng = np.random.default_rng(seed + subject_num)
    n_per_session = 144
    X_list, y_list, session_list = [], [], []
    for session_id in ["T", "E"]:
        X = rng.normal(size=(n_per_session, n_channels, n_times))
        y = rng.integers(0, 2, size=n_per_session)
        # covariance-domain separable signal (see decoders_riemann smoke
        # test note: a pure mean shift is invisible to covariance methods
        # and CSP's log-variance features alike need a variance change)
        for i in np.where(y == 1)[0]:
            shared = rng.normal(size=n_times) * 1.3
            X[i, :4, :] += shared
        X_list.append(X)
        y_list.append(y)
        session_list.append(np.full(n_per_session, session_id))
    return (np.concatenate(X_list), np.concatenate(y_list),
            np.concatenate(session_list))


def run_subject(subject_num: int, cfg: dict, mods: dict, demo_mode: bool,
                 logger: T1RLogger, rng: np.random.Generator):
    stage = f"subject_{subject_num}"

    if demo_mode:
        X, y, session_ids = _synthetic_subject_epochs(subject_num, seed=cfg["seed"])
    else:
        data_dir = Path(_require(cfg, "data.data_dir", logger, stage))
        subj = mods["bci_iv_2a"].load_subject(data_dir, subject_num)
        band_lo, band_hi = _require(cfg, "preprocessing.band_pass_hz", logger, stage)
        filt_order = _require(cfg, "preprocessing.filter_order", logger, stage)
        tmin, tmax = _require(cfg, "preprocessing.epoch_window_s", logger, stage)
        baseline = cfg["preprocessing"]["baseline_window_s"]  # optional

        X_list, y_list, session_list = [], [], []
        for session_id, raw in subj.raw_by_session.items():
            # Confirmed real signature: bandpass_filter(raw, low, high,
            # order, picks) -- not bandpass(raw, l_freq, h_freq, order)
            # as originally guessed before eeg_filters.py was shared.
            filtered = mods["eeg_filters"].bandpass_filter(
                raw, low=band_lo, high=band_hi, order=filt_order
            )
            events = subj.events_by_session[session_id]
            labels = subj.labels_by_session[session_id]
            X_sess, y_sess = mods["epoching"].epoch(
                filtered, events, labels, tmin=tmin, tmax=tmax, baseline=baseline
            )
            X_list.append(X_sess)
            y_list.append(y_sess)
            session_list.append(np.full(len(y_sess), session_id))
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)
        session_ids = np.concatenate(session_list)

    # --- feature family + decoder (config-selected, §5.1 step 3-4) ---
    family = cfg["features"]["family"]
    if family == "csp":
        pipeline_cfg = decoders_linear.CSPsLDAConfig(**cfg["features"]["csp"])
        build_pipeline = decoders_linear.build_pipeline
        fit_predict_fold = decoders_linear.fit_predict_fold
    elif family == "riemann":
        pipeline_cfg = decoders_riemann.RiemannConfig(**cfg["features"]["riemann"])
        build_pipeline = decoders_riemann.build_pipeline
        fit_predict_fold = decoders_riemann.fit_predict_fold
    else:
        raise ValueError(f"Unknown feature family: {family}")

    # --- split: block-wise, from eval/splits.py only (§5.1 step 5) ---
    fold_accuracies = []
    all_preds, all_true = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(
        eval_splits.get_split(session_ids=session_ids)
    ):
        pipeline = build_pipeline(pipeline_cfg)
        preds, _ = fit_predict_fold(
            pipeline, X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        )
        acc = float(np.mean(preds == y[test_idx]))
        fold_accuracies.append(acc)
        all_preds.append(preds)
        all_true.append(y[test_idx])

        logger.log(RunRecord(
            stage=f"{stage}_fold_{fold_idx}",
            band_edges_hz=cfg["preprocessing"]["band_pass_hz"],
            filter_order=cfg["preprocessing"].get("filter_order"),
            epoch_window_s=cfg["preprocessing"].get("epoch_window_s"),
            baseline_window_s=cfg["preprocessing"].get("baseline_window_s"),
            split_policy="session_wise",
            fold_index=fold_idx,
            subject_id=f"A{subject_num:02d}",
            seed=cfg["seed"],
            git_commit=_git_commit(),
            extra={"fold_accuracy": acc, "feature_family": family,
                   "demo_mode": demo_mode},
        ))

    # --- label-shuffle surrogate + permutation test (§5.1 step 6) ---
    n_surrogates = cfg["evaluation"]["n_surrogates"]
    null_accs = np.empty(n_surrogates)
    for s in range(n_surrogates):
        shuffled_y = eval_surrogates.label_shuffle(y, rng)
        # reuse the same split structure for the null, fresh pipeline per draw
        accs = []
        for train_idx, test_idx in eval_splits.get_split(session_ids=session_ids):
            pipeline = build_pipeline(pipeline_cfg)
            preds, _ = fit_predict_fold(
                pipeline, X[train_idx], shuffled_y[train_idx],
                X[test_idx], shuffled_y[test_idx]
            )
            accs.append(np.mean(preds == shuffled_y[test_idx]))
        null_accs[s] = np.mean(accs)

    real_stat = float(np.mean(fold_accuracies))
    perm_result = eval_stats.permutation_test(real_stat, null_accs, alternative="greater")

    # --- bootstrap CI on per-subject accuracy (§5.1 step 7) ---
    ci_result = eval_stats.bootstrap_ci(
        np.array(fold_accuracies),
        n_resamples=cfg["evaluation"]["n_bootstrap_resamples"],
        ci=cfg["evaluation"]["ci_percent"],
        rng=rng,
    )

    return {
        "subject_id": f"A{subject_num:02d}",
        "mean_accuracy": real_stat,
        "ci_low": ci_result["ci_low"],
        "ci_high": ci_result["ci_high"],
        "surrogate_p_value": perm_result["p_value"],
        "fold_accuracies": fold_accuracies,
    }


def make_figure(results: list[dict], published: dict, out_path: Path):
    """Figure T1-R.1: per-subject reproduced-vs-published scatter, identity
    line, CIs (§5.1 expected figures)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    xs, ys, yerr_lo, yerr_hi = [], [], [], []
    for r in results:
        pub = published.get(r["subject_id"])
        if pub is None:
            continue
        xs.append(pub)
        ys.append(r["mean_accuracy"])
        yerr_lo.append(r["mean_accuracy"] - r["ci_low"])
        yerr_hi.append(r["ci_high"] - r["mean_accuracy"])

    if xs:
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="o", capsize=3,
                     label="Per-subject (this run)")
        lims = [min(xs + ys) - 0.05, max(xs + ys) + 0.05]
        ax.plot(lims, lims, "--", color="gray", label="Identity")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    else:
        ax.text(0.5, 0.5, "No published_reference values in config yet\n"
                           "(TODO(PR-2026-01) — see t1r_reproduction.yaml)",
                 ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Published accuracy")
    ax.set_ylabel("Reproduced accuracy (this pipeline)")
    ax.set_title("Figure T1-R.1: Reproduced vs. published accuracy")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--demo", action="store_true",
                         help="Force synthetic demo mode even if Dev A's "
                              "modules are importable (for testing).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger = T1RLogger(cfg["logging"]["out_dir"])
    rng = np.random.default_rng(cfg["seed"])

    mods, missing = _try_load_dev_a_modules()
    demo_mode = args.demo or bool(missing)

    if demo_mode:
        print("=" * 70)
        print("SYNTHETIC DEMO MODE.")
        if missing:
            print(f"Missing Developer A modules: {missing}")
        print("Results below are NOT a T1-R result. They validate the "
              "orchestration logic only (split -> decoder -> surrogate -> "
              "stats -> figure -> log) against fabricated data. Nothing "
              "here is written as if it were a real reproduced accuracy.")
        print("=" * 70)

    results = []
    for subject_num in cfg["data"]["subjects"]:
        result = run_subject(subject_num, cfg, mods, demo_mode, logger, rng)
        results.append(result)
        print(f"{result['subject_id']}: acc={result['mean_accuracy']:.3f} "
              f"CI=[{result['ci_low']:.3f}, {result['ci_high']:.3f}] "
              f"surrogate p={result['surrogate_p_value']:.4f}")

    fig_path = Path(cfg["outputs"]["figure_path"])
    if demo_mode:
        fig_path = fig_path.parent / "DEMO_ONLY_figure.png"
    make_figure(results, cfg["published_reference"]["per_subject_accuracy"], fig_path)
    print(f"Figure written to {fig_path}")
    print(f"Structured log written to {logger.path}")

    if demo_mode:
        print("\nRe-run without --demo once data/bci_iv_2a.py, "
              "preprocessing/eeg_filters.py, and preprocessing/epoching.py "
              "are on track-a, and preprocessing.*_s / band_pass_hz are "
              "filled from the signed prereg.")

    return results


if __name__ == "__main__":
    main()