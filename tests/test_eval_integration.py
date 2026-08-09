"""
tests/test_eval_integration.py

Verifies eval/splits.py, eval/surrogates.py, eval/stats.py integrate
correctly with the decoders -- the actual wiring your orchestrator
depends on, not just each piece in isolation. All data fabricated.
"""
import numpy as np
import pytest

from eval import splits as eval_splits
from eval import stats as eval_stats
from eval import surrogates as eval_surrogates
from eeg.decoders_linear import CSPsLDAConfig, build_pipeline, fit_predict_fold


def test_session_split_train_test_disjoint():
    """The one bug the manual treats as catastrophic: any overlap between
    train and test indices. Verify it directly rather than trusting it."""
    session_ids = np.array(["T"] * 144 + ["E"] * 144)
    for train_idx, test_idx in eval_splits.get_split(session_ids=session_ids):
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(train_idx) + len(test_idx) == len(session_ids)


def test_get_split_rejects_pooled_single_session():
    """get_split must not silently produce a degenerate split when only
    one session exists for a subject -- trial_block_split is the correct
    fallback, not a same-session leak."""
    session_ids = np.array(["T"] * 50)  # only one session
    folds = list(eval_splits.get_split(session_ids=session_ids, n_blocks=5))
    assert len(folds) == 5  # fell through to trial_block_split
    for train_idx, test_idx in folds:
        assert len(set(train_idx) & set(test_idx)) == 0


def test_bootstrap_ci_enforces_resample_floor():
    """§5.1 sets a hard floor of >=1000 resamples. Confirm the floor is
    actually enforced, not just documented."""
    with pytest.raises(ValueError):
        eval_stats.bootstrap_ci(np.array([0.7, 0.8, 0.75]), n_resamples=100)


def test_full_wiring_one_subject():
    """
    End-to-end for one synthetic subject: split -> decoder -> label-shuffle
    surrogate -> bootstrap CI -> permutation test. This is exactly the
    sequence run_t1r.py runs per subject, just without the config/logging
    layer around it, so it's fast to run in isolation.
    """
    rng = np.random.default_rng(0)
    n_channels, n_times = 22, 500
    session_ids = np.array(["T"] * 144 + ["E"] * 144)
    y = rng.integers(0, 2, size=288)
    X = rng.normal(size=(288, n_channels, n_times))
    for i in np.where(y == 1)[0]:
        X[i, :4, :] *= 1.8

    cfg = CSPsLDAConfig(n_components=4)
    fold_accs = []
    for train_idx, test_idx in eval_splits.get_split(session_ids=session_ids):
        pipeline = build_pipeline(cfg)
        preds, _ = fit_predict_fold(pipeline, X[train_idx], y[train_idx],
                                     X[test_idx], y[test_idx])
        fold_accs.append(np.mean(preds == y[test_idx]))

    ci = eval_stats.bootstrap_ci(np.array(fold_accs), n_resamples=1000, rng=rng)
    assert ci["ci_low"] <= ci["median"] <= ci["ci_high"]

    null_accs = []
    for _ in range(15):  # kept small for test speed; real runs use config n_surrogates
        y_shuffled = eval_surrogates.label_shuffle(y, rng)
        accs = []
        for train_idx, test_idx in eval_splits.get_split(session_ids=session_ids):
            pipeline = build_pipeline(cfg)
            preds, _ = fit_predict_fold(pipeline, X[train_idx], y_shuffled[train_idx],
                                         X[test_idx], y_shuffled[test_idx])
            accs.append(np.mean(preds == y_shuffled[test_idx]))
        null_accs.append(np.mean(accs))

    real_stat = float(np.mean(fold_accs))
    perm = eval_stats.permutation_test(real_stat, np.array(null_accs), alternative="greater")
    assert 0.0 < perm["p_value"] <= 1.0


if __name__ == "__main__":
    test_session_split_train_test_disjoint()
    test_get_split_rejects_pooled_single_session()
    test_bootstrap_ci_enforces_resample_floor()
    test_full_wiring_one_subject()
    print("All eval integration tests passed.")