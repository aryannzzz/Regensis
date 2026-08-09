"""
tests/test_decoders_linear.py

Verifies eeg/decoders_linear.py mechanically: pipeline runs, shapes are
sane, and — per §5.1's failure-mode guidance ("reproduce far above the
published value -> suspect leakage before celebrating") — that a
label-shuffled control collapses toward chance instead of also scoring
high. All data here is fabricated; nothing in this file is a T1-R result.
"""
import numpy as np
from eeg.decoders_linear import CSPsLDAConfig, build_pipeline, fit_predict_fold, csp_components


def make_synthetic_epochs(n_epochs=80, n_channels=22, n_times=500, seed=0):
    """CSP operates on log-variance of spatially filtered signal, so the
    planted signal needs a genuine variance/covariance change per class
    (a pure mean shift is invisible to it, same as for the Riemannian
    decoder — see tests/test_decoder_riemann_smoke.py)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_epochs, n_channels, n_times))
    y = rng.integers(0, 2, size=n_epochs)
    for i in np.where(y == 1)[0]:
        X[i, :4, :] *= 1.8  # variance increase on a channel subset
    return X, y


def test_pipeline_runs_and_predicts():
    X, y = make_synthetic_epochs()
    cut = int(0.7 * len(y))
    pipeline = build_pipeline(CSPsLDAConfig(n_components=4))
    preds, decision = fit_predict_fold(pipeline, X[:cut], y[:cut], X[cut:], y[cut:])
    assert preds.shape == y[cut:].shape
    assert decision.shape == y[cut:].shape
    assert set(np.unique(preds)).issubset({0, 1})


def test_csp_components_shape():
    X, y = make_synthetic_epochs()
    pipeline = build_pipeline(CSPsLDAConfig(n_components=4))
    pipeline.fit(X, y)
    comps = csp_components(pipeline, X)
    assert comps.shape == (X.shape[0], 4)


def test_planted_signal_is_recoverable():
    """Pipeline-health check only, not a validated surrogate test:
    accuracy should clear chance on an obviously separable signal."""
    X, y = make_synthetic_epochs(n_epochs=120, seed=1)
    cut = 84
    pipeline = build_pipeline(CSPsLDAConfig(n_components=4))
    preds, _ = fit_predict_fold(pipeline, X[:cut], y[:cut], X[cut:], y[cut:])
    acc = np.mean(preds == y[cut:])
    assert acc > 0.6


def test_label_shuffle_collapses_toward_chance():
    """
    The check §5.1 asks you to run mentally on every real result: does a
    label-shuffled version of this same pipeline still score high? If it
    does, something leaks. Averaged over several shuffles, accuracy
    should sit near 0.5, not near the real (separable) accuracy.
    """
    from eval.surrogates import label_shuffle

    X, y = make_synthetic_epochs(n_epochs=120, seed=2)
    cut = 84
    rng = np.random.default_rng(0)

    shuffled_accs = []
    for _ in range(10):
        y_shuffled = label_shuffle(y, rng)
        pipeline = build_pipeline(CSPsLDAConfig(n_components=4))
        preds, _ = fit_predict_fold(
            pipeline, X[:cut], y_shuffled[:cut], X[cut:], y_shuffled[cut:]
        )
        shuffled_accs.append(np.mean(preds == y_shuffled[cut:]))

    assert 0.30 < np.mean(shuffled_accs) < 0.70, (
        f"Label-shuffled accuracy {np.mean(shuffled_accs):.3f} is far from "
        f"chance -- if this ever happens on real data, stop and check the "
        f"split before trusting any real-label result (§5.1)."
    )


if __name__ == "__main__":
    test_pipeline_runs_and_predicts()
    test_csp_components_shape()
    test_planted_signal_is_recoverable()
    test_label_shuffle_collapses_toward_chance()
    print("All decoders_linear tests passed.")