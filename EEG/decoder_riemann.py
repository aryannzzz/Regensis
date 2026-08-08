"""
Smoke test for eeg/decoders_riemann.py.

Uses FABRICATED synthetic data — not BCI-IV-2b, not any real corpus.
This only proves the pipeline runs end-to-end and produces sane shapes;
it says nothing about decoding accuracy on real EEG and must never be
cited as a result (§0.3: results live in results/PR-YYYY-NN/, not here).
"""
import numpy as np
from eeg.decoders_riemann import RiemannConfig, build_pipeline, fit_predict_fold, tangent_space_components


def make_synthetic_epochs(n_epochs=40, n_channels=8, n_times=250, seed=0):
    """
    Two synthetic classes, separable in COVARIANCE structure (not mean).
    Covariance-based decoders (Riemannian tangent-space) are invariant
    to a pure per-channel mean shift — the signal has to live in
    variance/cross-channel correlation to be visible to them at all.
    Class 1 gets elevated variance + added cross-correlation on a
    channel subset, loosely mimicking how ERD manifests as a power
    change rather than a level shift. Not meant to resemble real
    ERD/MRCP structure beyond that.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_epochs, n_channels, n_times))
    y = rng.integers(0, 2, size=n_epochs)

    for i in np.where(y == 1)[0]:
        shared = rng.normal(size=n_times) * 1.5
        X[i, :3, :] += shared  # correlated power increase on 3 channels

    return X, y


def test_pipeline_runs_and_predicts():
    X, y = make_synthetic_epochs()
    split = n = len(y)
    cut = int(0.7 * n)
    X_train, y_train = X[:cut], y[:cut]
    X_test, y_test = X[cut:], y[cut:]

    pipeline = build_pipeline(RiemannConfig())
    preds, decision = fit_predict_fold(pipeline, X_train, y_train, X_test, y_test)

    assert preds.shape == y_test.shape
    assert decision.shape == y_test.shape
    assert set(np.unique(preds)).issubset({0, 1})


def test_tangent_space_components_shape():
    X, y = make_synthetic_epochs()
    pipeline = build_pipeline(RiemannConfig())
    pipeline.fit(X, y)
    ts_components = tangent_space_components(pipeline, X)
    # tangent space dim for n_channels=8 is n_channels*(n_channels+1)/2 = 36
    assert ts_components.shape[0] == X.shape[0]
    assert ts_components.ndim == 2


def test_planted_signal_is_recoverable():
    """
    Sanity check only (not a validated surrogate test): with an
    obviously injected signal, accuracy should clear chance. This is
    a pipeline-health check, not a T1-R result.
    """
    X, y = make_synthetic_epochs(n_epochs=100, seed=1)
    cut = 70
    pipeline = build_pipeline(RiemannConfig())
    preds, _ = fit_predict_fold(pipeline, X[:cut], y[:cut], X[cut:], y[cut:])
    acc = np.mean(preds == y[cut:])
    assert acc > 0.6  # well above chance=0.5 given the injected signal


if __name__ == "__main__":
    test_pipeline_runs_and_predicts()
    test_tangent_space_components_shape()
    test_planted_signal_is_recoverable()
    print("All smoke tests passed.")