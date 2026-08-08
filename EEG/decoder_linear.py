"""
eeg/decoders_linear.py — CSP + shrinkage-LDA decoder for T1-R (§5.1, step 3-4).

Interface mirrors eeg/decoders_riemann.py so experiments/t1r_reproduction.yaml
can select the feature family by config alone (§5.1: "CSP or Riemannian
tangent-space features, matching the source paper's family").

All numeric parameters are injected via config (CSPsLDAConfig) — nothing
here hardcodes epoch window or band edges. Those come from
preprocessing/epoching.py and eeg_filters.py (Developer A), driven by
configs/t1r_base.yaml, driven by the signed prereg/PR-2026-01.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


@dataclass
class CSPsLDAConfig:
    n_components: int = 4          # even number; CSP keeps top/bottom pairs
    reg: str | float | None = "ledoit_wolf"   # CSP spatial-cov regularization
    lda_shrinkage: str | float = "auto"
    log: bool = True               # log-variance features (standard CSP)


def build_pipeline(cfg: CSPsLDAConfig) -> Pipeline:
    """
    Build the CSP -> sLDA pipeline.

    Input to .fit/.predict is band-passed, epoched signal, shape
    (n_epochs, n_channels, n_times) — NOT pre-computed features. CSP is
    fit inside the pipeline so its spatial filters are learned only on
    whatever fold eval/splits.py hands it (no fitting on held-out data).
    """
    csp = CSP(n_components=cfg.n_components, reg=cfg.reg,
              log=cfg.log, norm_trace=False)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=cfg.lda_shrinkage)
    return Pipeline([("csp", csp), ("lda", lda)])


def fit_predict_fold(pipeline: Pipeline, X_train, y_train, X_test, y_test):
    """
    One fold. Caller supplies the fold from eval/splits.py (session_split
    or trial_block_split via get_split) — this function never generates
    its own split (§0.1/§8.1: splitters come from eval/ only).

    Returns (predictions, decision_values) for X_test — both needed
    downstream: predictions for accuracy, decision_values if a
    calibration/ROC step is ever added.
    """
    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_test), pipeline.decision_function(X_test)


def csp_components(pipeline: Pipeline, X):
    """
    Project X through the fitted CSP step only (skip the classifier),
    for the Checkpoint step in §5.1: 'plot the first two CSP or
    tangent-space components per class' before proceeding to
    classification. Returns log-variance features per component if
    cfg.log=True, else raw CSP-filtered variance.
    """
    return pipeline.named_steps["csp"].transform(X)