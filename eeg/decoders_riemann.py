"""
eeg/decoders_riemann.py — Riemannian tangent-space decoder for T1-R.

Matches the interface of decoders_linear.py so experiments/*.yaml can
select the feature family by config (§5.1: "CSP or Riemannian
tangent-space features, matching the source paper's family").

All parameters are injected via config — nothing here assumes a
specific source paper's epoch window or band edges. Those come from
the signed pre-registration.

Pipeline: covariance estimation -> tangent-space projection -> sLDA.
This keeps the classifier identical across both feature families
(§5: "three decoder families implemented so that method effect can be
separated from signal effect") — only the feature extraction differs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


@dataclass
class RiemannConfig:
    cov_estimator: str = "oas"     # shrinkage covariance estimator
    lda_shrinkage: str | float = "auto"


def build_pipeline(cfg: RiemannConfig) -> Pipeline:
    """
    Build the covariance -> tangent-space -> sLDA pipeline.

    Input to .fit/.predict is raw epoched signal, shape
    (n_epochs, n_channels, n_times) — NOT pre-computed features.
    Covariances is fit inside the pipeline so it respects whatever
    fold `eval/splits.py` hands it (preprocessing statistics fitted
    within fold, per §0.1).
    """
    cov = Covariances(estimator=cfg.cov_estimator)
    ts = TangentSpace()
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=cfg.lda_shrinkage)
    return Pipeline([("cov", cov), ("ts", ts), ("lda", lda)])


def fit_predict_fold(pipeline: Pipeline, X_train, y_train, X_test, y_test):
    """
    One fold. Caller supplies the fold from eval/splits.py — this
    function never generates its own split (§0.1: splitters come from
    eval/ only).
    """
    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_test), pipeline.decision_function(X_test)


def tangent_space_components(pipeline: Pipeline, X):
    """
    Project X through the fitted covariance + tangent-space steps only
    (skip the classifier), for the Checkpoint step in §5.1's pipeline:
    'Plot the first two CSP/tangent-space components per class' before
    proceeding to classification.
    """
    cov = pipeline.named_steps["cov"].transform(X)
    return pipeline.named_steps["ts"].transform(cov)