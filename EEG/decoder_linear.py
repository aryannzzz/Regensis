"""CSP + shrinkage-LDA decoder for T1-R.
All parameters are injected via config — nothing here assumes a specific
source paper's conditions. Do not hardcode epoch window or band edges.
"""
from dataclasses import dataclass
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

@dataclass
class CSPsLDAConfig:
    n_components: int = 4          # set from prereg once signed
    lda_shrinkage: str | float = "auto"
    reg: str | None = None         # CSP regularization method

def build_pipeline(cfg: CSPsLDAConfig) -> Pipeline:
    csp = CSP(n_components=cfg.n_components, reg=cfg.reg,
              log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis(solver="lsqr",
                                      shrinkage=cfg.lda_shrinkage)
    return Pipeline([("csp", csp), ("lda", lda)])

def fit_predict_fold(pipeline: Pipeline, X_train, y_train, X_test, y_test):
    """One fold. Caller is responsible for supplying eval/splits.py folds —
    this function never generates its own split."""
    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_test), pipeline.decision_function(X_test)