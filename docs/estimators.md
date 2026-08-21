# Estimators

Every estimator returns the same auditable fields: estimate in bits, direction, name, sample size, seed, runtime, diagnostics, and warnings. Negative finite-sample values are retained.

## Exact discrete plug-in

For genuinely discrete variables,

\[
I(X;Y\mid Z)=H(X,Z)+H(Y,Z)-H(Z)-H(X,Y,Z).
\]

The implementation also evaluates `H(X|Z)-H(X|Y,Z)` and asserts that the identities agree. It exactly recovers all deterministic Generator A answers. It is **not applicable** to multivariate continuous EEG/EMG features and is not a future E2 candidate.

## Gaussian analytic reference

For the generic scalar Gaussian benchmark,

\[
I(X;Y\mid Z)=-\frac12\log_2(1-\rho_{XY\cdot Z}^2).
\]

A multivariate log-determinant form is also implemented. This is an oracle/reference calculation only when joint Gaussian assumptions actually hold. The biological target `G` is never converted to a continuous number to use it.

## Nested decoder/log-loss difference

Matched multinomial logistic models estimate

\[
\widehat I_{dec}=CE(G\mid M)-CE(G\mid M,E)
\]

on untouched outer folds, with natural-log loss divided by `ln(2)`. The reverse direction swaps the base channel. Scaling and tuning happen within training folds. Both arms receive the same model family, folds, `C` grid, and search count. Accuracy, multiclass Brier score, and expected calibration error are reported, but the information interpretation is based on log loss.

This estimate is decoder-dependent. Model misspecification can make it negative or strongly underestimate the Bayes value. Primary validation retained those negatives. Its aggregate bias was -0.216 bits, RMSE was 0.354 bits, normal-interval coverage was 23.3%, small-effect detection was 5.6%, and the disjoint-seed null false-positive rate was 0%.

## Direct mixed-data CMIh

The direct estimator is a clean implementation of the hybrid entropy decomposition from Lei Zan et al., *A Conditional Mutual Information Estimator for Mixed Data and an Associated Conditional Independence Test* (Entropy, 2022): <https://doi.org/10.3390/e24091234>.

The method decomposes each mixed entropy as

\[
H(D,C)=H(D)+\sum_d p(d)H(C\mid D=d),
\]

uses exact frequencies for qualitative components, and a local maximum-norm k-nearest-neighbor entropy estimate for continuous components. Four such mixed entropies form CMI. `G` remains categorical; only continuous features are standardized and receive machine-scale tie breaking.

The authors' reference code was inspected at <https://github.com/leizan/CMIh2022>; it explicitly supports multivariate mixed variables and is MIT licensed. The repository also points to Tigramite's maintained `CMIknnMixed` implementation. No third-party source was copied or vendored here.

CMIh is fast and supports the real variable types directly. In the primary matrix its aggregate bias was -0.093 bits, RMSE was 0.403 bits, normal-interval coverage was 20.0%, small-effect detection was 33.3%, and the disjoint-seed null false-positive rate was 15%. It also substantially overestimated several reverse-direction effects. Its `k` fraction must therefore be validated and pre-registered; current results do not justify selecting it for real E2.

## Independent Bayes oracle

The oracle is not a candidate estimator. It uses the generator's known class priors, means, covariances, and exact Gaussian likelihoods, then estimates

\[
\mathbb E\left[\log_2\frac{p(G\mid M,E)}{p(G\mid M)}\right]
\]

on a large independent Monte Carlo sample. It reports Monte Carlo standard error and whether its target precision was achieved. Analytic discrete/Gaussian checks validate its likelihood-ratio convention and base conversion.

## Current recommendation

No continuous estimator is ready for real E2.

The nested matched decoder is the most defensible **provisional diagnostic** because it enforces fold discipline and equal tuning budgets, had the lower primary RMSE, and did not exceed its disjoint-seed null threshold. It still showed material negative bias, poor interval coverage, and only 5.6% small-effect detection. CMIh remains a useful direct comparator but had higher RMSE, a 15% null false-positive rate, and strong direction/dimension sensitivity.

Before real data, add repeated group-bootstrap uncertainty, tune CMIh only on training simulations, and determine whether a richer but equally budgeted calibrated decoder closes the oracle gap. The completed primary matrix is validation evidence, not estimator clearance.
