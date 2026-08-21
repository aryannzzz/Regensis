# Scientific design

## Quantity and target

`G` is one configurable categorical variable: abstract intended grasp type. The default has four balanced values. `E` and `M` are multivariate continuous EEG-like and forearm-EMG-like feature vectors. The framework reports

\[
I(G;E\mid M)=H(G\mid M)-H(G\mid M,E)
\]

and, independently,

\[
I(G;M\mid E)=H(G\mid E)-H(G\mid E,M).
\]

The directions are not assumed equal. The exact three-bit benchmark makes them 1 and 2 bits respectively and fails if code silently reuses one direction.

## Independent unit and splits

One trial is the default independent unit. Metadata always retain `subject_id`, `block_id`, `trial_id`, `trial_uid`, `window_id`, and window class. Reusable split utilities support whole-trial, whole-block, and whole-subject folds. Automated assertions reject trial, block, or subject overlap as appropriate.

The decoder uses nested group cross-validation. The untouched outer fold supplies probabilities for the reported log loss. Hyperparameters are chosen on inner group folds. Baseline and augmented arms use the same classifier family, folds, parameter grid, and number of candidates. Scaling lives inside each fitted pipeline; diagnostics record that each scaler saw only its training fold.

## Pre-onset versus post-onset

The generator is called separately for `pre` and `post`; no primary result pools them.

- Pre-onset EMG target strength is near zero. Positive EEG CMI represents simulated anticipation or temporal precedence.
- Post-onset both channels may be informative. Positive EEG CMI represents simultaneously available complementarity.

These are distinct scientific interpretations, even if their numerical units match.

## Myogenic false positive

The artifact-only world contains no target-related cortical component. A target-related cranial/jaw/neck component enters recorded EEG, while the conditioning set contains forearm EMG. The resulting raw \(I(G;E\mid M)\) is positive even though its origin is entirely muscular.

Five complementary controls expose this mechanism:

1. Band-resolved features: myogenic loadings are strongest in the configurable `high` group.
2. Montage: artifact loadings are stronger over peripheral/frontal/temporal channels than central channels.
3. Myogenic conditioning: the known `C` and imperfect proxies can be included in the conditioning set.
4. Matched degradation: the EEG-side muscle signal is attenuated alongside forearm EMG without attenuating cortical `K`.
5. Discriminating surrogate: target-related cortical structure is permuted/intervened on while `G`, `M`, effort, and cranial artifact remain fixed.

The ideal `E - W_C C` diagnostic is reported separately from realistic proxy conditioning. It is possible only because this is synthetic data.

## EMG degradation

The simulator stores forearm signal and sensor noise separately:

\[
M_\alpha=\alpha S_M+N_M.
\]

At `alpha=0`, observed EMG equals its retained noise. This loses target information. Uniformly scaling observed `M` would be invertible and is explicitly tested against.

In the ordinary sweep, EEG artifact is untouched. In the matched sweep,

\[
E_\alpha=K+W_C(\alpha S_C+N_C)+N_E.
\]

Thus only the EEG-side muscle signal is degraded; the genuine cortical component is never attenuated as a side effect.

## Conditioning richness

The rich representation retains all EMG features. The intermediate representation averages declared channel groups. The sparse representation is a fixed one-dimensional information-losing projection and is labelled as such; it is not called an NMF synergy.

The redundant world demonstrates the danger cleanly: EEG is exactly redundant given rich observed EMG, yet information discarded by sparse conditioning reappears as apparent unique EEG information. Strict monotonicity is interpreted only for parameterizations where it is observed/guaranteed.

## Nulls and false positives

A label shuffle destroys every relationship between target and neural or muscular signals. It is useful as a generic no-association check, but an artifact-only effect can exceed this null because the artifact genuinely carries target information.

The discriminating surrogate instead preserves target, EMG, effort, and myogenic contamination while removing target-related `K`. Artifact-only observed and surrogate estimates should agree; genuine and mixed worlds should fall after intervention.

False-positive rates in the scorecard do not count every positive estimate. The first half of an independent set of known-null simulator seeds defines a one-sided 95th-percentile threshold; the second half, using disjoint seeds, estimates the false-positive rate.

## Oracle alignment

The simulator retains the exact class prior, mean, covariance, mixing matrices, and subject/block parameter seeds. The Bayes oracle samples a large independent Monte Carlo set and evaluates posterior probabilities with stable Gaussian log likelihoods and `logsumexp`. The small estimator-evaluation sample is never reused.

Oracle sampling grows until its standard error meets the configured target or reaches a safe cap. If stress-mode drift is enabled, the oracle conditions explicitly on known subject/block metadata; it never compares an unconditional estimator with a differently conditioned truth without saying so.

Debug-mode subjects have identical generating parameters, so equal per-subject oracle points are expected and labelled as such. Stress-mode reports restrict each per-subject oracle to that subject's block keys rather than duplicating a pooled value.
