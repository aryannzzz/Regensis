# Synthetic worlds

## Generator A: exact mathematical benchmarks

| World | Construction | Known answer | Failure caught |
|---|---|---:|---|
| Entropy sanity | Balanced 1, 2, 4, or 8 values | 0, 1, 2, or 3 bits | Wrong log base or counting |
| Complementary four-class | `G=2A+B`, `M=A`, `E=B` | `I(G;E|M)=1` | Broken CMI identity |
| Noisy complementary | Flip `E=B` with probability `q` | `1-h2(q)` | Noise/channel calculation error |
| Redundant Markov | `G -> observed M -> E` | ordinary MI positive; forward CMI zero | Confusing correlation with complementarity |
| Asymmetric | `M` reveals `A,B`; `E` reveals `C` | forward 1 bit; reverse 2 bits | Reusing the flattering direction |
| Generic Gaussian | scalar `X,Y,Z` with known covariance | `-0.5 log2(1-rho_partial^2)` | Gaussian/oracle numerical error |

Exact and sampled variants are separate. Sampled plug-in bias is measured rather than mistaken for a change in ground truth.

## Generator B: causal feature worlds

Generator B is a causal statistical abstraction, not a complete physiological model. Independent Gaussian sources are mixed into multichannel features. Central channels receive relatively stronger cortical weights; peripheral/frontal/temporal and high-band groups receive relatively stronger myogenic weights. All profiles and random parameter seeds are saved.

### Null

Forearm EMG contains grasp information. Recorded EEG is independent of target and EMG. The true forward CMI is zero. This catches upward estimator bias and false-positive testing errors.

### Exactly redundant EEG

Observed EEG is a noisy linear transformation of observed EMG:

\[
G\rightarrow M\rightarrow E.
\]

EEG has ordinary target information, but its forward CMI given rich `M` is exactly zero. Sparse conditioning can make a positive value legitimately reappear because it no longer conditions on the full Markov mediator.

### Genuine cortical

The target shifts a cortical-unique component. Cranial activity has no target shift. Raw and true-`C`-conditioned forward CMI should both be positive. The cortical spatial profile favors central channels and mu/beta feature groups, but those labels are stress-test metadata, not biological proof.

### Artifact only

The cortical component has no target information. A target/effort-related cranial component enters EEG. Raw forward CMI is positive; conditioning on true `C` or subtracting `W_C C` makes it zero. This is the central false-positive world.

### Cortical plus artifact

Both cortical and cranial components carry target information. Raw forward CMI is positive. True-`C` conditioning makes it smaller but leaves a positive cortical remainder.

## Subject/block modes

Debug mode uses identical subject parameters, balanced classes, and no drift. Stress mode supports subject-specific mixing/gains/noise, class imbalance, and block offsets. The stress oracle conditions on the known subject/block key so it measures the same quantity as the configured comparison.

Named-world cortical, artifact, shared-EEG, shared-motor, and EMG-unique strengths can be overridden in `GeneratorConfig`. This supports targeted zero/very-small/small/medium/large matrices without changing causal code or manually editing results.

## Windows and representations

Pre- and post-onset datasets are generated and reported separately. Rich, intermediate, and sparse EMG representations are deterministic declared projections. EEG indices can be selected by full/central montage and by band group without changing the underlying generated trial.
