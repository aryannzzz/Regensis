# Limitations

- These are synthetic multichannel features, not raw EEG/EMG waveforms. Filtering, spectral estimation, ICA, onset detection, phase randomization, and end-to-end preprocessing are not validated here.
- The causal graph is deliberately simplified and mostly linear-Gaussian. It captures dependencies relevant to conditional information and artifact controls, not complete neurophysiology.
- Synthetic success validates software behavior only. It does not establish cortical grasp information in humans and does not confirm Regenesis H1.
- `G` is currently an abstract categorical intended-grasp-type variable. Real-data labels, trial exclusions, onset semantics, and feature schemas remain future decisions.
- The oracle can condition on known simulator subject/block state. Real analyses will not have all latent or nuisance variables, so the estimand must be aligned explicitly.
- True cranial component conditioning and `E-W_C C` are ideal synthetic controls. Real myogenic proxies will be imperfect; residual CMI after an imperfect proxy is not proof of cortical origin.
- Band names and montage groups are configurable feature groups. High frequency or central location alone cannot prove muscular or cortical origin.
- Simulated EMG degradation removes a declared signal component while retaining noise. It is not a physiological model of amputation, reinnervation, electrode lift, fatigue, or population transfer.
- Decoder CMI is model-dependent and can be negative. The current logistic family underestimates several oracle effects.
- The direct CMIh estimator is sensitive to dimension, sample size, and neighbor fraction. Its primary performance, including a 15% disjoint-seed null false-positive rate, is not sufficient for real E2 selection.
- Normal intervals across the primary causal seeds are diagnostics, not definitive confidence intervals. Group-bootstrap or pre-registered repeated-simulation coverage is still required.
- The generic label shuffle tests no association; it cannot distinguish cortical information from target-correlated myogenic artifact.
- Waveform-level discriminating nulls are postponed because no time series are generated.
- The extended stress configuration remains opt-in and was not run. The current NumPy/SciPy/scikit-learn estimators have no CUDA path, so a GPU would not accelerate them without changing the registered methods.
- No real dataset was downloaded or analyzed, no remote was created, and no code/result was pushed externally.
