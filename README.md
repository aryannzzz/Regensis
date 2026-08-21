# Regenesis conditional-information validation

This repository builds and tests the measuring instrument for the future Regenesis E2 experiment. It does **not** analyze human EEG/EMG and it does **not** test or confirm Regenesis H1.

The target is the categorical variable `G = intended grasp type`, with four balanced abstract classes by default. The primary quantity is

\[
I(G;E\mid M),
\]

the information about grasp supplied by EEG features after EMG features are already known. The reverse quantity, \(I(G;M\mid E)\), is always calculated separately.

## Entropy, information, and units

Entropy \(H(G)\) measures uncertainty in `G`. Mutual information \(I(G;E)\) measures how much knowing `E` reduces that uncertainty. Conditional mutual information asks how much reduction remains after another observation is known. A target with one class has \(H(G)=0\) and is rejected by configuration validation.

All reported values use base-2 units and are labelled **bits per independent trial**. Finite-sample estimates may be negative; those values are preserved because clipping would hide estimator failure.

## Why synthetic validation comes first

Real data do not reveal the true cortical, forearm-muscle, and cranial-muscle contributions. The framework therefore creates worlds where those components and the exact class-conditional law are known. It tests whether estimators can distinguish:

- complementary EEG information;
- redundant EEG information;
- estimator bias under a true null;
- an entirely myogenic EEG false positive;
- genuine cortical information mixed with artifact.

The causal simulator generates multichannel **features**, not raw waveforms. Raw filtering, spectral estimation, ICA, and phase-randomized waveform controls are deliberately postponed.

## Quick start

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

Validate analytic worlds:

```bash
.venv/bin/python -m regenesis_cmi.cli validate-exact
```

Regenerate the complete smoke result directory from one configuration:

```bash
MPLCONFIGDIR=/tmp/regenesis-mpl \
  .venv/bin/python -m regenesis_cmi.cli run --config configs/smoke.yaml
```

Run the completed primary matrix (100 exact seeds, five causal seeds, and 40
independent null-calibration seeds):

```bash
MPLCONFIGDIR=/tmp/regenesis-mpl \
  .venv/bin/python -m regenesis_cmi.cli run --config configs/primary.yaml
```

Regenerate figures from saved source tables:

```bash
MPLCONFIGDIR=/tmp/regenesis-mpl \
  .venv/bin/python -m regenesis_cmi.cli report --results results/smoke
```

Estimate extended runtime without starting it:

```bash
.venv/bin/python -m regenesis_cmi.cli estimate-runtime --config configs/extended.yaml
```

The extended configuration is opt-in and must not be run automatically if its estimate exceeds 30 CPU minutes.

## Outputs

Each result directory contains:

- the exact configuration and seeds;
- package, Python, platform, timestamp, runtime, Git state, and warnings;
- compact estimator/oracle/control tables under `tables/`;
- every figure as PNG and SVG, with its exact source CSV under `figures/`;
- deterministic generator parameters and model fingerprints;
- no real data and no large raw synthetic arrays.

The completed primary artifact is under `results/primary/`. It passed all ten
scientific checks with zero oracle-precision or estimator failures. The
continuous-estimator scorecard still does not justify confirmatory real-data
use: the nested decoder is the safer provisional diagnostic, while direct
CMIh showed a 15% disjoint-seed null false-positive rate in this matrix.

The principal audit files after a benchmark run are
`tables/scientific_checks.csv`, `tables/causal_core.csv`,
`tables/estimator_scorecard.csv`, `figures/myogenic_honesty.png`, and
`run_metadata.json`. `results/primary/remote_execution.json` records the
GiveMeANode execution receipt; `artifact_manifest.sha256` verifies each file.

## Documentation

- [Scientific design](docs/scientific_design.md)
- [Synthetic worlds](docs/synthetic_worlds.md)
- [Estimators](docs/estimators.md)
- [Limitations](docs/limitations.md)

The newer `Regenesis_Track_C_Operating_Manual.md` and the older `Project Regensis- From scratch.pdf` were inspected before implementation. The Track C manual governs. The older proposal's claim that NMF of scalp EEG band power automatically discovers fixed neural synergies is not implemented.
