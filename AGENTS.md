# Regenesis CMI agent guide

## Layout

- `src/regenesis_cmi/information/`: exact, Gaussian, decoder, and published mixed-data estimators.
- `src/regenesis_cmi/synthetic/`: exact worlds, causal feature generator, independent oracle, degradation, and surrogates.
- `src/regenesis_cmi/evaluation/`: group splits, metrics, benchmark runner, and plots.
- `configs/`: smoke, primary, and opt-in extended experiment definitions.
- `tests/`: deterministic scientific and leakage checks.
- `docs/`: design, estimator, world, and limitation documentation.
- `results/`: generated compact summaries/figures only; raw synthetic arrays are not tracked.

## Commands

Setup: `python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'`

Tests: `.venv/bin/python -m pytest -q`

Smoke: `MPLCONFIGDIR=/tmp/regenesis-mpl .venv/bin/python -m regenesis_cmi.cli run --config configs/smoke.yaml`

Primary core: `MPLCONFIGDIR=/tmp/regenesis-mpl .venv/bin/python -m regenesis_cmi.cli run --config configs/primary.yaml --core`

## Scientific invariants

- `G` means categorical intended grasp type; never allow a single-class dataset.
- Report information in bits per independent trial and both CMI directions.
- Keep pre- and post-onset results separate.
- Never clip negative finite-sample estimates.
- Never split correlated trials/blocks across folds; fit preprocessing inside training folds.
- EMG degradation is `alpha*S_M + N_M`, never uniform scaling of observed `M`.
- Artifact controls must include band, montage, true/proxy myogenic conditioning, matched degradation, and a discriminating surrogate.
- Label shuffle is a generic association null, not evidence of cortical origin.
- Do not download real EEG/EMG or revive the rejected claim that EEG NMF automatically identifies neural synergies.

