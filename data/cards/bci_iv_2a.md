## url - https://www.bbci.de/competition/iv/#datasets

accessed on 07/08/26

# Dataset Card — BCI Competition IV Dataset 2b

## Source

- **Original name:** BCI Competition IV, Data sets 2b ("Graz data set B")
- **Also listed as:** BNCI Horizon 2020 dataset 004-2014, "Two class motor imagery"
- **Citation:** R. Leeb, F. Lee, C. Keinrath, R. Scherer, H. Bischof, G.
  Pfurtscheller. "Brain-computer communication: motivation, aim, and
  impact of exploring a virtual apartment." IEEE Transactions on Neural
  Systems and Rehabilitation Engineering 15, 473–482, 2007.
- **License:** Creative Commons Attribution No Derivatives (CC BY-ND 4.0)
- **Licensor:** Institute for Knowledge Discovery, Graz University of
  Technology

## Where we got it

- **Raw data (.gdf files):** downloaded from the BNCI Horizon 2020
  database, dataset 004-2014
  https://bnci-horizon-2020.eu/database/data-sets
  Access date: 2026-08-07

- **True labels for evaluation sessions (.mat files):** the raw .gdf
  files for evaluation sessions (04E, 05E) ship with class labels masked
  (event code 783, "Cue unknown") — this is intentional, left over from
  the original 2008 competition format where evaluation labels were
  withheld until after judging. The true labels were released
  separately after the competition and downloaded from:
  https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip
  Access date: 2026-08-07

  Both must be present in `data/raw/bci_iv_2b/` for the loader to work:
  `.gdf` files for all sessions, `.mat` files for the two `E` sessions
  per subject only.

## Structure

- **Subjects:** 9 (B01–B09)
- **Classes:** 2 — left-hand motor imagery, right-hand motor imagery
  (mapped to 0/1 respectively in our loader)
- **Chance level:** 50%
- **Sessions per subject:** 5 total
  - `01T`, `02T` — training, no feedback (screening)
  - `03T` — training, with smiley feedback
  - `04E`, `05E` — evaluation, with smiley feedback
- **Channels:** 6 total — 3 bipolar EEG (C3, Cz, C4), 3 monopolar EOG.
  EOG channels are provided for artifact-handling purposes and are
  excluded before classification.
- **Sampling rate:** 250 Hz
- **Raw band-pass:** 0.5–100 Hz, 50 Hz notch filter applied at recording
  time

## Trial counts (verified by running the loader, not assumed)

| Session | Left/right-hand trials |
|---|---|
| 01T | 120 |
| 02T | 120 |
| 03T | 160 |
| 04E | 160 |
| 05E | 160 |

Trial count is **not uniform across sessions** — 01T/02T (screening,
6 runs × 20 trials) have fewer trials than 03T/04E/05E (feedback
sessions, more runs). This matters for how "session-wise" splits are
interpreted downstream — sessions aren't equal-sized folds.

## Event codes used by our loader

| Code | Meaning | Used for |
|---|---|---|
| 768 | Start of a trial | Locating trial onsets in E sessions (where 769/770 are masked) |
| 769 | Cue onset, left hand (class 1) | Direct label source in T sessions |
| 770 | Cue onset, right hand (class 2) | Direct label source in T sessions |
| 783 | Cue unknown (masked) | Present in E sessions instead of 769/770 — ignored, real labels come from the paired .mat file |

## Known caveats

- Per the original dataset description, sessions `B0102T` and `B0504E`
  are missing their EOG-calibration block due to a recording issue
  at collection time. This doesn't affect the motor-imagery trials
  themselves, only the eye-artifact calibration segment, but is worth
  knowing if EOG-based artifact correction is ever added to this
  pipeline.
- Electrode placement (exact distance/orientation of the three bipolar
  derivations) varied slightly per subject — see the original paper for
  per-subject montage details if fine-grained channel geometry ever
  matters.

## Reproducibility

Dataset hash (from `compute_dataset_hash()` in `data/bci_iv_2b.py`,
covering all `.gdf` and `.mat` files together): recorded in
`results/PR-2026-01/data.sha256` once the pipeline is run end-to-end.