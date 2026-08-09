## url - https://www.bbci.de/competition/iv/#datasets

accessed on 07/08/26

# Dataset Card — BCI Competition IV Dataset 2a

## Source

- **Original name:** BCI Competition IV, Data sets 2a ("Graz data set A")
- **Also listed as:** BNCI Horizon 2020 dataset 001-2014, "Four class motor imagery"
- **Citation:** C. Brunner, R. Leeb, G. R. Müller-Putz, A. Schlögl, G. Pfurtscheller. "BCI Competition 2008 – Graz data set A." Institute for Knowledge Discovery, Graz University of Technology, 2008.
- **License:** Creative Commons Attribution No Derivatives (CC BY-ND 4.0)
- **Licensor:** Institute for Knowledge Discovery, Graz University of Technology

## Where we got it

- **Raw data (.gdf files):** downloaded from the BNCI Horizon 2020 database, dataset 001-2014
  https://bnci-horizon-2020.eu/database/data-sets
  Access date: 2026-08-07

- **True labels for evaluation sessions (.mat files):** the raw .gdf files for evaluation sessions (02E) ship with class labels masked (event code 783, "Cue unknown") — this is intentional, left over from the original 2008 competition format where evaluation labels were withheld until after judging. The true labels were released separately after the competition and downloaded from:
  https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip
  Access date: 2026-08-07

  Both must be present in `data/raw/bci_iv_2a/` for the loader to work: `.gdf` files for all sessions, `.mat` files for the `02E` session per subject only.

## Structure

- **Subjects:** 9 (A01–A09)
- **Classes:** 4 — left hand, right hand, both feet, tongue (mapped to 0/1/2/3 respectively in our loader)
- **Chance level:** 25%
- **Sessions per subject:** 2 total
  - `01T` — training session (6 runs × 48 trials)
  - `02E` — evaluation session (6 runs × 48 trials)
- **Channels:** 25 total — 22 monopolar EEG (Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz), 3 monopolar EOG. Left mastoid serving as reference, right mastoid as ground. EOG channels are provided for artifact-handling purposes and are excluded before classification.
- **Sampling rate:** 250 Hz
- **Raw band-pass:** 0.5–100 Hz, 50 Hz notch filter applied at recording time

## Trial counts (verified by running the loader, not assumed)

| Session | Left / right / foot / tongue trials | Total trials |
|---|---|---|
| 01T | 72 / 72 / 72 / 72 | 288 |
| 02E | 72 / 72 / 72 / 72 | 288 |

Trial counts are **uniform across sessions** — 6 runs per session with 48 trials per run (12 trials per class), yielding 288 trials per session and 576 total trials per subject.

## Event codes used by our loader

| Code | Meaning | Used for |
|---|---|---|
| 768 | Start of a trial | Locating trial onsets in E sessions (where 769–772 are masked) |
| 769 | Cue onset, left hand (class 1) | Direct label source in T sessions |
| 770 | Cue onset, right hand (class 2) | Direct label source in T sessions |
| 771 | Cue onset, foot (class 3) | Direct label source in T sessions |
| 772 | Cue onset, tongue (class 4) | Direct label source in T sessions |
| 783 | Cue unknown (masked) | Present in E sessions instead of 769–772 — ignored, real labels come from the paired .mat file |
| 1023 | Rejected trial / artifact | Marked by expert inspection for artifact filtering |

## Known caveats

- Subject `A04T` has a truncated EOG-calibration block prior to the main runs due to a technical issue at recording time (contains only the eye movement condition).
- Expert visual inspection marked artifact-contaminated trials (event code `1023`). Depending on the experiment design, loaders can either filter out these rejected trials or keep them.
- While the dataset natively contains 4 classes, many studies restrict experiments to a 2-class setup (e.g., left hand vs. right hand) when evaluating models directly against binary datasets like 2b.

## Reproducibility

Dataset hash (from `compute_dataset_hash()` in `data/bci_iv_2a.py`, covering all `.gdf` and `.mat` files together): recorded in `results/PR-2026-01/data.sha256` once the pipeline is run end-to-end.