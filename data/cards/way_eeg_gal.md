## url - TODO(PR-2026-02): fill in the exact resolved figshare URL used and access date before this card is considered complete (S6: "The card records the exact URL used and the access date -- never a remembered one.")

accessed on: NOT YET DOWNLOADED

# Dataset Card — WAY-EEG-GAL

**STATUS: NOT YET VERIFIED AGAINST A REAL FILE.** Everything below is
sourced from the published data descriptor and public secondary
sources, not from opening a real downloaded file (S6: "no dataset
enters this project on the strength of a description. Every corpus is
downloaded, opened, recounted and verified against its publication
before any modelling"). This is prereg/PR-2026-02.md open item 1 --
this card must be updated with confirmed structure (real MATLAB struct
field names, real trial counts recounted from the file, real channel
names) before any E3-A result is reported.

## Source

- **Original name:** WAY-EEG-GAL (Wearable interfaces for hAnd function
  recoverY — EEG Grasp-And-Lift)
- **Citation:** Luciw, M. D., Jarocka, E. & Edin, B. B. "Multi-channel
  EEG recordings during 3,936 grasp and lift trials with varying weight
  and friction." *Scientific Data* 1, 140047 (2014).
  https://doi.org/10.1038/sdata.2014.47
- **Corpus DOI:** 10.6084/m9.figshare.988376 (figshare collection, one
  record per participant)

## Where to get it

- Data descriptor: *Scientific Data*, https://doi.org/10.1038/sdata.2014.47
- Corpus: figshare collection at the DOI above, one dataset record per
  participant (P1-P12). TODO: resolve the exact per-participant URLs
  used and record the access date here once downloaded.

## Structure (per the data descriptor — not yet recounted from files)

- **Participants:** 12 (P1-P12)
- **Series per participant:** 10
- **Trials:** 328 per participant, 3,936 total. 6 weight series (34
  lifts/series), 2 surface series (34 lifts/series), 2 mixed series (28
  lifts/series) per participant.
- **Task:** cued reach, grasp (thumb + index finger), lift, hold,
  replace, release, return to rest. Object weight (165/330/660 g)
  and/or surface friction (sandpaper/suede/silk) varied unpredictably
  between trials. **Single grip type** — per the E3-A protocol doc,
  results scoped to this grasp only, not grasp type in general.
- **EEG:** 32 channels (10-20-style montage; exact channel names not
  yet confirmed against a real file — see TODO below).
- **EMG:** 5 channels (arm and hand muscles; exact muscle labels not
  yet confirmed).
- **Also recorded (not used by E3-A):** 3D hand/object position
  (Polhemus FASTRAK, 4 sensors), contact-plate force/torque.
- **File types per participant:**
  - `PX_AllLifts.mat` — per-lift windowed/holistic structure `P.AllLifts`
    (condition metadata, 16 event times, 18 behavioural measures per
    trial). **Not used by `data/way_eeg_gal.py`** — E3-A needs
    continuous data to epoch relative to EMG threshold-onset, not
    pre-cut lift windows.
  - `HS_PX_SY.mat` — continuous per-series recording (e.g.
    `HS_P3_S2.mat` = participant 3, series 2). **This is what
    `data/way_eeg_gal.load_series()` reads.**

## Preprocessing already applied by the dataset authors

- EEG and EMG recorded on separate acquisition systems (SC/ZOOM and
  BCI2000 respectively) and synchronized post-hoc via cross-correlation
  of two sync signals (max lag search: 5,000 samples). **This is a
  coarse, once-off, corpus-prep-time alignment — not a guarantee of
  zero residual offset for any given series.** This is exactly why
  `data/way_eeg_gal.measure_sync_gap()` exists and why the E3-A
  protocol doc requires the gap to be measured per series, not assumed.
- EMG: mean removed. No other EMG preprocessing.
- EEG: raw. **No artifact rejection applied by the dataset authors**
  (no blink/eye-movement removal) — this is why E3-A's own rejection
  step (`preprocessing/eeg_artifacts.py`) is necessary before use.

## Known issues / caveats

- Synchronization between EEG and EMG must be measured per series, not
  trusted (S6.1 known issue; see above).
- Event/onset labels should be inspected, not assumed correct —
  protocol doc: hand-verify EMG-derived onset on at least 30 trials.
- **Biases:** all able-bodied, young-adult laboratory participants,
  single laboratory, single protocol.
- **Limitations:** n = 12. Narrow grasp vocabulary — do not use this
  corpus for grasp-*type* diversity claims (S6.1 common mistake #1,
  already made once in this project's history per the manual).

## Common mistakes (per manual S6.1, apply here too)

1. Using this dataset for grasp-*type* diversity claims — it doesn't
   support them; only grasp *onset* anticipation is in scope for E3-A.
2. Trusting the corpus's own EEG/EMG synchronization instead of
   measuring it per series.
3. Epoching to cue rather than to EMG-marked onset — silently converts
   an anticipation result into a cue-response result.
4. Pooling pre- and post-onset windows in the same feature set.

## TODO before this card (and PR-2026-02) can be signed off

- [ ] Download at least one participant's `HS_*.mat` file; record the
      exact resolved URL and access date at the top of this card.
- [ ] Open the file in MATLAB or `scipy.io.loadmat` and confirm the
      real top-level struct name and field names (`eeg`/`EEG`,
      `emg`/`EMG`, sampling-rate field, channel-name field) against
      `data/way_eeg_gal.py`'s `EEG_DATA_FIELD_CANDIDATES` etc.; update
      the loader if any guess is wrong.
- [ ] Recount trials and series against the data descriptor (328/participant,
      3,936 total) directly from the downloaded files, not assumed.
- [ ] Confirm EEG and EMG sampling rates from the file (not yet stated
      here because not yet confirmed) and record them.
- [ ] Confirm the 32 EEG channel names and 5 EMG muscle labels from the
      file.