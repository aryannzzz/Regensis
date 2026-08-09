# PROJECT REGENESIS
# TRACK A OPERATING MANUAL
## EEG Signal Processing & Decoding

**Canonical identity:** Track A = Track 1 of the Phase II Handbook §13, "EEG Signal Processing & Decoding."

**Status:** Standalone working manual for the Track A team. Everything required to execute this track independently is contained here. You should not need to open the Handbook during ordinary work; if you find yourself needing to, that is a defect in this manual and should be reported.

**Authority:** Derived entirely from the Phase II Handbook and the Phase II Execution Guide. Where this manual and the Handbook appear to disagree, **the Handbook governs** and you raise the discrepancy the same day. No statement in this manual is new science.

**Team model:** Track A operates as one collaborative research team. There is no internal ownership split. Every result carries the track's name; every result must be reproducible by any member of the track and by at least one member outside it.

---

# 0. STANDING RULES — READ ONCE, APPLY ALWAYS

These are laboratory-wide and are inlined here so you never have to leave this document.

### 0.1 Non-negotiables

| Rule | Consequence of breaking it |
|---|---|
| Splits are **block-wise** — by trial block, by session, or by subject. **Never window-level.** | Inflated accuracy by tens of points; result withdrawn |
| Splitters, surrogates, estimators and metrics come from `eval/` only. **No local copies.** | Results across tracks become non-comparable |
| Preprocessing statistics are fitted **within fold**. | Leakage; result withdrawn |
| Hyperparameters are selected on a **nested inner fold**, never the evaluation fold. | Slow leak; result withdrawn |
| Every experiment ships with its negative control, **named in the pre-registration before data is touched**. | Pre-registration not signed |
| Every claim carries its scope: population, condition, offline/online. | Ledger entry rejected |
| Every citation is opened, graded A–F, entered in the Evidence Table before use. **Unopened citations are deleted.** | Claim flagged unsupported |
| Reading is predict-then-read: expected method, expected result, one expected limitation, written **before** opening the paper. | Journal Club slot forfeited |

### 0.2 Evidence grades

**A** peer-reviewed, adequate design, target-adjacent population · **B** peer-reviewed, limiting design (single subject, offline only, non-target population, small *n*) · **C** preprint · **D** non-peer-reviewed (vendor docs, industry reports, blogs) · **F** unverifiable or quote not located — **deleted from the record**.

A quote from a paper's **Introduction is secondhand** — the paper repeating someone else, usually with conditions stripped. Grade one level down and trace to origin or remove.

### 0.3 Definition of done

A task is complete when **all** of the following exist:

- `results/PR-YYYY-NN/` containing `figure.png`, `claim.md` (**one sentence, with scope, nothing else**), `methods.md`, `surrogates.md`, `env.lock`, `data.sha256`, `seed.txt`, `deviations.md`.
- Someone outside Track A regenerated it from the repository in under thirty minutes.
- The relevant `ledger/` entry is updated or a new one opened.

Anything else is in progress, however finished it feels.

### 0.4 Failure and refutation are different fields

A **Refutation Condition** is defined on the hypothesis: the observation showing the claim is false.
A **Failure Condition** is defined on the project: the observation showing this path is blocked.

Both appear in every pre-registration, separately. We are indifferent between hypothesis outcomes and very much not indifferent between project outcomes.

### 0.5 The three standing questions

Asked of anything, by anyone, of anyone:

> **What is your control? What is your split? What would this look like if you were wrong?**

### 0.6 Escalate the same day if

You cannot reproduce a published number · your accuracy exceeds expectation by a wide margin (**suspect leakage before celebrating**) · dataset metadata disagrees with its publication · two Regenesis documents contradict each other · you find yourself constructing an argument for a result you have already decided is true.

---

# 1. TRACK OVERVIEW

## 1.1 Why this track exists

Regenesis rests on a single anatomical claim: for a transradial amputee, the muscles that would encode digit intent are gone, but the cortical machinery that planned their movement is intact. If that claim is to do any work, the surviving cortical activity must be **readable through a skull, at a usable rate, and distinguishable from muscle.** Track A determines whether it is.

The track therefore sets the upper bound on the entire programme. No amount of downstream fusion, representation learning or control sophistication can recover information that never survived the scalp. When Track A reports a number, every other track's ambition is bounded by it.

## 1.2 Which hypotheses this track supports

| Hypothesis | Track A's contribution |
|---|---|
| **H1** — EEG carries grasp information not redundant with simultaneously recorded EMG, largest under EMG-degraded conditions | Supplies the EEG features **and**, decisively, the controls that determine whether any positive result is cortical or myogenic |
| **H2** — a decodable fraction of EEG grasp/intent information is available before EMG onset at usable lead times and false-positive rates | Builds and sweeps the anticipatory detector |

Track A does **not** own H1 or H2. Track C (Fusion) owns H1 through E2 and the forward-feed half of H2. Track A owns the EEG-side inputs to both, and owns the artifact analysis outright.

## 1.3 The scientific question

> **What does scalp EEG actually deliver for grasp intent — at what class cardinality, with which features, under artifact-controlled and leakage-safe evaluation — and how much of any apparent signal is cortical rather than myogenic?**

## 1.4 Why the result matters

Three reasons, in ascending order of consequence.

**It selects the feature family.** The project originally specified mu/beta band power. The current architecture specifies low-frequency MRCP (0.3–3 Hz), because that is what the reach-and-grasp literature actually uses. Track A quantifies the difference on our data rather than inheriting the assertion.

**It bounds the architecture.** EEG is a discrete, low-rate source in the current architecture — never a continuous finger controller. That decision was made on an information ceiling of roughly r ≈ 0.3–0.5 for continuous kinematic decoding. Track A produces our own measurement of what class cardinality is reliably supported, which is what a grasp vocabulary can be built on.

**It decides whether Gate B is real.** This is the one that matters. Myogenic contamination of scalp EEG — cranial, facial, jaw and neck musculature — reproduces H1's *entire* confirmatory signature with no cortical contribution whatsoever. Track A owns the three controls that separate the two. If Gate B passes on artifact, every subsequent tier, every subsequent paper and the flagship are built on it.

## 1.5 Why the rest of the laboratory depends on this work

| Who | What they need from Track A | When |
|---|---|---|
| **Track C (Fusion)** | Band-resolved EEG features; montage-ablated variants; the myogenic component estimator; a jointly signed E2 pre-registration | Week 8 |
| **Track C (Fusion)** | The E3 detector output, for forward-feed and the latency-reduction measurement | Week 11 |
| **Track C (Fusion)** | An EEG encoder for E4's paired arm | Week 13+ |
| **Track E (Evidence)** | Our reproduced values and their deltas, for the Number Board | Week 4 |
| **Whole laboratory** | The honest statement of what EEG delivers, with its scope clause, entered in the belief ledger | Week 12 |

Track A is not on the critical path in Weeks 1–3 — Track E is — but from Week 8 onward **Track C cannot run E2 without Track A's controls.** Slipping the artifact work slips Gate B, and Gate B gates Tiers 2 and 3.

---

# 2. SCIENTIFIC BACKGROUND

Only what this team must understand completely before starting. Everything here is examinable without warning.

## 2.1 Where EEG sits in the project's ontology

Regenesis holds three levels rigorously separate:

- **Level 1 — motor intent `x(t)`.** The time-evolving, modality-independent internal state of the user's motor system. **Unobservable in principle.**
- **Level 2 — physiological observations `y_m(t)` = `g_m(x(t))` + noise.** What instruments measure. EEG is `g_EEG`; EMG is `g_EMG`. **No single projection is invertible.**
- **Level 3 — learned representations `z(t)`.** Our computational approximations of Level 1, estimated from Level 2. Always partial. **Never to be mistaken for `x(t)`.**

Two consequences you will use daily:

**Amputation is the loss of an observation channel, not of the intent.** A transradial amputation degrades or removes `g_EMG`; `x(t)` is untouched. This is the entire principled reason EEG is in the project — not "EEG is earlier," but "EEG is the remaining channel for a specific class of information."

**Temporal precedence is structural, not anecdotal.** `g_EEG` taps the state upstream of `g_EMG` in the projection chain, so EEG *can* carry `x(t)` before EMG does. That is the correct statement of the anticipation argument, and it is what E3 tests. The older readiness-potential framing — cortical potentials precede movement by 1–2 s, therefore we gain a head start — is retired, because the raw fact of precedence says nothing about whether the lead survives processing latency.

## 2.2 The physiology you must be able to explain on demand

**Event-related desynchronisation (ERD) and synchronisation (ERS).** Movement and motor imagery produce a decrease in mu (8–12 Hz) and beta (13–30 Hz) band power over sensorimotor cortex, followed by a post-movement rebound. This is the phenomenon the entire EEG path rests on. You must be able to state its approximate time course, its spatial distribution, and its inter-subject variability.

**Movement-related cortical potentials (MRCP).** Low-frequency (≈0.3–3 Hz) negative-going potentials preceding and accompanying voluntary movement. **This is our primary feature family**, adopted over band power because it is what the reach-and-grasp decoding literature uses. Band power is retained as a comparison arm.

**Volume conduction.** Cortical current sources are spatially blurred by the skull and scalp before reaching the electrode. Consequences: adjacent cortical representations mix; finger representations, which are adjacent and overlapping, are especially hard to separate; and **no downstream algorithm can undo this** — it is destruction, not merely mixing, at the resolution we care about.

**What a transradial amputation removes.** The forearm distal to the amputation level, including the intrinsic hand muscles and the distal portions of the extrinsic finger flexors and extensors. What remains: proximal forearm musculature, the elbow, and the entire central nervous system. This is why EMG can carry wrist and forearm intent and cannot carry digit intent.

**BCI illiteracy.** Roughly 15–30% of users do not produce classifiable motor-imagery patterns with usable reliability. This is a hard constraint, replicated across independent lines. It means system performance is partly a property of the user, not of the algorithm — and it is why every EEG-dependent path in the architecture has an EMG-only fallback.

## 2.3 The myogenic contamination mechanism — the most important thing in this section

You must be able to derive this from memory. It is the reason Track A's controls exist.

1. Scalp EEG contains myogenic activity from cranial, facial, jaw and neck musculature. This activity is **correlated with movement execution and with effort.**
2. That contamination therefore carries grasp-correlated information — not because cortex encoded it, but because **the body moved**.
3. Conditional mutual information `I(grasp ; EEG | EMG) > 0` follows immediately: the contamination is not redundant with *forearm* EMG, because it originates in entirely different muscles.
4. Now degrade the forearm EMG channel — attenuate it, add noise, simulate fatigue or dropout. **The contamination in the EEG channel is untouched.** Forearm-EMG information falls; EEG-side information holds constant; the fusion gain rises.
5. The gain-versus-degradation curve slopes upward **exactly as H1 predicts**.

**A label-shuffle permutation null does not catch this.** Shuffling grasp labels destroys the artifact's information along with the neural information, so the artifact-driven result passes the null comfortably. A null that both hypotheses pass is not a test.

Three properties of the contamination give us leverage, and they are the basis of our three controls:

- **Frequency.** Myogenic activity is broadband and dominant above roughly 20 Hz. MRCP features at 0.3–3 Hz are comparatively protected; mu/beta features are not.
- **Space.** Contamination is largest at peripheral, temporal and frontal sites. Genuine grasp information should be lateralised over contralateral sensorimotor cortex.
- **Separability.** A high-frequency myogenic component can be estimated from the EEG montage itself and conditioned on.

## 2.4 Evaluation concepts you must hold

**Leakage.** Any path by which information from the evaluation set influences training. In neural time-series the dominant forms are: overlapping windows from one trial placed in both train and test; trial-adjacent splitting; preprocessing statistics fitted across the full dataset; hyperparameter selection on the evaluation fold; and subject-pooled cross-validation reported as cross-subject. **These compound.** In this data family, evaluation policy is frequently a larger effect than method choice.

**Surrogate.** A transformation of the data that destroys the effect of interest while preserving as much else as possible. Ours: label shuffle, phase randomisation, spectrally matched noise, time reversal, channel permutation. Provided by `eval/surrogates/`, validated by Track E in **both** directions — a planted real effect must survive, and a planted artifact must die. A suite validated only against under-correction is not validated.

**Scope clause.** Every result carries its population, condition and offline/online status as part of the claim. Not *"EEG decodes grasp at 68%"* but *"EEG decodes 4-class grasp at 68% (per-subject median, 95% CI ...) in 12 able-bodied subjects, offline, session-wise splits, MRCP features, central-only montage."*

## 2.5 What is settled and what is open

**Settled enough to build on** — but every value is being re-verified by Track E's Numbers Audit before any threshold is set against it:
- EEG grasp decoding is discrete and modest: binary grasp-versus-grasp around the low 70s %, four-class peaking in the mid 60s %.
- Those results use low-frequency MRCP features, not mu/beta band power.
- Continuous kinematic/force decoding from scalp EEG has an honestly validated ceiling around r ≈ 0.3–0.5; several published values above 0.7 are likely inflated by low-frequency motion, EMG or mechanical artifact.
- BCI illiteracy affects 15–30% of users.

**Open, and what this track exists to address:**
- The magnitude and conditionality of unique EEG information beyond EMG (Track C owns the measurement; we own its validity).
- The real anticipatory operating curve for grasp *type*, as opposed to grasp *onset* — onset is comparatively well studied, type is not.
- How much of any apparent EEG contribution is myogenic.

---

# 3. SCOPE

## 3.1 Track A IS responsible for

- All EEG preprocessing: filtering, surface Laplacian, ICA and artifact handling, epoching.
- MRCP and mu/beta band-power feature extraction.
- Three decoder families — linear/sLDA, Riemannian tangent-space, EEGNet — implemented so that **method effect can be separated from signal effect**.
- The reproduction of a published motor-imagery decoding number under strict splits, with the delta reported.
- The grasp/onset decoding ladder with its full surrogate panel.
- **The three artifact controls of the E2 protocol: band-resolved features, montage ablation, and the myogenic component estimator.**
- The E3 anticipatory detector and its lead × TPR × FP/min sweep.
- Dataset cards for every EEG corpus used.
- An EEG encoder for E4's paired arm (Weeks 13+).
- Ledger maintenance for the EEG-related beliefs.

## 3.2 Track A is NOT responsible for

| Not ours | Whose | Why the boundary is here |
|---|---|---|
| Computing conditional mutual information | Track C | Track C owns H1 and the estimator; we supply the features and controls it conditions on |
| The fusion ladder and its tuning budgets | Track C | Single owner prevents divergent evaluation |
| Deciding Gate B | Track C | We supply the evidence that makes the decision valid |
| The forward-feed latency measurement in E3 | Track C | We build the detector; Track C measures what it buys downstream |
| Any EMG processing or synergy extraction | Track B | Including the EMG side of WAY-EEG-GAL |
| Splitters, surrogates, metrics, statistical utilities | Track E | Single implementation, laboratory-wide |
| Verifying published literature values for the Number Board | Track E | We supply our reproduced values; Track E audits the published ones |
| Any closed-loop or hardware work | Track D / Phase 2 | Behind Gate E |
| The band-power-NMF generative-justification simulation | **Explicitly deferred** | Open question Q7 in the Handbook; off the critical path; the mechanism it interrogates is already removed from the architecture. **Do not start it in Phase II unless the PI reassigns it in writing.** |

## 3.3 Boundary conditions

- **No EEG result of ours is ever presented as evidence about fusion.** We report what the EEG channel carries. Whether it carries anything *beyond* EMG is Track C's measurement.
- **No benchmark motor-imagery number is ever cited as bearing on H1.** BCI-IV-2a establishes that our pipeline works. It says nothing about grasp intent in this project.
- **Epoching is relative to EMG threshold-onset, not to trial start or cue.** Cue-aligned epoching silently converts an anticipation result into a cue-response result.
- **We do not modify Track B's EMG outputs**, and Track B does not modify ours.
- Where the data does not support a claim about grasp-type diversity, we do not make one. WAY-EEG-GAL has a narrow grasp vocabulary; this constraint is stated in every figure caption, not absorbed.

## 3.4 Deliverables expected from this track

Summarised here; specified in full in §12.

**Research:** the statement of what EEG delivers, per subject, with CIs and scope clauses; the MRCP-versus-band-power comparison quantified; the artifact-versus-cortical separation; the E3 operating curve.
**Software:** the MNE pipeline; feature extractors; three decoder families; band-resolved routing and montage-ablation utilities; the myogenic estimator.
**Documentation:** dataset cards; Number Board entries for our own values; pre-registrations; ledger updates.
**Presentation:** weekly Figure Clinic figure; Lab Meeting deep presentations on rotation; three Journal Club papers with reproduction attempts.
**Decision input:** the evidence packet on which Gate B's validity rests, and the E3 curve for Gate C.

---

# 4. RESEARCH QUESTIONS

Every question Track A is expected to answer. Confidence levels are the laboratory's current position and are ours to move.

### RQ-A1 — What is the reliable class cardinality for grasp decoding from scalp EEG in our data?

**Why it matters.** It bounds the grasp vocabulary the architecture can support. The current architecture treats EEG as a discrete low-rate selector; the value of *k* determines whether that selector is useful or trivial.
**Current belief.** Discrete and modest — binary around the low 70s %, four-class peaking in the mid 60s %.
**Confidence.** Medium, pending the Numbers Audit.
**Experiment.** T1-R (reproduction), then the decoding ladder within E2's EEG arm.
**Success criteria.** Per-subject accuracy at *k* = 2, 3, 4 reported with bootstrap CIs, on session-wise splits, surviving the full surrogate panel, with a stated scope clause.
**Failure criteria.** Cannot exceed the surrogate null at any *k* → report it. This is a Tier-1-relevant negative and it is publishable.

### RQ-A2 — Do MRCP features outperform mu/beta band power for grasp intent on our data?

**Why it matters.** The architecture already specifies MRCP over band power, on the strength of the published reach-and-grasp literature rather than our own measurement. This question converts an inherited assertion into a result.
**Current belief.** MRCP is the working feature family for grasp decoding; band power is not.
**Confidence.** Medium-High, pending the Numbers Audit.
**Experiment.** The MRCP-versus-band-power arm of the decoding ladder, with equal tuning budgets.
**Success criteria.** A quantified per-subject comparison with a paired statistical test and CIs.
**Failure criteria.** None — either outcome is informative. If band power wins, that is a finding and the ledger moves.

### RQ-A3 — How much of any apparent EEG grasp information is myogenic rather than cortical?

**Why it matters.** **This is the most consequential question in the track.** It determines whether Gate B's decision is valid, and therefore whether Tiers 2 and 3 are built on something real.
**Current belief.** The mechanism by which contamination reproduces H1's signature is established deductively; its magnitude in our data is unknown.
**Confidence.** High on the mechanism, Unknown on the magnitude.
**Experiment.** The three controls: band-resolved decoding, montage ablation, myogenic nuisance conditioning.
**Success criteria.** All three figures produced; the before/after myogenic-conditioning difference quantified and delivered to Track C.
**Failure criteria.** If the effect is concentrated above 20 Hz, collapses under a central-only montage, or largely disappears after myogenic conditioning — **that is the finding, it is reported as the headline, and Gate B is decided accordingly.** This is not a failure of the track.

### RQ-A4 — What is the anticipatory operating curve for grasp onset, and where the data permits, grasp type?

**Why it matters.** Anticipation is the one thing better EMG can never replace, because before onset there is no EMG signal to be redundant with. Grasp-*type* anticipation is substantially under-studied relative to onset, and the achievable lead-versus-false-positive trade-off is the quantity a designer actually needs.
**Current belief.** Usable lead at acceptable false-positive rates is plausible for onset; type anticipation is harder.
**Confidence.** Medium for onset, Low for type.
**Experiment.** E3 detector sweep.
**Success criteria.** The full lead × TPR × FP/min curve with per-subject spread — **not a single operating point**.
**Failure criteria.** Detection achievable only at unusable false-positive rates → report the curve as the finding. The curve is publishable whether or not the operating point is usable.

### RQ-A5 — Does subject variability exceed method variability in our EEG results?

**Why it matters.** It determines whether effort should go into better decoders or into screening and per-subject calibration, and it bears directly on the BCI-illiteracy constraint.
**Current belief.** BCI illiteracy at 15–30% is a hard constraint; between-subject variance is expected to be large.
**Confidence.** High that illiteracy is real; unmeasured in our data.
**Experiment.** Reported as a by-product of the decoding ladder — per-subject distributions across all three decoder families.
**Success criteria.** Per-subject × per-method table with variance attributable to each, reported alongside the ladder.
**Failure criteria.** None. This is descriptive and always informative.

---

# 5. EXPERIMENTS

Three experiments. No ambiguity is left; where a choice exists it is made here and any deviation is logged in `deviations.md`.

---

## 5.1 T1-R — Reproduction of a published motor-imagery decoding result

**Objective.** Establish that our pipeline works and that we can hit a published number under honest splits.

**Scientific motivation.** Before this track's own numbers mean anything, we must show we can reproduce someone else's under our evaluation policy. The delta between our value and the published one is itself informative: it calibrates how to read every EEG number we will later compare ourselves against, and it feeds Track E's Numbers Audit.

**Inputs.** BCI-IV-2a. One published two-class motor-imagery accuracy with its conditions extracted first: *n*, subjects, class definitions, chance level, split policy, feature family, classifier.

**Outputs.** Reproduced accuracy per subject; the delta from published; a written hypothesis for the discrepancy.

**Pipeline.**
1. Load through `data/` with a version hash recorded.
2. Band-pass 8–30 Hz. Epoch per the source paper's timing, recorded explicitly.
3. CSP or Riemannian tangent-space features, matching the source paper's family.
4. sLDA classifier.
5. **Block-wise split from `eval/splits/`** — session-wise where the dataset provides sessions, else trial-block.
6. Label-shuffle surrogate.
7. Per-subject accuracy, bootstrap CIs.

**Expected intermediate results.** After step 3, feature dimensionality and class separability should be visually inspectable — plot the first two CSP or tangent-space components per class. If classes are not visibly separated for the best subjects, stop and debug before proceeding to step 4.

**Controls.** Block-wise split; label-shuffle surrogate. Both mandatory.

**Failure modes.** Cannot locate the source paper's conditions → escalate, and grade the source down. Reproduce far above the published value → **suspect leakage before celebrating**; check the split first. Reproduce far below → check epoch timing and band definition before concluding anything.

**Evaluation metrics.** Per-subject accuracy; the delta from published; the surrogate null distribution.

**Statistical tests.** Bootstrap percentile CIs on per-subject accuracy (≥1000 resamples). Permutation test against the label-shuffle null. No paired test is required here; this is a single-arm reproduction.

**Expected figures.** *Figure T1-R.1:* per-subject reproduced-versus-published scatter with the identity line and CIs.

**Interpretation.** Points near the identity line: pipeline validated. Points systematically below: our evaluation is stricter, and the size of the gap is the calibration we needed. Points above: investigate leakage before anything else.

**Decision gate.** None directly. Blocks nothing formally, but no Track A modelling result is reported before T1-R closes.

**Publication relevance.** Feeds Track E's Numbers Audit and Publication 1. Not a standalone contribution.

**Estimated duration.** 2 weeks.

**Complete when.** Result directory complete per §0.3; delta reported with a hypothesis; Number Board entry created.

---

## 5.2 E2-A — Track A's contribution to the Tier 1 information decomposition

**Objective.** Supply Track C with EEG features and — decisively — with the evidence that separates cortical from myogenic contribution.

**Scientific motivation.** E2 tests H1. Its confirmatory signature is `I(grasp ; EEG | EMG) > 0` together with fusion gain growing under EMG degradation. That signature is reproducible in full by myogenic contamination with no cortical contribution (§2.3). Track A's controls are what make Gate B's decision mean anything. **This is the track's principal scientific responsibility.**

**Inputs.** WAY-EEG-GAL. Full and central-only montages. MRCP (0.3–3 Hz) and mu/beta (8–30 Hz) feature sets.

**Outputs, delivered to Track C.**
1. Band-resolved feature sets, so conditional MI can be reported per frequency band.
2. Montage-ablated variants: full montage versus central-only, with peripheral, temporal and frontal channels removed.
3. A high-frequency myogenic component estimated from the EEG montage itself, for use as a nuisance covariate.
4. A jointly signed E2 pre-registration.

**Pipeline.**
1. Load WAY-EEG-GAL; version hash recorded; **synchronisation measured, not assumed**.
2. Epoch relative to **EMG threshold-onset**, keeping pre-onset and post-onset windows separate throughout. Hand-verify onset labels on ≥30 trials before proceeding.
3. Two parallel feature paths: MRCP band-pass 0.3–3 Hz; band power 8–30 Hz.
4. Surface Laplacian. ICA for ocular artifact — **validated against a planted effect before adoption** (see failure modes).
5. Band-resolved routing: features computed and retained per band so per-band conditional MI is possible downstream.
6. Montage variants generated: full, central-only.
7. Myogenic component estimated from high-frequency content across the montage.
8. Decoding diagnostics per band and per montage, as our own read on what the controls will show Track C.

**Expected intermediate results.** After step 5, per-band decoding accuracy should be inspectable. Expect: MRCP band carries grasp information; if the >20 Hz bands carry *comparable or greater* information, that is the first sign the effect is myogenic and it should be reported at the next Figure Clinic, not at the end of the experiment.

**Controls.** Ours are three of the five in the E2 protocol:
- **Band-resolved decoding** — report per frequency band, always, never pooled.
- **Spatial specificity** — recompute with peripheral, temporal and frontal channels ablated.
- **Myogenic nuisance regressor** — supply the component so Track C can report `I(grasp ; EEG | EMG, myo)` alongside `I(grasp ; EEG | EMG)`. **The difference between those two numbers is the honesty figure of the entire project.**

The remaining two controls — a degradation manipulation that also perturbs the artifact, and a discriminating null — are Track C's, and are specified in the joint pre-registration.

**Failure modes.**
- *ICA over-correction removing movement-correlated neural activity along with ocular artifact.* Detect: decoding drops sharply after ICA on central channels. Recover: validate the ICA step against a planted effect exactly as Track E validates surrogates; if it kills the planted effect, fix or drop it.
- *Onset labels misaligned.* Detect: anticipation appears at implausibly long leads. Recover: hand-verify 30 trials; this is why step 2 exists.
- *Inheriting the synchronisation assumption.* Detect: nobody in the track can say what the measured synchronisation error is. Recover: measure it.
- *Reporting a pooled CMI-relevant number.* Detect: any figure with one accuracy value and no band or montage label. Recover: re-run separated.

**Evaluation metrics.** Per-band decoding accuracy; full-versus-central-only accuracy; before/after myogenic-conditioning decoding accuracy. These are our diagnostics; Track C converts them into conditional MI.

**Statistical tests.** Paired Wilcoxon signed-rank across subjects for band comparisons and for full-versus-central-only. Bootstrap percentile CIs on every reported value. Permutation nulls from `eval/surrogates/`. Holm correction across the band family, stated in the pre-registration.

**Expected figures.**
- *Figure E2-A.1:* decoding accuracy by frequency band, per subject, with the null band.
- *Figure E2-A.2:* full montage versus central-only, paired per subject.
- *Figure E2-A.3:* before versus after myogenic conditioning, paired per subject.

**Interpretation.** Effect concentrated in 0.3–3 Hz, surviving central-only montage, largely surviving myogenic conditioning → **cortical, and Gate B may proceed on it.** Effect concentrated above 20 Hz, or collapsing under central-only montage, or largely disappearing after conditioning → **myogenic, and this is the finding.** Report it as the headline; Gate B is decided accordingly; the track has done its job.

**Decision gate.** **Gate B.** Track C decides; Track A supplies the evidence that makes the decision valid.

**Publication relevance.** Publication 2. Figure E2-A.3 is the honesty figure of the project and belongs in the main text regardless of which way it goes.

**Estimated duration.** 4 weeks.

**Complete when.** All three figures exist; Track C has integrated the controls; the joint pre-registration is signed.

---

## 5.3 E3-A — The anticipatory detector

**Objective.** Establish the real anticipatory operating curve for grasp onset and, where the data permits, grasp type.

**Scientific motivation.** H2 holds that a decodable fraction of EEG grasp information is available before EMG onset. This is EEG's structural advantage: `g_EEG` taps the state upstream of `g_EMG`, so before onset there is no EMG signal to be redundant with. The quantity a designer needs is not a single accuracy but the trade-off surface between lead time, true-positive rate and false positives per minute — and for grasp *type* that surface has not been published cleanly.

**Critically: E3 runs regardless of Gate B's outcome.** A Tier-1 null measured in post-onset windows says nothing about anticipation, because in the anticipatory regime there is no EMG signal to be redundant with. Gate B cannot terminate this experiment.

**Inputs.** WAY-EEG-GAL, epoched to EMG threshold-onset. MRCP features.

**Outputs.** Detection latency relative to onset; TPR; FP/min — **swept across operating points, not reported at one**.

**Pipeline.**
1. Onset labels from EMG threshold-crossing, hand-verified on ≥30 trials.
2. MRCP features in sliding windows preceding onset.
3. sLDA and EEGNet detectors, equal tuning budgets, nested inner-fold selection.
4. Threshold sweep across the full decision-value range.
5. At each threshold: mean detection lead, TPR, FP/min computed over rest periods.
6. Per-subject curves; bootstrap CIs.

**Expected intermediate results.** After step 3, detector decision values should show a visible pre-onset rise on averaged trials for the best subjects. If no subject shows this, stop and check onset alignment before sweeping.

**Controls.**
- **Time-reversed surrogate** — reverse the epoch; genuine anticipation should not survive.
- **Shuffled-onset null** — randomise onset times within the recording; detection should fall to the false-positive floor.
- **Pre-baseline-shift check** — confirm the detector is not keying on a baseline drift artifact by re-running with baseline correction applied over a window that excludes the pre-onset period.

**Failure modes.**
- *Detecting the cue rather than the movement.* Detect: leads cluster suspiciously near the cue-to-onset interval. Recover: epoch to onset, verify cue timing is not in the feature window.
- *Baseline drift masquerading as MRCP.* Detect: effect survives time reversal. Recover: the pre-baseline-shift check.
- *Reporting a single operating point.* Detect: one number in the figure. Recover: sweep.

**Evaluation metrics.** Lead time (ms before EMG onset); TPR; FP/min; per-subject spread.

**Statistical tests.** Permutation test against the shuffled-onset null at each operating point. Bootstrap CIs on the curve. Paired Wilcoxon across subjects for detector-family comparison. Holm correction across operating points, stated in the pre-registration.

**Expected figures.**
- *Figure E3-A.1:* lead time versus TPR at fixed FP/min, per-subject curves plus the group median.
- *Figure E3-A.2:* the same, for grasp type rather than onset, **where the grasp vocabulary supports it** — and a stated limitation where it does not.
- *Figure E3-A.3:* surrogate panel — the curve under time reversal and shuffled onset.

**Interpretation.** Usable lead at acceptable FP/min → anticipation becomes a co-primary contribution and Track C measures what it buys downstream. Detection only at unusable FP/min → **report the curve as the finding.** It is publishable either way, because the trade-off surface for grasp type has not been published cleanly and a designer needs it regardless of whether it is favourable.

**Decision gate.** **Gate C.** Requires both our curve and Track C's forward-feed latency measurement.

**Publication relevance.** Publication 3, in either direction.

**Estimated duration.** 3 weeks.

**Complete when.** Curve produced with per-subject spread and CIs; surrogate panel passing; handed to Track C for forward-feed.

---

# 6. DATASETS

**Standing rule: no dataset enters this project on the strength of a description.** Every corpus is downloaded, opened, recounted and verified against its publication before any modelling. Output is a dataset card in `data/`, co-reviewed with Track E. The card records the **exact URL used and the access date** — never a remembered one.

---

## 6.1 WAY-EEG-GAL

**Purpose.** The only public corpus with synchronized EEG + EMG + kinematics + force and labelled behavioural events including movement onset. **Without it there is no E2 and no E3.**

**Download source.** Published as a data descriptor in *Scientific Data* (Luciw, Jarocka & Edin) with the corpus hosted on figshare. Locate via the data descriptor's own accession link; record the resolved URL, version and access date in the dataset card. Do not use a mirror without recording that it is one.

**Expected preprocessing.**
- Band-pass 0.3–3 Hz for the MRCP path; 8–30 Hz for the band-power comparison path.
- Surface Laplacian.
- ICA for ocular artifact — **validated against a planted effect before adoption**.
- Epoch relative to **EMG threshold-onset**. Pre-onset and post-onset windows kept separate throughout, never pooled.
- Per-channel normalisation fitted **within fold**.

**Splits.** Session-wise where the recording structure permits; otherwise trial-block. **Never window-level.** Splits come from `eval/splits/` and are pre-registered before any modelling result is reported.

**Known issues.** Synchronisation between modalities must be *measured*; multimodal corpora have documented synchronisation error and it is a known failure mode. Event labels must be inspected rather than trusted.

**Biases.** All able-bodied. All young-adult laboratory participants. Single laboratory, single protocol.

**Limitations.** n = 12. Narrow grasp vocabulary. Degradation conditions available to us are *simulated*, not physiological.

**Common mistakes.**
1. **Using it for grasp-*type* diversity claims.** It does not support them. This error has already been made once in this project's history and was caught by cross-document review.
2. **Trusting synchronisation instead of measuring it.**
3. **Epoching to cue rather than onset**, which silently converts an anticipation result into a cue-response result.
4. Pooling pre- and post-onset windows, which conflates a bandwidth claim with a temporal one.

**Recommended sanity checks.**
- Recount trials and subjects against the data descriptor.
- Verify channel count, sampling rate and channel names against the descriptor.
- Measure EEG–EMG synchronisation error and record the number.
- Hand-verify EMG threshold-onset on ≥30 randomly chosen trials.
- Confirm event-code semantics against the descriptor's own table.

**Required visualisations before modelling.**
- Raw traces for 10 random epochs, EEG and EMG on a common time axis, with the detected onset marked.
- Per-channel power spectral density, averaged and per subject, to identify bad channels and line noise.
- ERD/ERS time-frequency map over sensorimotor channels, onset-aligned — **you should be able to see the phenomenon before you try to decode it.**
- Grand-average MRCP, onset-aligned, per subject.
- Artifact-rejection rate per subject.

---

## 6.2 BCI-IV-2a (and 2b)

**Purpose.** Standard motor-imagery benchmark for decoder sanity-checking and for the T1-R reproduction. **Not an experimental dataset for Regenesis claims.**

**Download source.** BNCI Horizon 2020 dataset repository / the BCI Competition IV archive. Record the resolved URL and access date.

**Expected preprocessing.** Standard MI pipeline: band-pass 8–30 Hz, epoch per the source paper's timing (recorded explicitly), CSP or Riemannian tangent-space features.

**Splits.** Session-wise — the dataset provides distinct sessions and this is exactly what makes it useful for demonstrating our splitting policy.

**Known issues.** Class definitions and epoch timing vary between published analyses of the same data; this is the main source of reproduction deltas.

**Biases.** MI paradigm, cued, laboratory setting. Different task structure from grasping.

**Limitations.** It is motor imagery, not grasp. Nothing about it bears on H1.

**Common mistakes.**
1. **Reporting a benchmark number as if it bore on H1.** It establishes that the pipeline works. Nothing more.
2. Comparing our value to a published one without extracting the published epoch timing and band definition first.

**Recommended sanity checks.** Verify subject count, session structure, class labels and trial counts against the competition documentation.

**Required visualisations before modelling.**
- Per-class ERD time-frequency maps over C3/C4 — the effect should be visible.
- First two CSP or tangent-space components, coloured by class.
- Per-subject class balance.

---

## 6.3 EEGMMIDB

**Purpose.** Secondary MI benchmark, available if T1-R's primary target proves unsuitable. Optional.

**Download source.** PhysioNet. Record the resolved URL and access date.

**Expected preprocessing, splits, checks.** As BCI-IV-2a.

**Known issues.** Documented annotation irregularities for a subset of subjects — search the errata before use and record what you find in the dataset card.

**Common mistakes.** Using it without checking the errata.

---

# 7. LITERATURE

**Standing rule.** Predict-then-read: write expected method, expected result and one expected limitation **before** opening the paper. Grade A–F on first read (§0.2), extract *n*, population, task, class count, chance level and offline/online, and enter in the Evidence Table. **A paper is presented at Journal Club only after one of its numbers has been reproduced or a documented attempt has failed.**

Volume is not the objective. Roughly six papers read to the depth of being able to redraw their figures from memory beats forty skimmed.

---

## 7.1 Must read

**Pfurtscheller & Lopes da Silva — event-related desynchronisation and synchronisation.**
*Why we read it:* the phenomenon the entire EEG path rests on. *Insight it contributes:* what ERD is, its time course, its spatial distribution, and how much it varies across people. *Experiment that depends on it:* all of them — you cannot interpret a band-power result without it. *Assumptions:* supports the premise that sensorimotor rhythms carry movement-related information; its variability findings challenge any expectation of a subject-general decoder.

**Muthukumaraswamy / Whitham — muscle artifact contamination of scalp EEG.**
*Why we read it:* **the single most important paper for this track.** It is the mechanism behind our most probable false positive. *Insight it contributes:* where myogenic activity dominates in frequency (broadly, above ~20 Hz) and in space (peripheral, temporal, frontal sites). *Experiment that depends on it:* E2-A's three controls are constructed directly from these two facts. *Assumptions:* challenges any uncontrolled positive result for H1; supports the claim that MRCP features are comparatively protected.

**Niazi et al. — MRCP-based movement detection.**
*Why we read it:* the feature family we adopted over band power. *Insight it contributes:* how MRCP detection is done and at what operating point — the reported true-positive rate and lead. *Experiment that depends on it:* E3-A's detector design. *Assumptions:* supports the MRCP-over-band-power decision recorded in the project's decision log.

**Lew et al. — anticipatory movement-intent detection.**
*Why we read it:* the evidence base for H2 and for the lead times we consider plausible. *Insight it contributes:* achievable pre-onset detection latency and its false-positive cost. *Experiment that depends on it:* E3-A. *Assumptions:* supports H2; the false-positive figures challenge any assumption that anticipation is free.

**Lawhern et al. — EEGNet.**
*Why we read it:* our compact-CNN decoder arm. *Insight it contributes:* why a small, structured architecture outperforms larger ones on small EEG datasets — the design reasoning matters more than the architecture. *Experiment that depends on it:* T1-R, E2-A, E3-A. *Assumptions:* supports the choice of three decoder families to separate method effect from signal effect.

**Lotte et al. — review of classification algorithms for EEG-based BCIs.**
*Why we read it:* prevents reinvention and over-reach. *Insight it contributes:* the method landscape and, importantly, why simple methods frequently win at our sample sizes. *Experiment that depends on it:* T1-R baseline selection. *Assumptions:* supports keeping a linear baseline in every comparison.

**Luciw, Jarocka & Edin — WAY-EEG-GAL data descriptor.**
*Why we read it:* our spine dataset. *Insight it contributes:* exactly what was recorded, how, what the event labels mean, and what the protocol was. *Experiment that depends on it:* E2-A, E3-A. *Assumptions:* its protocol description is the ground truth against which our dataset card is verified.

---

## 7.2 Strongly recommended

**Barachant / Congedo — Riemannian approaches to BCI.**
*Why:* our Riemannian tangent-space decoder arm and, later, cross-session alignment. *Insight:* why covariance-based representations are robust to some forms of non-stationarity. *Depends:* T1-R, E2-A, and the Week 13+ encoder for E4. *Assumptions:* supports the belief that session drift is partially correctable geometrically.

**Schwarz & Müller-Putz — reach-and-grasp decoding from EEG.**
*Why:* the closest published analogue to our task. *Insight:* what class cardinality is achievable and with which features — this is the source of the low-70s binary and mid-60s four-class figures in our belief table. *Depends:* RQ-A1, RQ-A2. *Assumptions:* supports the discrete-selector architecture; **its exact values are being re-verified by Track E's Numbers Audit and must not be used as a threshold until that clears.**

**Varoquaux — cross-validation pitfalls in neuroimaging.**
*Why:* calibrates how to read every number we compare ourselves against. *Insight:* why small-*n* cross-validation is unstable and how it misleads. *Depends:* every experiment's split policy. *Assumptions:* supports the block-wise splitting mandate.

---

## 7.3 Background

**Surrogate-data methodology.** *Why:* you use `eval/surrogates/` daily and must know what each surrogate preserves and destroys. *Insight:* the logic of constructing a null that removes the effect while preserving nuisance structure. *Depends:* every experiment's control panel.

**BCI illiteracy literature.** *Why:* the 15–30% constraint is a design parameter, not trivia. *Insight:* that system performance is partly a property of the user. *Depends:* RQ-A5; the EMG-only fallback in the architecture.

**Statistical power in neuroscience.** *Why:* at n = 12 a null and an underpowered study are easy to confuse. *Insight:* why underpowered positives are unreliable even when significant. *Depends:* how we report every null.

---

## 7.4 Historical

**Bernstein — the degrees-of-freedom problem.** *Why:* the origin of the dimensionality framing the whole project inherits. *Insight:* why the motor system must be simplifying a problem it cannot solve variable-by-variable. *Depends:* nothing directly in Track A; read for shared vocabulary with Track B.

**Gallego, Perich & Miller — neural manifolds.** *Why:* read specifically to understand **why the scalp analogy is a category error.** The invasive manifold literature rests on hundreds of simultaneously recorded single units; we have ~16–32 channels of volume-conducted scalp potential. *Insight:* what the manifold claim actually requires. *Depends:* nothing — **do not import its methods.** *Assumptions:* its requirements are what made "neural synergies from EEG band power" a rejected mechanism in this project.

**EEG foundation-model literature.** *Why:* exploratory only, so the track can say why it is not adopted in Phase II. *Depends:* nothing.

---

# 8. SOFTWARE

## 8.1 Where Track A's code lives

```
regenesis/
├── data/                    # loaders + version pointers + dataset cards
│   ├── way_eeg_gal.py
│   ├── bci_iv_2a.py
│   └── cards/               # one markdown card per corpus
├── preprocessing/           # ◄ TRACK A OWNS
│   ├── eeg_filters.py       # band-pass, notch, Laplacian
│   ├── eeg_artifact.py      # ICA, rejection; includes the planted-effect validator
│   ├── epoching.py          # onset-relative epoching
│   └── features_eeg.py      # MRCP + band power, band-resolved routing
├── eeg/                     # ◄ TRACK A OWNS
│   ├── decoders_linear.py   # sLDA, CSP
│   ├── decoders_riemann.py  # tangent-space
│   ├── decoders_eegnet.py
│   ├── montage.py           # full / central-only ablation utilities
│   ├── myogenic.py          # high-frequency myogenic component estimator
│   └── anticipation.py      # E3 detector + threshold sweep
├── eval/                    # ◄ TRACK E OWNS — CONSUME ONLY, NEVER COPY
├── experiments/
│   ├── t1r_reproduction.yaml
│   ├── e2a_controls.yaml
│   └── e3a_anticipation.yaml
├── configs/                 # shared config fragments; seeds pinned
├── prereg/                  # PR-YYYY-NN.md, immutable after sign-off
├── results/                 # one directory per pre-registration
├── ledger/                  # versioned beliefs
└── tests/
```

**Track A never writes into `eval/`.** If a splitter, surrogate, metric or statistical test is missing, file an issue against `eval/` and Track E implements it. A local copy is a defect, caught at code review.

## 8.2 Scripts and their expected outputs

| Script | Input | Output | Written to |
|---|---|---|---|
| `data/way_eeg_gal.py` | raw corpus | canonical in-memory schema + version hash | — |
| `preprocessing/*` | canonical schema | filtered, epoched, feature matrices | cached under `results/<PR>/cache/` |
| `eeg/decoders_*.py` | feature matrices + `eval/splits/` fold | per-fold predictions, decision values | `results/<PR>/raw/` |
| `eeg/montage.py` | feature matrices | montage-ablated variants | `results/<PR>/raw/` |
| `eeg/myogenic.py` | full montage | myogenic component per epoch | `results/<PR>/raw/` + delivered to Track C |
| `eeg/anticipation.py` | onset-aligned features | lead × TPR × FP/min sweep | `results/<PR>/raw/` |
| `experiments/*.yaml` | — | orchestrates the above end to end | `results/<PR>/` |

## 8.3 Naming conventions

- **Pre-registrations:** `PR-YYYY-NN.md`, sequential, never renumbered.
- **Result directories:** `results/PR-YYYY-NN/`, matching the pre-registration exactly.
- **Figures:** `figure.png` for the single headline figure; supplementary as `fig_<slug>.png`.
- **Configs:** `<experiment>_<variant>.yaml`, e.g. `e2a_controls_central_only.yaml`.
- **Branches:** `track-a/<experiment>-<short-description>`.
- **Ledger entries:** the belief ID from the laboratory table, e.g. `ledger/B19.md`.

## 8.4 Logging

Structured logging, mandatory, at every stage:
- Dataset version hash and access date.
- Every preprocessing parameter — band edges, filter order, ICA component count and rejection criterion, epoch window, baseline window.
- Split policy and fold assignment.
- Hyperparameter search space, budget, and the selected values per fold.
- Random seed.
- Wall-clock runtime per stage.

**A parameter that is not logged does not exist.** If a reviewer cannot tell from the log which band edges produced a figure, the figure is withdrawn.

## 8.5 Experiment tracking

Every run is registered with its config, seed, data hash and git commit. A run that cannot be tied to a commit is not a result. Failed and abandoned runs are kept — they are the evidence for `deviations.md`.

## 8.6 Configuration management

Config-as-code. No parameter is passed on the command line that is not also in the config. **Configs are committed before the run**, not after, so that the record shows what was intended rather than what worked.

## 8.7 Version control expectations

- One pre-registration, one branch, one merge.
- Code review before merge, by someone outside Track A for anything that touches a shared interface.
- **Pre-registrations are immutable after sign-off.** Deviations are appended to `deviations.md`, dated, never edited into the original.
- Commits reference the pre-registration ID.
- No result is cited internally before its branch is merged.

## 8.8 Reproducibility floor

`env.lock`, `data.sha256`, `seed.txt` in every result directory. The binding test: **someone outside Track A regenerates the figure from the repository in under thirty minutes.** Run this test on the track's own results before claiming completion, not after someone else fails.

---

# 9. WEEKLY WORKFLOW

Not a calendar — a rhythm. The laboratory's meetings are fixed points; the rest is how the track uses the time between them.

## 9.1 The shape of a typical week

**Monday — Lab Meeting (90 min) and re-planning.**
One track presents deeply on rotation. When it is Track A's turn: 25 minutes talk, 45 minutes interrogation, 20 minutes recording decisions. **The chair is never the presenter.** Anything decided is written to `decisions/` before people leave the room.
After the meeting, the track spends an hour re-planning the week against what changed. If nothing changed, say so explicitly — a week in which no belief moved is worth noticing.

**Tuesday — deep work, building.**
The heaviest implementation day. New pipeline stages, new decoders, new controls. Nothing is presented on Tuesday; things are built.

**Wednesday — Figure Clinic (45 min) and analysis.**
**Every member brings one figure. No exceptions, including weeks when nothing worked** — a figure showing that nothing worked is a figure. Five minutes each. Every figure receives the same first question: *what would this look like if your hypothesis were false?* If you cannot answer, the figure is withdrawn and returns next week.
The rest of Wednesday is analysis: running the sweeps, generating the panels, checking the surrogates.

**Thursday — controls, validation, writing.**
The day reserved for the work that is easy to defer forever: running the surrogate panel, checking the split policy, writing `methods.md`, updating the dataset card, filling the ledger. **Thursday is when a result becomes a result.** A track that treats Thursday as overflow capacity for Tuesday's work will arrive at Week 12 with figures and no evidence.

**Friday — Journal Club or Adversarial Review (60 min), then close-out.**
*Odd weeks — Journal Club.* Predict-then-read sheet plus one reproduced number or a documented failed attempt. Summary-only presentations are not permitted.
*Even weeks — Adversarial Review.* Red team / blue team on a live hypothesis, or pre-registration sign-off. In the duel, one side designs the strongest experiment that would demonstrate the claim; the other designs the cheapest confound producing the same positive result with the effect absent. **The demonstrating side wins only by naming a control that defeats the confound.** If it cannot, the experiment is not ready and the confound goes into the pre-registration as a required control.
Friday afternoon: close out the week's result directories, commit, update the notebook.

## 9.2 What happens every week without exception

- One figure per member at Figure Clinic.
- Laboratory notebook entries, daily, append-only, in-repository, one file per week:

```
## YYYY-MM-DD
Question today:
What I did:
What I saw:            (a number or a figure reference — not adjectives)
What I now believe, and how strongly:
What changed since yesterday:
Next:
The observation that would make me abandon this direction:
```

The last line is mandatory. **A week whose entries all carry the same unchanged abandonment condition is a week of no thinking.**

- Every citation touched that week is graded and entered in the Evidence Table.
- Every result directory reaching the §0.3 standard is closed and committed.

## 9.3 Monthly

- **Kill Meeting.** Something must be killed, downgraded, or moved to the graveyard. A Kill Meeting that kills nothing is recorded as a failed meeting and repeated within a fortnight.
- **Numbers Audit.** Every Track A entry on the Number Board is re-traced by someone who did not enter it.
- **State of Belief memo** (two pages, rotating author): what I believed a month ago, what I believe now, what moved it, what I am least certain of.

## 9.4 When experiments stop and discussions begin

The track follows the laboratory's four-move discussion protocol, and the chair enforces it:

1. **Claim.** State it as a sentence that could be false, with its scope.
2. **Truth condition.** What observation would settle it?
3. **Cost.** How long would that observation take?
4. **Route.**
 - **Under a day → stop talking and measure.** This is the default and it is the most common error to miss. The most frequent failure in a young laboratory is spending forty minutes on a question that fifteen minutes of computation would answer.
 - **Expensive → record both positions in the ledger as competing entries with a discriminating experiment, and do not resolve.** A disagreement that cannot be settled becomes two ledger entries and one experiment, not a winner.
 - **Definitional → write the definition down, adopt it by fiat, move on.**

Disagreements are never resolved by seniority and never by fluency. **The most articulate person in the room is not more likely to be right**, and the chair actively solicits the least confident position before closing any discussion.

## 9.5 Stop-work triggers

Halt the current analysis and escalate the same day if: accuracy exceeds expectation by a wide margin; a surrogate fails to behave as specified; the ICA validator kills a planted effect; onset labels fail hand-verification; or two Regenesis documents contradict each other.

---

# 10. VALIDATION CHECKLIST

An experiment is not complete until **every** box below is ticked. This checklist is applied per experiment, not once per track.

### Code
□ All parameters in a committed config; nothing passed only on the command line
□ Config committed **before** the run
□ Random seed pinned and recorded
□ Run tied to a git commit
□ No local copy of any splitter, surrogate, metric or statistical test
□ Code reviewed; anything touching a shared interface reviewed by someone outside Track A
□ Branch merged before the result is cited internally

### Data
□ Corpus downloaded, opened, and recounted against its publication
□ Channel count, sampling rate and channel names verified
□ Event-code semantics verified against the source documentation
□ **EEG–EMG synchronisation measured and the number recorded** (WAY-EEG-GAL)
□ Onset labels hand-verified on ≥30 randomly chosen trials
□ Errata searched for and findings recorded
□ Dataset card written, with the exact URL and access date
□ `data.sha256` in the result directory

### Plots
□ Required pre-modelling visualisations produced (§6) **before** any decoder was trained
□ ERD/ERS time-frequency map inspected — the phenomenon is visible before decoding was attempted
□ Headline figure present as `figure.png`
□ Every figure carries per-subject points, not only a group mean
□ Every figure caption states the scope: population, condition, feature family, montage, split policy
□ Where the grasp vocabulary does not support a claim, the limitation is **in the caption**, not only in the text

### Statistics
□ Bootstrap percentile CIs (≥1000 resamples) on every reported value
□ Paired Wilcoxon signed-rank for every within-subject comparison
□ Permutation test against the specified null
□ Multiple-comparison correction applied and **stated in the pre-registration, not chosen afterwards**
□ Per-subject distributions reported, not only grand means
□ For any null result: the detectable effect size at our *n* reported alongside it

### Ablations
□ Three decoder families run, so method effect is separable from signal effect
□ MRCP versus band power compared with **equal tuning budgets**, logged
□ Full montage versus central-only compared, paired per subject
□ Before versus after myogenic conditioning compared, paired per subject

### Controls
□ Block-wise split confirmed by inspection of fold membership, not assumed
□ Preprocessing statistics confirmed fitted within fold
□ Hyperparameters confirmed selected on a nested inner fold
□ Full surrogate panel run and behaving as specified
□ **ICA validated against a planted effect** — it does not kill real signal
□ For E3: time-reversed surrogate, shuffled-onset null, pre-baseline-shift check
□ `surrogates.md` written, stating what each control targets and what it showed

### Documentation
□ `claim.md` — **one sentence, with scope, nothing else**
□ `methods.md` — sufficient for a stranger to rerun
□ `deviations.md` — every departure from the pre-registration, dated
□ Ledger entry opened or updated, with confidence and evidence links
□ Number Board entry for any value that will be compared against or cited
□ Laboratory notebook current

### Reproducibility
□ `env.lock`, `data.sha256`, `seed.txt` present
□ **Someone outside Track A regenerated the figure from the repository in under thirty minutes**
□ The regeneration was actually performed and logged, not assumed

### Internal review
□ Pre-registration was red-teamed before sign-off
□ Result presented at Figure Clinic and the null question answered
□ Result presented at Lab Meeting where it bears on a gate
□ For E2-A: Track C has received and integrated the controls, and confirmed in writing
□ For E3-A: Track C has received the detector output

---

# 11. COMMON FAILURE MODES

For each: why it happens, how to detect it, how to recover.

## 11.1 Scientific mistakes

**Reporting a positive that is myogenic.**
*Why:* the artifact produces exactly the pattern the hypothesis predicts, including the degradation conditionality, so it feels like confirmation.
*Detect:* the effect is concentrated above 20 Hz; it collapses under a central-only montage; it largely disappears after myogenic conditioning.
*Recover:* **this is the designed outcome of the control, not a failure.** Report it as the headline. Gate B is decided accordingly and the track has done its most important job.

**Treating a benchmark motor-imagery number as evidence for H1.**
*Why:* it is a real EEG decoding number and the distinction feels pedantic.
*Detect:* any claim about EEG's value to Regenesis citing BCI-IV-2a.
*Recover:* scope clause. The benchmark establishes that the pipeline works.

**Making a grasp-type-diversity claim on a corpus with a narrow grasp vocabulary.**
*Why:* the analysis runs and produces numbers.
*Detect:* a figure comparing grasp types on WAY-EEG-GAL without a limitation in the caption.
*Recover:* restate the claim within scope, or move the analysis to a corpus that supports it.

**Presenting a single operating point instead of a curve in E3.**
*Why:* a single number is easier to state and looks more decisive.
*Detect:* one TPR value in the figure.
*Recover:* sweep. The curve is the deliverable, and it is publishable even when the operating point is not usable.

**Confusing "not significant" with "no effect" at n = 12.**
*Why:* a null feels conclusive when it is reported alone.
*Detect:* any null without the detectable-effect number beside it.
*Recover:* report power alongside every null.

## 11.2 Coding mistakes

**Window-level splitting through overlapping epochs.**
*Why:* epoching with overlap is standard and the leakage is invisible in the code.
*Detect:* accuracy well above the literature. **Suspect leakage before celebrating.**
*Recover:* `eval/splits/`, always; inspect actual fold membership rather than trusting the call.

**Preprocessing statistics fitted across the full dataset.**
*Why:* it is the natural order in which to write the script.
*Detect:* code review at the normalisation step.
*Recover:* refit within fold; re-run everything downstream.

**ICA over-correction removing movement-correlated neural activity.**
*Why:* aggressive artifact rejection looks like rigour.
*Detect:* decoding drops sharply after ICA on central channels; the planted-effect validator fails.
*Recover:* validate the ICA step against a planted effect before adoption, exactly as Track E validates surrogates. If it kills the planted effect, fix or drop it.

**Silent parameter drift between runs.**
*Why:* a band edge is changed to see what happens and never changed back.
*Detect:* two figures with the same name and different parameters in the log.
*Recover:* config-as-code, committed before the run.

## 11.3 Statistical mistakes

**Unequal tuning budgets between compared arms.**
*Why:* the interesting arm gets more attention. In this track that is usually MRCP over band power, or EEGNet over sLDA.
*Detect:* budgets not logged.
*Recover:* equal budgets, logged, re-run both.

**Grand means hiding subject heterogeneity.**
*Why:* one number is tidier.
*Detect:* no per-subject points in the figure.
*Recover:* per-subject distributions, always. Given BCI illiteracy, the spread is often the finding.

**Multiple comparisons across bands and operating points without correction.**
*Why:* each test is run separately and the family is never counted.
*Detect:* correction not stated in the pre-registration.
*Recover:* Holm correction across the declared family; if the family was not declared in advance, report both corrected and uncorrected and say so.

**Choosing the correction after seeing the result.**
*Why:* it is the easiest available error and rarely feels like one.
*Detect:* correction appears in `deviations.md` rather than the pre-registration.
*Recover:* report both, log the deviation, and treat the finding as exploratory.

## 11.4 Interpretation mistakes

**Reading temporal precedence as a latency advantage.**
*Why:* the older readiness-potential framing is intuitive and still common in the literature.
*Detect:* a claim that EEG "buys time" without a downstream latency measurement.
*Recover:* precedence is structural — `g_EEG` taps the state upstream of `g_EMG`. Whether it buys usable latency is Track C's forward-feed measurement, not ours.

**Treating a Level-3 representation as the Level-1 state.**
*Why:* the language invites it — "the latent" sounds like a thing.
*Detect:* a claim that a decoder recovered motor intent, rather than that it decoded a label.
*Recover:* restate. Intent is unobservable in principle.

**Letting a strong result stand without its scope clause.**
*Why:* the qualifiers weaken a sentence you are pleased with.
*Detect:* a claim in a slide or abstract without population, condition and offline/online.
*Recover:* the scope qualifier is part of the claim, not a caveat on it.

**Concluding "EEG doesn't work" from one corpus never designed for the question.**
*Why:* a null on WAY-EEG-GAL feels general.
*Detect:* a general claim sourced from a single narrow-vocabulary dataset.
*Recover:* scope it to the corpus and its vocabulary.

## 11.5 Literature mistakes

**Quoting a paper's Introduction as evidence.**
*Why:* introductions state claims cleanly and are the easiest part to quote.
*Detect:* the quoted number is not in the paper's own Results.
*Recover:* it is secondhand — grade one level down and trace to origin, or remove.

**Citing a value without its conditions.**
*Why:* the number is memorable and the conditions are not.
*Detect:* the Evidence Table entry lacks *n*, class count, chance level or split policy.
*Recover:* extract them or delete the entry. **A value quoted without its conditions will be compared against one of ours obtained under different conditions, and once it is in a table the mismatch is invisible.**

**Setting a threshold against an unverified published value.**
*Why:* the value is in the Handbook and looks settled.
*Detect:* a pre-registered threshold whose source has no Numbers Audit verdict.
*Recover:* wait for Track E's audit, or set the threshold from our own reproduced value instead.

**Importing invasive-manifold methods by analogy.**
*Why:* the mathematics transfers cleanly and the language is attractive.
*Detect:* any proposal to decompose EEG into "neural synergies."
*Recover:* the mechanism was removed from this project as a category error. It is in the graveyard with its revival condition. Do not reopen it without a written PI decision.

---

# 12. DELIVERABLES

Exactly what Track A hands to the laboratory.

## 12.1 Research notes
- **RN-A1 — What scalp EEG delivers for grasp intent in our data.** Per-subject accuracy at *k* = 2, 3, 4 with CIs, feature family compared, decoder families compared, subject-versus-method variance, full scope clause. Answers RQ-A1, RQ-A2, RQ-A5.
- **RN-A2 — Cortical versus myogenic contribution.** The three control figures, the quantified before/after myogenic-conditioning difference, and an unhedged statement of what they show. Answers RQ-A3. **Written honestly whichever way it goes.**
- **RN-A3 — The anticipatory operating curve.** Lead × TPR × FP/min for onset and, where supported, type, with the surrogate panel and the vocabulary limitation stated. Answers RQ-A4.

## 12.2 Presentations
- Weekly Figure Clinic figure (every member, every week).
- Lab Meeting deep presentations on rotation — first in Week 3, then per the laboratory rotation.
- **The Gate B evidence presentation**, jointly with Track C, at which Track A presents the artifact analysis **first**, before any positive result.
- Three Journal Club presentations, each with a reproduction attempt.

## 12.3 Notebooks
Laboratory notebooks, per member, append-only, in-repository, one file per week, daily entries per §9.2. These are deliverables, not personal records — they are read at handover and at the monthly State of Belief memo.

## 12.4 Plots

| ID | Content | Destination |
|---|---|---|
| T1-R.1 | Per-subject reproduced-vs-published scatter with identity line | Number Board; Publication 1 supporting |
| E2-A.1 | Decoding accuracy by frequency band, per subject, with null band | Publication 2 |
| E2-A.2 | Full montage vs central-only, paired per subject | Publication 2 |
| **E2-A.3** | **Before vs after myogenic conditioning, paired per subject** | **Publication 2, main text — the honesty figure** |
| E3-A.1 | Lead time vs TPR at fixed FP/min, per subject + group median | Publication 3 |
| E3-A.2 | The same for grasp type, where supported | Publication 3 |
| E3-A.3 | Surrogate panel — curve under time reversal and shuffled onset | Publication 3 supporting |
| — | MRCP vs band power, paired per subject | RN-A1 |
| — | Per-subject × per-method variance table | RN-A1 |

## 12.5 Tables
- Per-subject accuracy × class cardinality × feature family × decoder family, with CIs.
- Per-band decoding table.
- Montage-ablation table.
- Evidence Table rows for every paper this track read, graded, with design descriptors.
- Number Board entries for every Track A value, with conditions and provenance.

## 12.6 Code
- `preprocessing/` and `eeg/` complete, tested, documented.
- Three decoder families behind a common interface.
- Band-resolved routing, montage-ablation utilities, myogenic estimator — **delivered to Track C as a stable interface, not as scripts to be copied.**
- The E3 detector and threshold sweep.
- An EEG encoder for E4's paired arm (Weeks 13+).

## 12.7 Documentation
- Dataset cards for WAY-EEG-GAL, BCI-IV-2a, and any secondary corpus used.
- Pre-registrations: T1-R, E2-A (joint with Track C), E3-A.
- Result directories at the §0.3 standard.
- Ledger updates for the EEG-related beliefs, with confidence, scope, evidence links and refutation conditions.

## 12.8 Decision recommendation

Track A does not decide Gate B or Gate C. It hands the laboratory a **written recommendation** with each:

- **For Gate B:** a signed statement of whether the EEG contribution to E2 survives the three controls, with the magnitude of the myogenic component quantified, and an explicit recommendation on whether Gate B may be decided on the observed effect.
- **For Gate C:** the operating curve with a recommendation on whether any operating point is usable for control, and if not, a statement of what the curve nonetheless establishes.

Both recommendations state their scope and both are written to be readable by someone who was not in the room.

---

# 13. COMPLETION CRITERIA

Track A is **scientifically complete** — not "finished coding" — when all of the following are objectively true.

### Necessary conditions

**1. The three experiments are closed at the §0.3 standard.** T1-R, E2-A and E3-A each have a complete result directory, and each has been regenerated by someone outside Track A in under thirty minutes. This has been performed and logged, not assumed.

**2. Every research question in §4 has an answer or a documented reason it is undecidable with the available data.** RQ-A1 through RQ-A5. An honest "undecidable at n = 12 with this grasp vocabulary" satisfies this criterion; silence does not.

**3. The artifact analysis is complete and its verdict is recorded, whichever way it went.** All three control figures exist, the before/after myogenic-conditioning difference is quantified, and Track C has confirmed in writing that the controls are integrated into E2. **This criterion is satisfied equally by a finding that the effect is cortical and by a finding that it is myogenic.** It is not satisfied by an unmeasured effect.

**4. The E3 operating curve exists with per-subject spread, CIs and a passing surrogate panel.** Satisfied whether or not any operating point is usable.

**5. Every Track A value on the Number Board carries provenance and conditions**, and has been re-traced at a monthly Numbers Audit by someone who did not enter it.

**6. The belief ledger reflects what we now know.** Every EEG-related belief has moved to a scoped entry with a confidence level and evidence links, or is recorded as undecidable with the available data and why. The scope clause on every entry is present and correct.

**7. Both decision recommendations are written and delivered** — Gate B and Gate C — readable by someone who was not in the room.

**8. Every member has both written and received a serious internal review** in the laboratory's format: summary, significance, three major concerns, minor concerns, recommendation; plus a written response to reviewers including the disagreements.

**9. Every member passes the reconstruction test.** Handed a hypothesis statement with its Confirm and Refute clauses deleted, each member can write the missing clauses. Applied to H1 and H2, and to the common knowledge of §2. **A member who can reconstruct a falsification condition owns the work; a member who can only recognise it is carrying it.** This criterion is not softer than the others.

### What does NOT constitute completion

- A positive H1-relevant result. **Not ours to produce.** Track C measures conditional MI; we make its measurement valid.
- Gate B passing. Gate B may fail on our evidence and the track is still complete — indeed, a track that produced the artifact finding that stopped a false gate has done more for the project than one that produced a marginal positive.
- A usable E3 operating point. The curve is the deliverable.
- All code merged. Code is a means; the criteria above are the ends.

### The single test

If a new member joined the laboratory tomorrow, read only this manual and Track A's result directories, and could state — with numbers, scope clauses and their evidence — **what scalp EEG delivers for grasp intent in this project, how much of it is real, and what would change that answer**, then Track A is complete.

---

*Track A Operating Manual ends. The Phase II Handbook remains the laboratory's scientific operating manual; raise any disagreement between the two the same day.*
