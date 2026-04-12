# Project Journey: Avatar Arcade Gesture Recognition

## Project goal
Build a robust gesture-recognition ML pipeline for an elemental-combat game using OpenPose-derived skeleton sequences and a reproducible training/evaluation workflow.

## OpenPose setup journey
Early work focused on making OpenPose capture stable enough for repeated sessions. The team moved through setup friction, then landed on a reliable directory and recording workflow that made repeated capture practical.

## Data capture workflow
Data is organized by gesture/person/session/take so each take is one training sample:

`data/raw/openpose_json/<gesture>/<person>/<session>/<take_###>/`

This made it possible to track provenance and rebuild processed tensors deterministically from raw takes.

## Dataset design decisions
The project converged on a 9-class label space:

- attack_air
- defense_air
- attack_fire
- defense_fire
- attack_water
- defense_water
- attack_earth
- defense_earth
- idle

A key decision was treating `idle` as a real class, not a placeholder, so the model can explicitly learn non-action states.

## Feature representation decisions
Initial preprocessing used all BODY_25 joints. In practice, recordings were mostly hip-up, so lower-body points were frequently missing/noisy.

The project switched to a 15-joint upper-body subset:
Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, MidHip, RHip, LHip, REye, LEye, REar, LEar.

Using x/y only yields final per-sample shape `(90, 30)`.

## Missing-data and normalization decisions
The final normalization/repair strategy is:

- Center anchor: Neck (fallback MidHip)
- Robust weighted scale: shoulder width + torso length + hip width
- Temporal scale smoothing
- Weak confidence filter only
- Bad-frame repair and suspicious identity-switch/catastrophic-jump handling

This produced a usable representation even under imperfect OpenPose tracks.

## Preprocessing findings
Preprocessing quality improved significantly after upper-body focus and robust repair logic. The pipeline now reliably produces training tensors and metadata from raw takes.

## Dataset size corrections
A major correction was identifying stale processed/training context. The valid working dataset is 459 total samples (51 takes per class across 9 classes), not older smaller subsets used in early debugging runs.

## First modeling attempts
Early LSTM training on stale/smaller processed data (~140 samples) performed near chance, indicating either data, split, or modeling issues were still unresolved.

## Tiny-overfit debugging
Tiny-overfit tests became the fast diagnostic tool:

- Tiny overfit LSTM could not memorize 18 samples (stalled around ~33% train accuracy).
- Tiny overfit MLP memorized 18 samples to 100% train accuracy.

This strongly suggested that labels/features were learnable and pointed to sequence-model setup as the bottleneck.

## Split bug and fix
Another key issue was stale split reuse. Full-model metrics could look inflated when old split files were accidentally reused with updated datasets. Split validation and force-resplit logic were added to prevent stale/invalid split artifacts from silently skewing results.

## MLP baseline emergence
With corrected dataset and valid splits, full-dataset MLP became the strongest reliable baseline:

- ~89.86% validation accuracy
- ~91.30% test accuracy

Confusion patterns are generally strong, with remaining concentration around `attack_fire` vs `defense_fire` and slight `defense_earth` vs `idle` overlap.

## Sequence-model difficulties
The motion-aware sequence attempt (`lstm_motion`, using positions + deltas with input `(90, 60)`) still performs near chance (~11.6%).

Current interpretation:

- Dataset is learnable
- Preprocessing is usable
- MLP proves representation signal exists
- Current sequence-model formulation/training remains the weak point

## Current best result
The current best valid baseline is the **full-dataset MLP** trained on the corrected 459-sample dataset with validated train/val/test split handling.

## Open next steps
1. Add reproducible suite-level experiment orchestration (isolated run folders + summary artifacts).
2. Expand sequence-model search (GRU variants, alternative temporal encoders, tuned regularization, learning-rate schedules).
3. Add richer error analysis for confused class pairs.
4. Keep project-journey + experiment logs synchronized for report-ready documentation.

## Appendix: recent local git history (last 30)
Command used: `git log --oneline --decorate -n 30`

```text
0087c4a (HEAD -> work) Merge pull request #25 from MacquePanoramix/codex/improve-sequence-model-training-pipeline
5a63b54 Improve sequence training with standardization and GRU option
da66253 Merge pull request #24 from MacquePanoramix/codex/add-motion-aware-lstm-baseline
ce90508 Add motion-aware LSTM baseline option
55acb71 Merge pull request #23 from MacquePanoramix/codex/improve-mlp-training-settings-and-callbacks
761f8f1 Tune MLP training run length and callbacks
210fead Merge pull request #22 from MacquePanoramix/codex/enhance-training-script-for-split-validation
cf30be2 Harden split reuse with validation and force-resplit flag
941cd0e Merge pull request #21 from MacquePanoramix/codex/add-mlp-baseline-to-training-script
6c18aa6 Add full-dataset MLP baseline option to training script
018af3a Merge pull request #20 from MacquePanoramix/codex/enhance-tiny-overfit-diagnostic-in-train_lstm.py
a8a3358 Strengthen tiny overfit diagnostics with LSTM/MLP modes
5225bd2 Merge pull request #19 from MacquePanoramix/codex/add-tiny-overfit-diagnostic-mode-to-training-script
e4992df Add tiny overfit diagnostic mode to LSTM trainer
de13e7f Merge pull request #18 from MacquePanoramix/codex/implement-first-lstm-training-baseline
2895eb1 Implement first baseline LSTM training pipeline
60e0648 Merge pull request #17 from MacquePanoramix/codex/update-preprocessing-for-upper-body-body_25-subset
bead1d3 Use upper-body BODY_25 subset in preprocessing
6d0285f Merge pull request #16 from MacquePanoramix/codex/update-joint-level-imputation-policy
759a915 Update joint-level imputation for OpenPose preprocessing
c53e7ea Merge pull request #15 from MacquePanoramix/codex/implement-preprocessing-pipeline-v1
d7daf61 Add OpenPose preprocessing v1 dataset builder
9e9ff3d Merge pull request #14 from MacquePanoramix/codex/update-repository-for-agreed-dataset-schema
4c707ac Lock 9-class dataset schema and capture defaults
c88f052 Merge pull request #13 from MacquePanoramix/codex/add-idle-pose-capture-workflow
1f29605 Add continuous idle pose capture workflow
97bb426 Merge pull request #12 from MacquePanoramix/codex/extend-record_gesture_cycle_continuous.ps1-for-review-video
d36876f Add optional continuous review video and cycle manifest
446d31c Merge pull request #11 from MacquePanoramix/codex/fix-bugs-in-record_gesture_cycle_continuous.ps1-p6tx9t
3fb2e98 Fix continuous capture diagnostics and take label formatting
```

Short appendix summary: recent history shows a clear progression from capture/preprocessing stabilization to split correctness hardening, then MLP baseline strength, while sequence-model variants are still under active iteration.

## Follow-up experiment round: motion-feature comparison

The strongest current reference remains **full_mlp** (about 91.30% test accuracy on the corrected full dataset).  
At this stage, plain `full_lstm` is not the priority because it still underperforms substantially.  
However, `full_lstm_motion` is now interesting because it rose above chance-level and tiny-overfit checks indicate the sequence path can learn.

This reframes the next question:

**Does motion help because the feature representation is better, or only when combined with recurrence?**

To answer that cleanly, the next round compares:

- `mlp_motion` (same pose+delta features, flattened into an MLP)
- `lstm_motion` with checkpoint selection by `val_accuracy` for rerun stability
- `gru_motion` (same motion features, alternative recurrent cell)
- and keeps `full_mlp` in the suite as the practical baseline anchor

### What each outcome would mean

- If `mlp_motion` beats `full_mlp`, then motion features themselves are useful even without recurrence.
- If `mlp_motion` does not improve but `gru_motion` (or `lstm_motion` with val-accuracy checkpointing) improves, recurrence may be extracting meaningful temporal structure.
- If neither motion MLP nor motion recurrent models beat `full_mlp`, the flattened pose baseline remains the practical best model for now.

## Error analysis phase: inspecting full_mlp misclassifications

At this stage, the project direction shifts from architecture exploration to targeted error inspection.
The practical baseline conclusion is unchanged: **full_mlp remains the strongest model** on the corrected full dataset.
Motion follow-up experiments (including motion-feature MLP and recurrent motion variants) did not surpass full_mlp in a way that changes that baseline decision.

Because of that, the next logical step is not to replace the baseline, but to understand its remaining mistakes in more detail.
This error-analysis phase is intended to answer whether residual errors are mostly caused by:

- genuinely similar gesture patterns (for example, same-element attack vs defense overlap),
- label ambiguity in some takes,
- inconsistent performance concentrated in certain people/sessions/takes,
- preprocessing or normalization artifacts,
- timing misalignment inside fixed 90-frame windows.

### Error analysis implementation artifacts added

- `src/analysis/analyze_misclassifications.py`
  - New CLI workflow to analyze one completed run folder and produce machine-readable + human-readable reports.
  - Main entrypoint stays simple: `python -m src.analysis.analyze_misclassifications --run-dir <run-folder>`.
  - Convenience run targeting includes `--suite-dir` and `--latest-suite-dir` (defaulting to `full_mlp`).
- Per-run analysis outputs under `<run-dir>/misclassification_analysis/`:
  - `summary.json`, `summary.md`
  - `confusions_by_pair.csv`
  - `misclassified_samples.csv`
  - `highest_confidence_errors.csv`
  - `hardest_correct_samples.csv`
  - simple error-visualization plots when data is available.
- Richer training prediction export support in `train_lstm.py`:
  - Writes run-local `predictions.csv` (in addition to backward-compatible `*_test_predictions.csv`),
  - includes sample index, true/pred labels, confidence fields, per-class probabilities,
  - joins preprocessing metadata fields (`gesture`, `person`, `session`, `take`, `sample_path`) when available for take-level traceability.

## Analysis pipeline repair: confidence and metadata traceability

The first misclassification-analysis round was directionally useful, but only partially successful.
It confirmed the dominant confusion pairs (especially `attack_fire` ↔ `defense_fire`, plus `defense_earth` ↔ `idle` overlap), which matched the broader baseline narrative.
However, the analysis export itself still had critical gaps:

- `confidence_of_predicted_class` appeared but contained missing/NaN values in practice,
- sample traceability metadata was not consistently carried through to analysis outputs,
- this prevented reliable person/session/take-level forensic debugging.

Because of that, this repair step focused on two pipeline guarantees:

1. prediction-confidence export must be numerically valid per test row,
2. metadata alignment must preserve sample-level traceability from processed data into run predictions and analysis tables.

### Why this repair matters scientifically

Without confidence values, we cannot distinguish between:

- confidently wrong predictions (possible class boundary or label-ambiguity issue), and
- low-confidence borderline misses (possible representation uncertainty).

Without traceability metadata, we cannot test whether mistakes cluster by:

- person,
- session,
- take,
- or a specific original capture path.

That means interventions would be guesswork instead of evidence-driven.
The repair therefore improves the reliability of targeted debugging while keeping the project conclusion unchanged: **full_mlp remains the practical best baseline**.

### Repair implementation summary

Files changed:

- `src/training/train_lstm.py`
- `src/analysis/analyze_misclassifications.py`
- `README.md`
- `docs/project_journey.md`

New validations/warnings added:

- warnings for prediction/test-size mismatches,
- duplicate `sample_index` detection,
- metadata duplicate/drop alignment warnings,
- explicit confidence NaN/availability checks,
- explicit summary flags for confidence availability and metadata traceability availability.

Repaired outputs now contain (when available):

- predicted and true-class confidence values,
- second-best prediction label/confidence,
- per-sample traceability fields (`gesture`, `person`, `session`, `take`, `original_sample_path`),
- confidence and traceability availability stats in both `summary.json` and `summary.md`.

## Visual error inspection phase: reviewing confusing takes

After repairing misclassification confidence values and metadata traceability, the error pattern looked concentrated rather than random.
The remaining mistakes are still dominated by a few confusion clusters, especially:

- `attack_fire` vs `defense_fire`
- `defense_earth` vs `idle`
- a smaller `attack_air` -> `attack_water` case

That concentration suggested the next step should **not** be another immediate model replacement.
Instead, the more useful next step is targeted visual take inspection on the exact confusing samples from the best `full_mlp` run.

The purpose of this visual phase is to determine whether residual mistakes are driven more by:

- genuinely similar gesture shapes,
- inconsistent performer execution,
- ambiguous timing inside the fixed 90-frame window,
- possible labeling ambiguity on specific takes.

### Implementation note

- Added `src/analysis/review_confusing_takes.py`.
- The script builds visual confusion-case comparisons directly from existing run artifacts (`predictions.csv` + `misclassification_analysis/*.csv`).
- It saves outputs under `<run-dir>/take_review/`, including:
  - `review_summary.md`
  - `review_summary.json`
  - per-case comparison images (`review_case_*.png`)
  - optional `top_confusions_contact_sheet.png` dashboard
- Main command shape:
  - `python -m src.analysis.review_confusing_takes --run-dir models/experiment_runs/<timestamp>/full_mlp`

## Dataset refinement planning phase: targeted recollection and curation

The repaired confidence/traceability analysis and visual take review both succeeded, and they converged on the same practical conclusion:
**the current pose-only full_mlp baseline is still strong**.
At this point, remaining mistakes appear concentrated in a small number of ambiguous gesture boundaries rather than reflecting broad model-capacity failure.

That shifts the next step away from immediate model exploration and toward **data refinement planning**.
The goal of this phase is to convert existing error/review findings into a concrete, limited-scope recollection and curation plan before running more model variants.

Primary boundary targets identified for focused follow-up:

- `attack_fire` vs `defense_fire`
- `defense_earth` vs `idle`

### Why this phase matters

Because residual errors are concentrated, the highest-leverage intervention is to:

- identify which existing takes look ambiguous and should be reviewed first,
- prioritize which class boundaries should receive targeted recollection first,
- improve performer instructions where class definitions are currently too close in execution,
- define practical (modest) recollection batches instead of large broad recapture.

### Implementation note

- Added `src/analysis/plan_recollection.py`.
- Main command:
  - `python -m src.analysis.plan_recollection --run-dir models/experiment_runs/<timestamp>/full_mlp`
- The script reads existing run analysis artifacts and writes recollection planning outputs under:
  - `<run-dir>/recollection_plan/`
- Main outputs:
  - `recollection_plan.md`
  - `recollection_plan.json`
  - `priority_review_takes.csv`
  - `priority_recollect_targets.csv`
  - optional `boundary_priority_bar.png`

This phase is intentionally planning-only: it does **not** retrain models, relabel data automatically, or delete samples.
It provides actionable next steps so dataset improvements can be executed and measured before additional model exploration.

## Live real-time debug classifier for OpenPose JSON stream (v1)

To tighten the loop between offline experiments and practical gameplay integration, we added a **live debug inference tool** for the current best model path (pose-only `full_mlp`).
The immediate goal is to verify real-time behavior while OpenPose is still writing JSON frames, before wiring predictions into any gameplay gate.

### Why this was added

Offline metrics are strong, but they do not directly show how predictions behave under live frame drops, tracking jumps, and rolling-window flicker.
This tool adds an operational check layer that can be run side-by-side with OpenPose Demo output.

### How live preprocessing differs from offline preprocessing

Offline dataset build can use non-causal timeline repair (including start-of-sequence future fill and interpolation across future frames).
Live inference cannot rely on future frames, so runtime preprocessing now uses **causal-only repairs**:

- bad/suspicious whole frame -> copy last accepted processed frame,
- missing joint fallback order:
  1) previous valid joint value,
  2) mirrored symmetric counterpart (if available),
  3) zero fallback.

This keeps live behavior conceptually aligned with the offline normalization intent while staying runtime-safe.

### What v1 includes

- New shared runtime preprocessing module (`src/preprocessing/runtime_preprocess.py`) that matches the same upper-body 15-joint BODY_25 subset and x/y-only representation.
- Same center and weighted-scale strategy as offline (`Neck` primary, `MidHip` fallback; shoulder/torso/hip weighting), with causal temporal scale smoothing.
- Conservative suspicious-jump detection compatible with offline identity-switch handling intent.
- New live CLI tool (`src/inference/live_openpose_debug.py`) that:
  - watches a JSON directory continuously,
  - maintains a rolling `(90, 30)` buffer,
  - flattens to `(1, 2700)` for pose-only MLP inference,
  - shows raw + smoothed prediction and top-3 probabilities,
  - prints repair/debug state and buffer fill,
  - logs inference rows to CSV for later review.

## Live debug inference reporting refinement (v2)

The first real replay run validated that v1 was functionally correct (model/label-map/JSON stream loading, 90-frame warmup, rolling inference, and expected smoothing lag), but it also exposed an interpretation problem in the status output.

### What was confusing in v1 output

- `repair=True` appeared on most frames and looked like severe failure even when only a few joints were missing.
- In runtime preprocessing, `repair=True` actually represented broad "some repair happened" semantics, not specifically catastrophic frame fallback.
- This blurred together minor and major cases:
  - minor: partial joint-level fills (`missing_joint_count > 0`),
  - major: full frame copy-forward fallback (`used_prev_frame_copy=True`),
  - major + likely unstable tracking: `suspicious_jump=True`.

### Why refinement was needed

The replay log was useful, but status semantics were too noisy to trust at a glance.
For live/replay debugging, we need to immediately distinguish:

- routine partial-joint recovery,
- whole-frame fallback events,
- suspicious-jump-triggered protection behavior.

Without that distinction, operators can misread healthy-but-noisy streams as catastrophic instability.

### What changed in v2

- Runtime frame metadata now explicitly includes `had_joint_repair` in addition to existing flags.
- Live console status was compacted and clarified to show:
  - `joints_missing`,
  - `joint_repair` (yes/no),
  - `prev_copy` (yes/no),
  - `suspicious_jump` (yes/no),
  with stronger visual emphasis (`!!`) for major fallback cases.
- Added optional console noise controls:
  - `--print-every-n`
  - `--quiet-warmup`
- CSV logging now carries clearer repair semantics with `had_joint_repair` while retaining prior compatibility columns.
- Added end-of-run summary (printed + JSON artifact) with:
  - frame totals (total/warmup/inference),
  - repair/fallback counts,
  - missing-joint stats (avg/min/max),
  - raw and smoothed class-count histograms,
  - top raw->smoothed disagreement patterns for quick confusion inspection.

### Replay behavior takeaway after refinement

The replay test continues to show realistic temporal behavior:

- smoothing adds expected lag on transitions,
- recurring boundary confusion (notably `attack_earth` vs `defense_water`) is visible and now easier to quantify post-run,
- minor joint repair is common and mostly non-catastrophic,
- serious fallback events can now be isolated clearly for targeted debugging instead of being hidden inside generic `repair=True`.
