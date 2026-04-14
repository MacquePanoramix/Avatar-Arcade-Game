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

## Runtime live-selection refactor: stable person assignment groundwork

Live replay analysis showed repeated suspicious-jump and previous-frame-copy fallback segments during inference, even when camera conditions looked acceptable.
One likely contributor was person selection fragility: the runtime path always took `people[0]`, which can wobble when OpenPose detection ordering changes frame to frame.

That behavior is risky for current single-person debugging and even more risky for the final two-player game plan, where two real users share one camera frame and must remain mapped to stable logical identities:

- player 1 = left-side player
- player 2 = right-side player

### What changed in runtime selection

A lightweight deterministic assignment/tracking layer was added to runtime preprocessing.
The parser now evaluates *all* detected people each frame (not only the first), and computes per-candidate signals used for matching:

- center anchor (Neck preferred, MidHip fallback)
- weighted body scale estimate
- usable joint count
- mean usable-joint confidence

Two tracking modes are now supported:

1. `single_person` (default)
   - on startup, picks the best-quality candidate
   - afterward, matches to the previously tracked person by temporal consistency (center distance + scale similarity + quality tie-break)

2. `two_player_left_right` (future-facing groundwork)
   - keeps separate `left` and `right` tracked states
   - initializes from x-ordering when both are first seen
   - later frames use temporal assignment costs with a weak crossing penalty so identities do not instantly flip on jitter

Current live debug inference still runs one classifier stream, but assignment now exposes left/right selected detection indices and notes, so wiring two independent gameplay/controller streams is straightforward in a follow-up step.

### Why this is the current takeaway

- The runtime path no longer blindly trusts OpenPose detection index ordering.
- Identity consistency is explicitly part of selection heuristics now.
- This reduces one known risk factor for fallback-heavy debug runs.
- It also aligns runtime architecture with the final two-player camera requirement.

### Remaining limitations

- This is intentionally heuristic and lightweight, not a full multi-object tracker.
- Extended long occlusions or full person crossings may still require stronger identity logic later.
- Two-player mode currently prepares assignment/state/debug output, but does not yet run separate model windows/classifiers per player.

## Live OpenPose + classifier one-command launcher (Windows-first)

### Step 1: local external-path config design
- **Attempted:** Added a local runtime-config pattern for external OpenPose path wiring.
- **Why:** This repository does not bundle OpenPose binaries, so hardcoded machine paths in tracked files would break portability.
- **Observed:** Existing repo config (`configs/paths.yaml`) is project-level and not suitable for user-specific executable paths.
- **Changed:** Added tracked template `configs/local_paths.example.yaml` and local-only override target `configs/local_paths.yaml` (gitignored).
- **Takeaway:** Keeping machine-specific paths in an untracked local file enables reliable one-command startup without polluting shared config.

### Step 2: one-command orchestration script
- **Attempted:** Implemented `tools/live/start_live_debug.ps1` to launch OpenPose and the Python live debug classifier in one flow.
- **Why:** Real live testing previously required manual multi-terminal startup and manual folder alignment.
- **Observed:** The key integration point is a shared JSON directory and synchronized runtime arguments.
- **Changed:** Launcher now validates config, prepares a fresh live JSON folder, starts `OpenPoseDemo.exe --write_json <dir>`, then runs `python -m src.inference.live_openpose_debug` against that same directory.
- **Takeaway:** A single launcher command removes most setup friction and lowers operator error during iterative live tests.

### Step 3: shutdown reliability + operator defaults
- **Attempted:** Added clean-stop behavior and practical defaults for live-debug iteration.
- **Why:** Without coordinated shutdown, OpenPose can be left running in the background after classifier exit.
- **Observed:** Keeping classifier in the primary PowerShell session makes Ctrl+C behavior predictable for users.
- **Changed:** Script traps exit via `try/finally`, terminates the spawned OpenPose process, and prints JSON/log/summary output locations. Defaults include `single_person`, `print-every-n=10`, and quiet warmup enabled (toggle via `-NoQuietWarmup`).
- **Takeaway:** Session lifecycle is now easier to control, making repeated debug passes and upcoming two-player live test passes much smoother.

### Step 4: user-facing documentation update
- **Attempted:** Added a README quick-start section for one-command live testing.
- **Why:** The launcher is only useful if setup steps are explicit and copy-paste friendly.
- **Observed:** The minimal setup is a one-time local config copy/edit plus one launcher command from repo root.
- **Changed:** README now documents the exact config file to edit and the launcher command shape.
- **Takeaway:** Onboarding to live OpenPose + classifier testing is now short, repeatable, and report-ready.

## Live-launcher reliability fix: no-frame startup failure (`frames_total=0`)

- **First attempt behavior:** The one-command launcher appeared to start OpenPose and then start the classifier, but session summaries could end with `frames_total=0`, `warmup=0`, `inference=0`.
- **Failure symptom meaning:** The classifier itself was alive, but no live JSON frame files were arriving in the configured `live_json_dir`.
- **Diagnosis:** This was an integration/startup issue (launcher/OpenPose output path handling), not a model/classifier-learning issue.
- **Fix implemented:**
  1. Explicit OpenPose process working directory (defaulting to the executable's parent directory, with optional local override).
  2. Startup handshake that waits for the first JSON file in `live_json_dir` (timeout-based) before starting classifier inference.
  3. Stronger startup diagnostics: resolved executable path, working directory, model folder, JSON output directory, and pre-launch JSON state.
  4. Clear fail-fast error when no JSON appears in time:  
     `OpenPose started but no JSON files appeared in <folder> within <timeout> seconds.`
- **Takeaway:** The launcher now verifies real frame flow before inference starts, preventing silent zero-frame sessions and making OpenPose-output failures immediately visible.

## Launcher safety correction: OpenPose shutdown policy

The first one-command live launcher implementation used aggressive cleanup:

- it waited for first JSON with a short startup timeout,
- force-stopped OpenPose if timeout was hit,
- and force-stopped OpenPose again in `finally` when the classifier exited or Ctrl+C was pressed.

That behavior turned out to be unsafe on at least one real Windows machine, where force-stopping OpenPose can cause system instability/crash risk.

The launcher was therefore changed to **safe-by-default** behavior:

- no automatic OpenPose kill on startup-timeout failure,
- no automatic OpenPose kill on classifier exit/Ctrl+C,
- explicit diagnostics are printed when JSON does not appear,
- OpenPose is intentionally left running for manual inspection/shutdown.

Automatic OpenPose termination is now **opt-in only** via an explicit kill flag.

Current takeaway: for live debugging on heterogeneous Windows setups, manual OpenPose shutdown is the safer default; aggressive process-kill cleanup must be explicit and optional.

## Live debug confidence instrumentation: intended labels + threshold analysis

Recent manual live/replay debug runs showed a useful pattern: idle predictions were usually very strong, while elemental boundaries (especially `attack_fire` vs `defense_fire`) still produced ambiguity. That made it hard to decide a safe deployment confidence threshold from eyeballing terminal output alone.

To make threshold design evidence-driven, we added explicit run-level intended-label tagging and a lightweight confidence summary script focused on live debug CSV artifacts.

### What was attempted and why

- Added run-level `--intended-label` support to `src.inference.live_openpose_debug`.
  - Goal: tag each replay/live-debug run with the gesture we expected, so confidence behavior can be analyzed against intent.
- Added `src/analysis/analyze_live_debug_confidence.py`.
  - Goal: compute compact confidence and prediction-distribution summaries that are directly useful for choosing an abstain/confidence threshold rule.
- Updated README usage docs for the manual two-step workflow.
  - Goal: keep the process repeatable when launcher automation is not the immediate priority.

### What changed

- Live debug CSV rows now include `intended_label` populated from the CLI value (empty when omitted).
- Live debug run summary JSON now also records the intended label used for the run.
- New confidence-analysis command produces:
  - total frames/inference frames,
  - raw + smoothed prediction counts,
  - overall and per-class confidence summaries,
  - frame counts above threshold buckets,
  - intended-label match vs non-match confidence stats,
  - top competing top-1 classes when intended label is provided.
- Analysis output is saved as `<csv_stem>_confidence_summary.json` for report-ready reuse.

### Current takeaway

We now have practical tooling to evaluate confidence behavior systematically per debug run, which is a concrete step toward a safe abstain/threshold policy for final game integration. The immediate next step is to run repeated intended-label sessions per gesture and compare retained-vs-abstained frame rates under candidate thresholds (for example 0.70, 0.80, 0.90).

## Live debug workflow hardening: stale JSON contamination fix

During manual live debugging, multiple runs were accidentally recorded into the same folder:
`data/raw/live_buffer/openpose_session/live_test`.
That introduced stale JSON contamination across runs.

### What was attempted and why

- Reviewed the manual 2-step live workflow (record with OpenPose, then replay with `live_openpose_debug`) to identify where cross-run state could leak.
- Prioritized workflow safety and traceability rather than classifier/model changes, because the observed failure mode was data-mixing, not a training/inference architecture bug.

### What was observed

- A later `defense_fire` replay reproduced the same later-frame predictions/confidences seen in an earlier `attack_earth` replay.
- The most likely cause was leftover frame JSON from the earlier run still present in the shared `live_test` folder.

### What changed

- Added a dedicated helper: `tools/live/new_live_capture_session.ps1`.
- The helper creates a fresh per-recording folder under `data/raw/live_buffer/openpose_session/` with naming:
  - `live_<intended-or-unknown>_<timestamp>`
- It also prints human-ready next commands for:
  - OpenPose recording into the new folder,
  - replay with `python -m src.inference.live_openpose_debug`,
  - confidence summary analysis command shape.
- Updated `README.md` to document the safer manual flow and explicitly explain stale-JSON contamination risk.

### Current takeaway

The reliable live debugging path remains manual, but each recording must use a new session folder.
Fresh per-run directories eliminate stale-file carryover and make replay/confidence interpretation trustworthy again.

## Live debug overlay stage: confidence-aware accept/abstain prototype

What was attempted and why:

- Moved the existing working live replay pipeline (`python -m src.inference.live_openpose_debug --json-dir ...`) into a true real-time debug stage by adding a lightweight on-screen terminal HUD and a confidence-aware decision layer.
- This was done because recent clean live tests (especially `defense_earth`) showed the pose-only MLP is strong enough to justify real-time trust diagnostics, while still needing conservative action gating for noisy boundaries (for example within fire-family classes).

What was observed beforehand:

- `idle` was already generally strong.
- `defense_earth` was notably strong/stable.
- `attack_earth` was detectable but less clean.
- `defense_fire` could be strong, but class-boundary ambiguity still appears in fire-family transitions.

What changed in this step:

- Added live overlay output fields for: raw label, smoothed label, top-1/top-2 confidence, margin, and decision state.
- Added configurable decision thresholds (`--accept-threshold`, `--margin-threshold`) and explicit `ACCEPT` vs `NO_ACTION` logic.
- Enforced non-idle-only acceptance: `idle` remains visible as prediction but does not produce action acceptance.
- Extended CSV logging with decision fields (`decision_label`, `decision_status`, `top1_margin`, threshold columns) while keeping prior log usefulness.
- Added launcher passthrough for threshold/overlay flags and README usage guidance for the new debug stage.

Current takeaway:

- This stage gives a deployment-prototype answer to the three critical live questions: what the model thinks now, how sure it is, and whether we trust it enough to act.
- The conservative accept/abstain policy is intentional because overreaction is more dangerous than underreaction in this game, and game-level cooldown/behavior logic can remain a later integration layer.

## Temporal trigger filtering layer: bridging frame decisions to gameplay actions

### Problem observed in recent live tests

The ACCEPT/NO_ACTION gate improved safety versus raw labels, but live behavior still showed a practical gameplay mismatch:
single-frame ACCEPT spikes can appear during noisy transitions, especially when gestures are changing or near class boundaries.

That means frame-level confidence gating alone is informative, but still too reactive for immediate action firing.

### Why frame-level ACCEPT is not enough

- `ACCEPT` answers: "Is this frame confident enough right now?"
- Gameplay needs: "Has this action stayed stable long enough to be intentional, and should we fire now?"

Without temporal filtering, one high-confidence frame can trigger too early.
Without suppression after firing, repeated near-identical frames can over-fire.

### What was changed

An additive temporal trigger layer was added on top of existing decision outputs in `live_openpose_debug`:

- New CLI flag: `--trigger-streak` (default `3`)
  - requires the same non-idle `ACCEPT` decision label for N consecutive inference frames before firing.
- New CLI flag: `--trigger-cooldown-frames` (default `15`)
  - after a trigger fires, suppresses new triggers for M inference frames.
- New final output state fields:
  - `final_action_status`: `TRIGGER` or `NO_TRIGGER`
  - `final_action_label`: triggered label or empty
- Streak reset behavior:
  - reset on `NO_ACTION`,
  - reset when decision label changes,
  - reset on idle/empty/no-action labels.
- Extended CSV and summary outputs with trigger-state and trigger-count metrics.
- Improved terminal overlay readability so each update prints cleanly as its own line (safer for PowerShell viewing).

### Expected benefit for game integration

This preserves full debug visibility while making output behavior much closer to game-ready action semantics:

- fewer false-positive action bursts from single-frame spikes,
- better intentionality via short temporal confirmation,
- reduced repeated-fire spam via cooldown latch,
- cleaner handoff path from debug inference logs to eventual Unity gameplay trigger integration.
