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
