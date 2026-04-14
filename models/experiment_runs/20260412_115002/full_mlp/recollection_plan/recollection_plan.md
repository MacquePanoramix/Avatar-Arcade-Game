# Recollection & Curation Plan

## Current conclusion

- The current best model conclusion is unchanged: **pose-only full_mlp remains strong**.
- Remaining errors are concentrated into a small number of boundaries, not broad random failure.
- This phase is intentionally a **dataset-refinement planning step** (not a model-training step).

## Priority boundaries

Boundary prioritization score combines: confusion count, confidence of wrong predictions, borderline-correct proximity, and visual take-review signals.

| Rank | Boundary | Score | Confusions | High-conf errors | Borderline correct | Visual cases |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `attack_fire vs defense_fire` | 22.69 | 3 | 3 | 12 | 2 |
| 2 | `defense_earth vs idle` | 18.57 | 2 | 2 | 10 | 2 |
| 3 | `attack_air vs attack_water` | 6.15 | 1 | 1 | 1 | 1 |
| 4 | `attack_earth vs defense_earth` | 4.25 | 0 | 0 | 5 | 0 |
| 5 | `defense_fire vs defense_water` | 4.25 | 0 | 0 | 5 | 0 |

## Flagged takes to review

- Total priority review rows: **7** (see `priority_review_takes.csv`).
- Rows include high-confidence errors, borderline-correct near-boundary samples, and visual-review-emphasized takes.

## Proposed recollection targets

- Targets are intentionally modest and focused (see `priority_recollect_targets.csv`).
- Start with top-ranked boundaries before collecting broader new data.

| Priority | Focus boundary | Class to recollect | Contrast class | New takes |
|---:|---|---|---|---:|
| 1 | `attack_fire vs defense_fire` | `attack_fire` | `defense_fire` | 10 |
| 1 | `attack_fire vs defense_fire` | `defense_fire` | `attack_fire` | 10 |
| 2 | `defense_earth vs idle` | `defense_earth` | `idle` | 8 |
| 2 | `defense_earth vs idle` | `idle` | `defense_earth` | 8 |
| 3 | `attack_air vs attack_water` | `attack_air` | `attack_water` | 6 |
| 3 | `attack_air vs attack_water` | `attack_water` | `attack_air` | 6 |
| 4 | `attack_earth vs defense_earth` | `attack_earth` | `defense_earth` | 6 |
| 4 | `attack_earth vs defense_earth` | `defense_earth` | `attack_earth` | 6 |
| 5 | `defense_fire vs defense_water` | `defense_fire` | `defense_water` | 6 |
| 5 | `defense_fire vs defense_water` | `defense_water` | `defense_fire` | 6 |

## Performer instruction notes

### attack_fire vs defense_fire
- attack_fire: use a committed, directional strike shape (clear forward intent, visible extension).
- defense_fire: use a guarded, held blocking shape (stable posture, reduced forward reach).
- Record short instruction reminders before each take so performers keep class intent distinct.

### defense_earth vs idle
- idle: stay neutral, relaxed, and non-elemental (avoid defensive arm structure).
- defense_earth: emphasize a grounded defensive frame (clear brace/guard, not relaxed standing).
- Use a 1-2 second neutral reset between takes to reduce drift from previous gestures.

### attack_air vs attack_water
- Clarify visual intent between attack_air and attack_water with one class-specific cue per take.
- Coach performers to exaggerate the discriminative pose in the first second of execution.

## Suggested curation workflow

1. Manually inspect the top flagged rows in `priority_review_takes.csv`.
2. For each flagged take, decide: keep as-is, relabel candidate, or exclude candidate for future training sets.
3. Track any repeated performer/session clusters and prioritize those for corrective recollection.
4. Keep a simple curation log (decision + reason) before modifying any dataset files.
5. Re-run analysis after curation/recollection to verify boundary separation improved.

## Next recommended action

- Review top-ranked flagged takes first, then recollect a small batch for the top 1-2 boundaries.
- Output directory: `recollection_plan` under run `D:\Documentos\Python Projects\Avatar-Arcade-Game\models\experiment_runs\20260412_115002\full_mlp`.
