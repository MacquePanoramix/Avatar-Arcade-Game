# Misclassification Analysis

- Run directory: `D:\Documentos\Python Projects\Avatar-Arcade-Game\models\experiment_runs\20260412_115002\full_mlp`
- Total test samples: **69**
- Correct: **63**
- Incorrect: **6**
- Test accuracy from predictions: **0.9130**

## Data availability checks

- confidence values available: **yes**
  - valid `confidence_of_predicted_class`: 69/69
  - valid `confidence_of_true_class`: 69/69
- metadata traceability available: **yes**
  - valid `person`: 69/69
  - valid `session`: 69/69
  - valid `take`: 69/69
  - valid `original_sample_path`: 69/69

## Top confusion pairs

- `attack_fire` → `defense_fire`: 3
- `attack_air` → `attack_water`: 1
- `defense_earth` → `idle`: 1
- `idle` → `defense_earth`: 1

## Class-by-class mistake breakdown

- `attack_fire`: mistakes=3 / total=8 (rate=0.375)
- `attack_air`: mistakes=1 / total=7 (rate=0.143)
- `defense_earth`: mistakes=1 / total=8 (rate=0.125)
- `idle`: mistakes=1 / total=7 (rate=0.143)
- `attack_earth`: mistakes=0 / total=8 (rate=0.000)
- `attack_water`: mistakes=0 / total=8 (rate=0.000)
- `defense_air`: mistakes=0 / total=7 (rate=0.000)
- `defense_fire`: mistakes=0 / total=8 (rate=0.000)
- `defense_water`: mistakes=0 / total=8 (rate=0.000)

## Short interpretation notes

- Use `highest_confidence_errors.csv` to inspect likely label/representation mismatches.
- Use `hardest_correct_samples.csv` to inspect borderline-but-correct examples.
- Use metadata fields (`person`, `session`, `take`, `original_sample_path`) to check if confusions cluster by take.

## Precision / recall snapshot

- `attack_air`: precision=1.000, recall=0.857, support=7
- `attack_earth`: precision=1.000, recall=1.000, support=8
- `attack_fire`: precision=1.000, recall=0.625, support=8
- `attack_water`: precision=0.889, recall=1.000, support=8
- `defense_air`: precision=1.000, recall=1.000, support=7
- `defense_earth`: precision=0.875, recall=0.875, support=8
- `defense_fire`: precision=0.727, recall=1.000, support=8
- `defense_water`: precision=1.000, recall=1.000, support=8
- `idle`: precision=0.857, recall=0.857, support=7

## Example take references for top confusion pairs

- `attack_fire` → `defense_fire`:
  - person=rin, session=s20, take=take_001, original_sample_path=data\raw\openpose_json\attack_fire\rin\s20\take_001
  - person=rin, session=s23, take=take_001, original_sample_path=data\raw\openpose_json\attack_fire\rin\s23\take_001
  - person=p1, session=s1, take=take_001, original_sample_path=data\raw\openpose_json\attack_fire\p1\s1\take_001
- `attack_air` → `attack_water`:
  - person=rin, session=s22, take=take_001, original_sample_path=data\raw\openpose_json\attack_air\rin\s22\take_001
- `defense_earth` → `idle`:
  - person=rin, session=s20, take=take_001, original_sample_path=data\raw\openpose_json\defense_earth\rin\s20\take_001
- `idle` → `defense_earth`:
  - person=luis, session=s6, take=take_003, original_sample_path=data\raw\openpose_json\idle\luis\s6\take_003
