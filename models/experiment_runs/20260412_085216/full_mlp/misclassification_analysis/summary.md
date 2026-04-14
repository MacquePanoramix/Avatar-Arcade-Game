# Misclassification Analysis

- Run directory: `D:\Documentos\Python Projects\Avatar-Arcade-Game\models\experiment_runs\20260412_085216\full_mlp`
- Total test samples: **69**
- Correct: **61**
- Incorrect: **8**
- Test accuracy from predictions: **0.8841**

## Data availability checks

- confidence values available: **no**
  - valid `confidence_of_predicted_class`: 0/69
  - valid `confidence_of_true_class`: 0/69
- metadata traceability available: **yes**
  - valid `person`: 69/69
  - valid `session`: 69/69
  - valid `take`: 69/69
  - valid `original_sample_path`: 69/69
- Missing confidence values mean this run may need re-training to export richer prediction probabilities.

## Top confusion pairs

- `attack_fire` → `defense_fire`: 3
- `defense_fire` → `attack_fire`: 2
- `attack_air` → `attack_water`: 1
- `defense_earth` → `idle`: 1
- `idle` → `defense_earth`: 1

## Class-by-class mistake breakdown

- `attack_fire`: mistakes=3 / total=8 (rate=0.375)
- `defense_fire`: mistakes=2 / total=8 (rate=0.250)
- `attack_air`: mistakes=1 / total=7 (rate=0.143)
- `defense_earth`: mistakes=1 / total=8 (rate=0.125)
- `idle`: mistakes=1 / total=7 (rate=0.143)
- `attack_earth`: mistakes=0 / total=8 (rate=0.000)
- `attack_water`: mistakes=0 / total=8 (rate=0.000)
- `defense_air`: mistakes=0 / total=7 (rate=0.000)
- `defense_water`: mistakes=0 / total=8 (rate=0.000)

## Short interpretation notes

- Use `highest_confidence_errors.csv` to inspect likely label/representation mismatches.
- Use `hardest_correct_samples.csv` to inspect borderline-but-correct examples.
- Use metadata fields (`person`, `session`, `take`, `original_sample_path`) to check if confusions cluster by take.

## Precision / recall snapshot

- `attack_air`: precision=1.000, recall=0.857, support=7
- `attack_earth`: precision=1.000, recall=1.000, support=8
- `attack_fire`: precision=0.714, recall=0.625, support=8
- `attack_water`: precision=0.889, recall=1.000, support=8
- `defense_air`: precision=1.000, recall=1.000, support=7
- `defense_earth`: precision=0.875, recall=0.875, support=8
- `defense_fire`: precision=0.667, recall=0.750, support=8
- `defense_water`: precision=1.000, recall=1.000, support=8
- `idle`: precision=0.857, recall=0.857, support=7

## Example take references for top confusion pairs

- `attack_fire` → `defense_fire`:
  - person=luis, session=s2, take=take_001, original_sample_path=data\raw\openpose_json\attack_air\luis\s2\take_001
  - person=luis, session=s5, take=take_001, original_sample_path=data\raw\openpose_json\attack_air\luis\s5\take_001
  - person=rin, session=s15, take=take_001, original_sample_path=data\raw\openpose_json\attack_air\rin\s15\take_001
- `defense_fire` → `attack_fire`:
  - person=rin, session=s19, take=take_001, original_sample_path=data\raw\openpose_json\attack_air\rin\s19\take_001
  - person=luis, session=s12, take=take_001, original_sample_path=data\raw\openpose_json\attack_earth\luis\s12\take_001
- `attack_air` → `attack_water`:
  - person=luis, session=s4, take=take_001, original_sample_path=data\raw\openpose_json\attack_air\luis\s4\take_001
- `defense_earth` → `idle`:
  - person=luis, session=s15, take=take_001, original_sample_path=data\raw\openpose_json\attack_earth\luis\s15\take_001
- `idle` → `defense_earth`:
  - person=p1, session=s1, take=take_001, original_sample_path=data\raw\openpose_json\attack_air\p1\s1\take_001
