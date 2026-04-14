# Take Review Summary

- Run directory: `D:\Documentos\Python Projects\Avatar-Arcade-Game\models\experiment_runs\20260412_115002\full_mlp`
- Total misclassified samples found: **6**
- Reviewed confusion pairs: **4**
- Review images generated: **5**

## Top confusion pairs (by count)

- `attack_fire` → `defense_fire`: 3
- `attack_air` → `attack_water`: 1
- `defense_earth` → `idle`: 1
- `idle` → `defense_earth`: 1

## Reviewed cases

- Case 001: `attack_fire` → `defense_fire` | image: `review_case_001_attack_fire_vs_defense_fire.png` | misclassified take: `data\raw\openpose_json\attack_fire\rin\s23\take_001`
- Case 002: `attack_fire` → `defense_fire` | image: `review_case_002_attack_fire_vs_defense_fire.png` | misclassified take: `data\raw\openpose_json\attack_fire\rin\s20\take_001`
- Case 003: `attack_air` → `attack_water` | image: `review_case_003_attack_air_vs_attack_water.png` | misclassified take: `data\raw\openpose_json\attack_air\rin\s22\take_001`
- Case 004: `defense_earth` → `idle` | image: `review_case_004_defense_earth_vs_idle.png` | misclassified take: `data\raw\openpose_json\defense_earth\rin\s20\take_001`
- Case 005: `idle` → `defense_earth` | image: `review_case_005_idle_vs_defense_earth.png` | misclassified take: `data\raw\openpose_json\idle\luis\s6\take_003`

## Short interpretation notes

- Compare each misclassified take against strong correct examples from both the true and confusing classes.
- Borderline correct references help identify whether the error looks like an ambiguous/weak execution.
- Use person/session/take/path metadata shown in image headers for traceable follow-up.
