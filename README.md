# Avatar Arcade Game (ML Pipeline Scaffold)

This repository contains the **Python / machine-learning side** of a gesture-controlled elemental combat game.  
The Unity game client is expected to live in a separate repository.

## Project Goals

- Build a clean end-to-end ML pipeline for gesture recognition:
  1. capture
  2. preprocessing
  3. training
  4. inference
  5. Unity bridge
- Use an **LSTM** classifier over pose/skeleton sequence data.
- Keep gesture scope configurable for **9 / 8 / 6 / 4** gesture modes (default first classifier: 8 gestures + idle).
- Keep architecture ready for either:
  - direct Kinect skeleton streams (future)
  - OpenPose JSON input fallback

---

## Folder Overview

```text
configs/      YAML configuration for paths, gesture sets, and training settings
data/         Raw and processed ML data (mostly ignored by Git)
models/       Trained checkpoints, exported artifacts, and reports
logs/         Capture, training, and inference logs
notebooks/    Exploratory analysis notebooks
src/          Python source code for capture -> preprocessing -> training -> inference -> bridge
tests/        Pytest test suite
```

### Main Source Packages

- `src/capture/`: data collection and import stubs (Kinect/OpenPose)
- `src/preprocessing/`: normalization, segmentation, dataset assembly, label maps
- `src/training/`: LSTM training/evaluation/export scaffolds
- `src/inference/`: live prediction flow and output gating
- `src/bridge/`: Unity communication stubs (UDP placeholder)
- `src/utils/`: shared helpers (paths, logging, seeding, joints)

---

## PyCharm Setup

1. Open PyCharm.
2. Select **Open** and choose this repository root.
3. Go to **Settings > Project: Avatar-Arcade-Game > Python Interpreter**.
4. Create a new virtual environment (recommended `.venv` at repo root).
5. Open the built-in terminal and install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
6. Mark `src/` as a **Sources Root** in PyCharm (Right-click `src` -> Mark Directory as -> Sources Root).
7. (Optional) Configure a pytest run configuration targeting `tests/`.

---

## Environment Setup (CLI)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` if you want local environment customization.

---

## Configuration Files

- `configs/config.yaml`: top-level pipeline settings (capture/training/inference)
- `configs/gestures.yaml`: gesture mode sets and target mode
- `configs/paths.yaml`: centralized project-relative paths

---

## Development Roadmap (Scaffold-First)

1. Finalize data schema for skeleton sequence format.
2. Implement OpenPose JSON import and dataset build path.
3. Add Kinect capture implementation once dependency strategy is stable.
4. Implement full training loop with metrics/reporting.
5. Add robust real-time inference and Unity bridge protocol.
6. Add model versioning and experiment tracking.

---

## OpenPose Preprocessing v1 (Dataset Build Only)

This project now includes a preprocessing entry point that converts raw OpenPose JSON captures into fixed-shape training tensors.

### Run preprocessing

```bash
python -m src.preprocessing.build_openpose_dataset
```

Optional inspection mode:

```bash
python -m src.preprocessing.build_openpose_dataset --inspect-index 0
```

### Input structure

Preprocessing reads takes from:

```text
data/raw/openpose_json/<gesture>/<person>/<session>/<take>/
```

Each take folder is treated as one sample.

### Produced files

The command writes:

- `data/processed/X.npy` with shape `(N, 90, 30)`
- `data/processed/y.npy` with shape `(N,)`
- `data/processed/metadata.csv` (one row per sample)
- `data/processed/label_map.json` (target mode + label mappings)

### v1 normalization / repair choices

- Uses an upper-body BODY_25 subset (15 joints), `x,y` only.
- Uses `Neck` as center anchor with `MidHip` fallback.
- Uses robust weighted body scale from shoulder width / torso length / hip width.
- Smooths accepted frame scale over time (`0.8 * prev + 0.2 * current`).
- Marks likely identity-switch/catastrophic jumps as bad frames.
- Repairs bad frames by copy-forward (and future-fill for bad leading frames).
- If a whole take has no valid frames, fills the sample with zeros and records that in metadata.

## Baseline training runs (LSTM + MLP)

Run the normal full-dataset **LSTM baseline** (default):

```bash
python -m src.training.train_lstm --model-type lstm
```

Run the normal full-dataset **MLP baseline**:

```bash
python -m src.training.train_lstm --model-type mlp
```

MLP note:

- Each sample is flattened from `(90, 30)` into `2700` input features before classification.
- The script reuses the same saved split files in `data/splits/*.npy` (when present) so LSTM vs MLP comparisons stay fair.

Expected processed inputs (must already exist):

- `data/processed/X.npy`
- `data/processed/y.npy`
- `data/processed/metadata.csv`
- `data/processed/label_map.json`

Main outputs are saved to:

- LSTM checkpoint: `models/checkpoints/best_lstm.keras`
- MLP checkpoint: `models/checkpoints/best_mlp.keras`
- LSTM reports: `models/reports/training_history.csv`, `models/reports/training_history.png`, `models/reports/classification_report.txt`, `models/reports/confusion_matrix.csv`, `models/reports/test_predictions.csv`
- MLP reports: `models/reports/mlp_training_history.csv`, `models/reports/mlp_training_history.png`, `models/reports/mlp_classification_report.txt`, `models/reports/mlp_confusion_matrix.csv`, `models/reports/mlp_test_predictions.csv`
- `data/splits/train_indices.npy`
- `data/splits/val_indices.npy`
- `data/splits/test_indices.npy`
