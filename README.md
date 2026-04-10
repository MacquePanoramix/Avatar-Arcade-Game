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
