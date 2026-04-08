"""Helpers to persist sequence arrays to disk."""

from pathlib import Path
from typing import Any

import joblib


def save_sequence(sequence: Any, output_path: Path) -> None:
    """Save a sequence artifact using joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(sequence, output_path)
