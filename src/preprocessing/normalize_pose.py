"""Pose normalization helpers."""

import numpy as np


def normalize_pose(sequence: np.ndarray) -> np.ndarray:
    """Normalize sequence values with a simple z-score fallback."""
    if sequence.size == 0:
        return sequence
    mean = float(sequence.mean())
    std = float(sequence.std()) or 1.0
    return (sequence - mean) / std
