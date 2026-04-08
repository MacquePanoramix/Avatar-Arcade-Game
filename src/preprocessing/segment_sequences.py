"""Sequence segmentation helpers."""

from typing import Iterable

import numpy as np


def segment_fixed_windows(frames: np.ndarray, window_size: int) -> Iterable[np.ndarray]:
    """Yield non-overlapping fixed windows from frame data."""
    for start in range(0, len(frames), window_size):
        window = frames[start : start + window_size]
        if len(window) == window_size:
            yield window
