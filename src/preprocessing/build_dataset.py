"""Dataset builder placeholders for training inputs/labels."""

from typing import Tuple

import numpy as np


def build_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Return placeholder dataset arrays.

    TODO: Replace with real data-loading and preprocessing pipeline.
    """
    x = np.zeros((8, 30, 50), dtype=np.float32)
    y = np.zeros((8,), dtype=np.int32)
    return x, y
