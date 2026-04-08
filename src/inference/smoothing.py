"""Prediction smoothing utilities."""

from collections import Counter, deque
from typing import Deque, Optional


class PredictionSmoother:
    """Simple majority-vote smoothing over recent predictions."""

    def __init__(self, window_size: int = 5) -> None:
        self.window: Deque[int] = deque(maxlen=window_size)

    def update(self, label_id: int) -> Optional[int]:
        """Add one prediction and return smoothed result."""
        self.window.append(label_id)
        if not self.window:
            return None
        return Counter(self.window).most_common(1)[0][0]
