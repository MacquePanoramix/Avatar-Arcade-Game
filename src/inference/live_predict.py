"""Real-time prediction scaffold using a sequence buffer and LSTM model."""

from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np
from tensorflow import keras


class LivePredictor:
    """Minimal live inference scaffold."""

    def __init__(self, model_path: Path, sequence_length: int = 30) -> None:
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)
        self.model: Optional[keras.Model] = None

    def load_model(self) -> None:
        """Load inference model from disk.

        TODO: Add robust model version checks and error handling.
        """
        if self.model_path.exists():
            self.model = keras.models.load_model(self.model_path)
        else:
            self.model = None

    def add_frame(self, features: np.ndarray) -> None:
        """Append one frame of pose features to the rolling buffer."""
        self.buffer.append(features)

    def predict(self) -> Optional[int]:
        """Predict the most likely class ID from the current buffer.

        TODO: Wire this up to real-time Kinect/OpenPose stream ingestion.
        """
        if self.model is None or len(self.buffer) < self.sequence_length:
            return None
        seq = np.stack(self.buffer, axis=0)[None, ...]
        probs = self.model.predict(seq, verbose=0)[0]
        return int(np.argmax(probs))
