"""Shared message schema helpers for Unity bridge payloads."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class PredictionMessage:
    """Minimal prediction payload schema."""

    gesture_id: int
    confidence: float
    timestamp_utc: str


def build_prediction_message(gesture_id: int, confidence: float) -> dict:
    """Build a serializable prediction message dict."""
    msg = PredictionMessage(
        gesture_id=gesture_id,
        confidence=confidence,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    return asdict(msg)
