"""Shared temporal resampling policy for dataset build and live inference."""

from __future__ import annotations

import numpy as np

TARGET_FPS = 10.0
SOURCE_NOMINAL_FPS = 15.0
TARGET_SEQUENCE_LENGTH = 24
PRE_CONTEXT_RATIO = 0.25
POST_CONTEXT_RATIO = 0.25


def source_window_frames_for_target_span(
    *,
    target_sequence_length: int = TARGET_SEQUENCE_LENGTH,
    source_nominal_fps: float = SOURCE_NOMINAL_FPS,
    target_fps: float = TARGET_FPS,
) -> int:
    """Convert target sequence span to source-frame span at nominal source FPS."""
    if target_sequence_length < 1:
        raise ValueError("target_sequence_length must be >= 1.")
    if source_nominal_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS values must be positive.")
    return max(1, int(round(target_sequence_length * (source_nominal_fps / target_fps))))


def resample_sequence_fixed_length(
    frames: list[np.ndarray] | np.ndarray,
    target_sequence_length: int = TARGET_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Deterministically resample any sequence to fixed length via index sampling."""
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D frame array [T, F], got shape {arr.shape}.")
    if target_sequence_length < 1:
        raise ValueError("target_sequence_length must be >= 1.")
    if arr.shape[0] == 0:
        raise ValueError("Cannot resample an empty frame sequence.")

    source_len = arr.shape[0]
    if source_len == target_sequence_length:
        return arr.copy()

    idx = np.linspace(0, source_len - 1, num=target_sequence_length)
    idx = np.clip(np.rint(idx).astype(np.int32), 0, source_len - 1)
    return arr[idx]


def crop_with_active_context(
    frames: list[np.ndarray] | np.ndarray,
    *,
    active_start_frame: int,
    active_end_frame: int,
    pre_context_ratio: float = PRE_CONTEXT_RATIO,
    post_context_ratio: float = POST_CONTEXT_RATIO,
) -> np.ndarray:
    """Crop around active span with ratio-based context; frame indices are inclusive."""
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D frame array [T, F], got shape {arr.shape}.")
    n = int(arr.shape[0])
    if n == 0:
        raise ValueError("Cannot crop an empty frame sequence.")

    start = int(np.clip(active_start_frame, 0, n - 1))
    end = int(np.clip(active_end_frame, 0, n - 1))
    if end < start:
        start, end = end, start

    active_len = (end - start) + 1
    pre_frames = max(0, int(round(active_len * pre_context_ratio)))
    post_frames = max(0, int(round(active_len * post_context_ratio)))

    crop_start = max(0, start - pre_frames)
    crop_end_exclusive = min(n, end + 1 + post_frames)
    return arr[crop_start:crop_end_exclusive]
