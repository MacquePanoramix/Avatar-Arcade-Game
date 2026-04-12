"""Runtime-safe preprocessing for live OpenPose JSON inference.

This module mirrors the training-time preprocessing intent while staying causal:
- no future-frame interpolation
- conservative bad-frame detection
- live-safe joint/frame repair fallbacks
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing.build_openpose_dataset import (
    CENTER_JUMP_SCALE_MULTIPLIER,
    DEFAULT_CONFIDENCE_CUTOFF,
    FEATURES_PER_FRAME,
    FULL_BODY25_JOINTS,
    HIP_WEIGHT,
    L_HIP_IDX,
    L_SHOULDER_IDX,
    MID_HIP_IDX,
    MIN_SCALE_EPS,
    MULTI_JOINT_JUMP_SCALE_MULTIPLIER,
    MULTI_JOINT_MIN_COUNT,
    NECK_IDX,
    NUM_JOINTS,
    R_HIP_IDX,
    R_SHOULDER_IDX,
    SAFE_FALLBACK_SCALE,
    SCALE_SMOOTH_ALPHA_NEW,
    SCALE_SMOOTH_ALPHA_OLD,
    SELECTED_BODY25_INDICES,
    SHOULDER_WEIGHT,
    STABLE_JOINTS,
    SYMMETRIC_JOINT_PAIRS,
    TORSO_WEIGHT,
)


@dataclass
class RuntimeFrameResult:
    """Processed frame and debug flags for live inference."""

    features_30: np.ndarray
    was_repaired_frame: bool
    had_joint_repair: bool
    suspicious_jump: bool
    used_prev_frame_copy: bool
    missing_joint_count: int


class RuntimePreprocessor:
    """Stateful causal preprocessor that emits one normalized frame at a time."""

    def __init__(
        self,
        confidence_cutoff: float = DEFAULT_CONFIDENCE_CUTOFF,
    ) -> None:
        self.confidence_cutoff = float(confidence_cutoff)
        self.counterpart_map = self._build_symmetric_counterpart_map()

        self.prev_valid_scale: float | None = None
        self.prev_smoothed_scale: float | None = None

        self.prev_accepted_raw_xy: np.ndarray | None = None
        self.prev_accepted_center: np.ndarray | None = None
        self.prev_accepted_usable: np.ndarray | None = None

        self.last_processed_frame_xy: np.ndarray | None = None

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _safe_distance(
        self,
        xy: np.ndarray,
        usable_mask: np.ndarray,
        a_idx: int,
        b_idx: int,
    ) -> float | None:
        if not usable_mask[a_idx] or not usable_mask[b_idx]:
            return None
        dist = float(np.linalg.norm(xy[a_idx] - xy[b_idx]))
        if dist <= 0:
            return None
        return dist

    def _choose_center(self, xy: np.ndarray, usable_mask: np.ndarray) -> np.ndarray | None:
        if usable_mask[NECK_IDX]:
            return xy[NECK_IDX]
        if usable_mask[MID_HIP_IDX]:
            return xy[MID_HIP_IDX]
        return None

    def _compute_weighted_scale(self, xy: np.ndarray, usable_mask: np.ndarray) -> float | None:
        candidates: list[tuple[float, float]] = []

        shoulder = self._safe_distance(xy, usable_mask, L_SHOULDER_IDX, R_SHOULDER_IDX)
        if shoulder is not None:
            candidates.append((shoulder, SHOULDER_WEIGHT))

        torso = self._safe_distance(xy, usable_mask, NECK_IDX, MID_HIP_IDX)
        if torso is not None:
            candidates.append((torso, TORSO_WEIGHT))

        hip = self._safe_distance(xy, usable_mask, L_HIP_IDX, R_HIP_IDX)
        if hip is not None:
            candidates.append((hip, HIP_WEIGHT))

        if not candidates:
            return None

        weighted_sum = sum(v * w for v, w in candidates)
        total_w = sum(w for _, w in candidates)
        return weighted_sum / total_w

    def _build_symmetric_counterpart_map(self) -> dict[int, int]:
        counterpart_map: dict[int, int] = {}
        for a_idx, b_idx in SYMMETRIC_JOINT_PAIRS:
            counterpart_map[a_idx] = b_idx
            counterpart_map[b_idx] = a_idx
        return counterpart_map

    def _is_suspicious_frame(
        self,
        current_xy: np.ndarray,
        current_center: np.ndarray,
        current_scale: float,
        current_usable: np.ndarray,
    ) -> bool:
        if (
            self.prev_accepted_raw_xy is None
            or self.prev_accepted_center is None
            or self.prev_accepted_usable is None
            or current_scale <= 0
        ):
            return False

        jump_threshold = MULTI_JOINT_JUMP_SCALE_MULTIPLIER * current_scale
        center_threshold = CENTER_JUMP_SCALE_MULTIPLIER * current_scale

        moved_count = 0
        for idx in STABLE_JOINTS:
            if current_usable[idx] and self.prev_accepted_usable[idx]:
                jump_dist = float(np.linalg.norm(current_xy[idx] - self.prev_accepted_raw_xy[idx]))
                if jump_dist > jump_threshold:
                    moved_count += 1

        if moved_count >= MULTI_JOINT_MIN_COUNT:
            return True

        center_jump = float(np.linalg.norm(current_center - self.prev_accepted_center))
        return center_jump > center_threshold

    def _parse_frame(self, frame_data: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
        people = frame_data.get("people", [])
        if not people:
            return None, None

        keypoints = people[0].get("pose_keypoints_2d", [])
        if len(keypoints) < FULL_BODY25_JOINTS * 3:
            return None, None

        arr = np.asarray(keypoints, dtype=np.float32).reshape(FULL_BODY25_JOINTS, 3)
        arr = arr[SELECTED_BODY25_INDICES, :]

        xy = arr[:, :2].copy()
        conf = arr[:, 2]
        usable_mask = conf >= self.confidence_cutoff
        return xy, usable_mask

    def _repair_missing_joints(
        self,
        normalized_xy: np.ndarray,
        usable_mask: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        repaired = normalized_xy.copy()
        missing_count = 0

        for joint_idx in range(NUM_JOINTS):
            if usable_mask[joint_idx]:
                continue
            missing_count += 1

            # 1) previous valid joint value
            if self.last_processed_frame_xy is not None:
                repaired[joint_idx] = self.last_processed_frame_xy[joint_idx]
                continue

            # 2) symmetric counterpart mirror if possible
            counterpart_idx = self.counterpart_map.get(joint_idx)
            if counterpart_idx is not None and usable_mask[counterpart_idx]:
                repaired[joint_idx, 0] = -normalized_xy[counterpart_idx, 0]
                repaired[joint_idx, 1] = normalized_xy[counterpart_idx, 1]
                continue

            # 3) zero fallback
            repaired[joint_idx] = 0.0

        return repaired, missing_count

    def _copy_last_processed(self) -> np.ndarray:
        if self.last_processed_frame_xy is None:
            return np.zeros((NUM_JOINTS, 2), dtype=np.float32)
        return self.last_processed_frame_xy.copy()

    def process_json_path(self, frame_json_path: Path) -> RuntimeFrameResult:
        """Process one OpenPose JSON file into a normalized 30-d frame."""
        frame_data = self._load_json(frame_json_path)
        xy, usable_mask = self._parse_frame(frame_data)

        # Whole-frame failure -> causal copy-forward fallback.
        if xy is None or usable_mask is None:
            copied = self._copy_last_processed()
            features = copied.reshape(FEATURES_PER_FRAME).astype(np.float32)
            return RuntimeFrameResult(
                features_30=features,
                was_repaired_frame=True,
                had_joint_repair=False,
                suspicious_jump=False,
                used_prev_frame_copy=True,
                missing_joint_count=NUM_JOINTS,
            )

        center = self._choose_center(xy, usable_mask)
        if center is None:
            copied = self._copy_last_processed()
            features = copied.reshape(FEATURES_PER_FRAME).astype(np.float32)
            return RuntimeFrameResult(
                features_30=features,
                was_repaired_frame=True,
                had_joint_repair=False,
                suspicious_jump=False,
                used_prev_frame_copy=True,
                missing_joint_count=NUM_JOINTS,
            )

        computed_scale = self._compute_weighted_scale(xy, usable_mask)
        if computed_scale is None:
            current_scale = self.prev_valid_scale if self.prev_valid_scale is not None else SAFE_FALLBACK_SCALE
        else:
            current_scale = computed_scale

        current_scale = max(float(current_scale), MIN_SCALE_EPS)

        suspicious = self._is_suspicious_frame(
            current_xy=xy,
            current_center=center,
            current_scale=current_scale,
            current_usable=usable_mask,
        )
        if suspicious:
            copied = self._copy_last_processed()
            features = copied.reshape(FEATURES_PER_FRAME).astype(np.float32)
            return RuntimeFrameResult(
                features_30=features,
                was_repaired_frame=True,
                had_joint_repair=False,
                suspicious_jump=True,
                used_prev_frame_copy=True,
                missing_joint_count=NUM_JOINTS,
            )

        if self.prev_smoothed_scale is None:
            smoothed_scale = current_scale
        else:
            smoothed_scale = (
                SCALE_SMOOTH_ALPHA_OLD * self.prev_smoothed_scale
                + SCALE_SMOOTH_ALPHA_NEW * current_scale
            )
        smoothed_scale = max(float(smoothed_scale), MIN_SCALE_EPS)

        normalized = (xy - center[None, :]) / smoothed_scale
        repaired, missing_joint_count = self._repair_missing_joints(normalized, usable_mask)

        # Update accepted-frame state only after a non-suspicious parse.
        self.prev_valid_scale = current_scale
        self.prev_smoothed_scale = smoothed_scale
        self.prev_accepted_raw_xy = xy
        self.prev_accepted_center = center
        self.prev_accepted_usable = usable_mask
        self.last_processed_frame_xy = repaired.copy()

        return RuntimeFrameResult(
            features_30=repaired.reshape(FEATURES_PER_FRAME).astype(np.float32),
            was_repaired_frame=missing_joint_count > 0,
            had_joint_repair=missing_joint_count > 0,
            suspicious_jump=False,
            used_prev_frame_copy=False,
            missing_joint_count=missing_joint_count,
        )
