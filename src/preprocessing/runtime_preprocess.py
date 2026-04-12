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
from typing import Any, Literal

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
    tracking_mode: str
    detected_people_count: int
    selected_person_index: int | None
    selected_left_person_index: int | None = None
    selected_right_person_index: int | None = None
    tracking_note: str = ""


TrackingMode = Literal["single_person", "two_player_left_right"]


@dataclass
class ParsedPersonCandidate:
    """Parsed OpenPose person candidate with lightweight quality signals."""

    person_index: int
    xy: np.ndarray
    usable_mask: np.ndarray
    usable_joint_count: int
    confidence_mean: float
    center: np.ndarray | None
    scale: float | None


@dataclass
class TrackedPersonState:
    """Small persistent state used for runtime identity continuity."""

    center: np.ndarray | None = None
    scale: float | None = None
    usable_joint_count: int = 0


class RuntimePreprocessor:
    """Stateful causal preprocessor that emits one normalized frame at a time."""

    def __init__(
        self,
        confidence_cutoff: float = DEFAULT_CONFIDENCE_CUTOFF,
        tracking_mode: TrackingMode = "single_person",
    ) -> None:
        self.confidence_cutoff = float(confidence_cutoff)
        if tracking_mode not in {"single_person", "two_player_left_right"}:
            raise ValueError(f"Unsupported tracking mode: {tracking_mode}")
        self.tracking_mode: TrackingMode = tracking_mode
        self.counterpart_map = self._build_symmetric_counterpart_map()

        self.prev_valid_scale: float | None = None
        self.prev_smoothed_scale: float | None = None

        self.prev_accepted_raw_xy: np.ndarray | None = None
        self.prev_accepted_center: np.ndarray | None = None
        self.prev_accepted_usable: np.ndarray | None = None

        self.last_processed_frame_xy: np.ndarray | None = None
        self.single_track = TrackedPersonState()
        self.left_track = TrackedPersonState()
        self.right_track = TrackedPersonState()

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

    def _parse_people(self, frame_data: dict[str, Any]) -> list[ParsedPersonCandidate]:
        people = frame_data.get("people", [])
        parsed: list[ParsedPersonCandidate] = []
        for person_index, person in enumerate(people):
            keypoints = person.get("pose_keypoints_2d", [])
            if len(keypoints) < FULL_BODY25_JOINTS * 3:
                continue

            arr = np.asarray(keypoints, dtype=np.float32).reshape(FULL_BODY25_JOINTS, 3)
            arr = arr[SELECTED_BODY25_INDICES, :]
            xy = arr[:, :2].copy()
            conf = arr[:, 2]
            usable_mask = conf >= self.confidence_cutoff
            usable_joint_count = int(np.count_nonzero(usable_mask))
            usable_conf = conf[usable_mask]
            confidence_mean = float(np.mean(usable_conf)) if usable_conf.size > 0 else 0.0
            center = self._choose_center(xy, usable_mask)
            scale = self._compute_weighted_scale(xy, usable_mask)
            parsed.append(
                ParsedPersonCandidate(
                    person_index=person_index,
                    xy=xy,
                    usable_mask=usable_mask,
                    usable_joint_count=usable_joint_count,
                    confidence_mean=confidence_mean,
                    center=center,
                    scale=scale,
                )
            )
        return parsed

    def _candidate_quality_score(self, candidate: ParsedPersonCandidate) -> float:
        return float(candidate.usable_joint_count) + (0.2 * candidate.confidence_mean)

    def _assignment_cost(
        self,
        track: TrackedPersonState,
        candidate: ParsedPersonCandidate,
    ) -> float:
        if candidate.center is None:
            return 1e6
        if track.center is None:
            return -self._candidate_quality_score(candidate)

        track_scale = max(float(track.scale or SAFE_FALLBACK_SCALE), MIN_SCALE_EPS)
        candidate_scale = max(float(candidate.scale or track_scale), MIN_SCALE_EPS)
        center_dist = float(np.linalg.norm(candidate.center - track.center))
        normalized_dist = center_dist / track_scale
        scale_delta = abs(np.log(candidate_scale / track_scale))
        quality_term = -0.03 * float(candidate.usable_joint_count)
        return normalized_dist + (0.35 * scale_delta) + quality_term

    def _update_track_state(self, track: TrackedPersonState, candidate: ParsedPersonCandidate) -> None:
        track.center = None if candidate.center is None else candidate.center.copy()
        if candidate.scale is not None:
            track.scale = float(max(candidate.scale, MIN_SCALE_EPS))
        track.usable_joint_count = int(candidate.usable_joint_count)

    def _pick_single_person(
        self,
        candidates: list[ParsedPersonCandidate],
    ) -> tuple[ParsedPersonCandidate | None, str]:
        if not candidates:
            return None, "no_people_detected"
        if self.single_track.center is None:
            best = max(candidates, key=self._candidate_quality_score)
            return best, "single_init_by_quality"
        best = min(candidates, key=lambda c: self._assignment_cost(self.single_track, c))
        return best, "single_temporal_match"

    def _pick_two_player(
        self,
        candidates: list[ParsedPersonCandidate],
    ) -> tuple[ParsedPersonCandidate | None, ParsedPersonCandidate | None, str]:
        if not candidates:
            return None, None, "no_people_detected"
        centered = [c for c in candidates if c.center is not None]
        if not centered:
            return None, None, "no_center_candidates"

        left_unset = self.left_track.center is None
        right_unset = self.right_track.center is None
        if left_unset and right_unset:
            best_two = sorted(centered, key=self._candidate_quality_score, reverse=True)[:2]
            best_two = sorted(best_two, key=lambda c: float(c.center[0]))
            if len(best_two) == 1:
                return best_two[0], None, "two_player_init_single_seen"
            return best_two[0], best_two[1], "two_player_init_x_order"

        best_left: ParsedPersonCandidate | None = None
        best_right: ParsedPersonCandidate | None = None
        best_score = float("inf")
        missing_penalty = 1.5
        cross_penalty = 0.25

        options: list[tuple[int | None, int | None]] = [(None, None)]
        for idx in range(len(centered)):
            options.append((idx, None))
            options.append((None, idx))
        for left_idx in range(len(centered)):
            for right_idx in range(len(centered)):
                if left_idx == right_idx:
                    continue
                options.append((left_idx, right_idx))

        for left_idx, right_idx in options:
            score = 0.0
            cand_left = centered[left_idx] if left_idx is not None else None
            cand_right = centered[right_idx] if right_idx is not None else None

            if cand_left is None:
                score += missing_penalty
            else:
                score += self._assignment_cost(self.left_track, cand_left)
            if cand_right is None:
                score += missing_penalty
            else:
                score += self._assignment_cost(self.right_track, cand_right)

            if cand_left is not None and cand_right is not None and cand_left.center[0] > cand_right.center[0]:
                score += cross_penalty

            if score < best_score:
                best_score = score
                best_left = cand_left
                best_right = cand_right

        return best_left, best_right, "two_player_temporal_match"

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
        candidates = self._parse_people(frame_data)
        detected_people_count = len(candidates)
        selected_person_index: int | None = None
        selected_left_person_index: int | None = None
        selected_right_person_index: int | None = None

        selected_candidate: ParsedPersonCandidate | None
        tracking_note: str
        selected_left_candidate: ParsedPersonCandidate | None = None
        selected_right_candidate: ParsedPersonCandidate | None = None

        if self.tracking_mode == "two_player_left_right":
            left_candidate, right_candidate, tracking_note = self._pick_two_player(candidates)
            if left_candidate is not None:
                selected_left_person_index = left_candidate.person_index
                selected_left_candidate = left_candidate
            if right_candidate is not None:
                selected_right_person_index = right_candidate.person_index
                selected_right_candidate = right_candidate

            # Current debug classifier remains single-stream: prefer left track, then right.
            selected_candidate = left_candidate if left_candidate is not None else right_candidate
            selected_person_index = (
                selected_candidate.person_index if selected_candidate is not None else None
            )
        else:
            selected_candidate, tracking_note = self._pick_single_person(candidates)
            if selected_candidate is not None:
                selected_person_index = selected_candidate.person_index
                selected_left_person_index = selected_candidate.person_index

        xy = None if selected_candidate is None else selected_candidate.xy
        usable_mask = None if selected_candidate is None else selected_candidate.usable_mask

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
                tracking_mode=self.tracking_mode,
                detected_people_count=detected_people_count,
                selected_person_index=selected_person_index,
                selected_left_person_index=selected_left_person_index,
                selected_right_person_index=selected_right_person_index,
                tracking_note=f"{tracking_note}|frame_copy_no_selection",
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
                tracking_mode=self.tracking_mode,
                detected_people_count=detected_people_count,
                selected_person_index=selected_person_index,
                selected_left_person_index=selected_left_person_index,
                selected_right_person_index=selected_right_person_index,
                tracking_note=f"{tracking_note}|frame_copy_no_center",
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
                tracking_mode=self.tracking_mode,
                detected_people_count=detected_people_count,
                selected_person_index=selected_person_index,
                selected_left_person_index=selected_left_person_index,
                selected_right_person_index=selected_right_person_index,
                tracking_note=f"{tracking_note}|frame_copy_suspicious",
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
        if self.tracking_mode == "two_player_left_right":
            if selected_left_candidate is not None:
                self._update_track_state(self.left_track, selected_left_candidate)
            if selected_right_candidate is not None:
                self._update_track_state(self.right_track, selected_right_candidate)
        elif selected_candidate is not None:
            self._update_track_state(self.single_track, selected_candidate)

        return RuntimeFrameResult(
            features_30=repaired.reshape(FEATURES_PER_FRAME).astype(np.float32),
            was_repaired_frame=missing_joint_count > 0,
            had_joint_repair=missing_joint_count > 0,
            suspicious_jump=False,
            used_prev_frame_copy=False,
            missing_joint_count=missing_joint_count,
            tracking_mode=self.tracking_mode,
            detected_people_count=detected_people_count,
            selected_person_index=selected_person_index,
            selected_left_person_index=selected_left_person_index,
            selected_right_person_index=selected_right_person_index,
            tracking_note=tracking_note,
        )
