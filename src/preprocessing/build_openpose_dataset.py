"""Build processed OpenPose dataset tensors for the first (preprocessing-only) pipeline.

This module intentionally focuses on:
1) Reading raw OpenPose JSON takes
2) Applying deterministic v1 preprocessing rules
3) Saving fixed-shape NumPy arrays + metadata

It does NOT perform model training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.preprocessing.label_map import get_active_labels
from src.utils.paths import load_paths_config, resolve_path

# -----------------------------
# Locked schema / preprocessing constants (v1)
# -----------------------------
SEQUENCE_LENGTH = 90
NUM_JOINTS = 25
NUM_COORDS = 2  # x, y only for model v1
FEATURES_PER_FRAME = NUM_JOINTS * NUM_COORDS
METADATA_COLUMNS = [
    "gesture",
    "person",
    "session",
    "take",
    "sample_path",
    "num_raw_frames",
    "num_bad_frames",
    "used_future_fill_for_start",
    "was_all_zero_sample",
]

# BODY_25 key joint indices (OpenPose order)
NECK_IDX = 1
R_SHOULDER_IDX = 2
L_SHOULDER_IDX = 5
MID_HIP_IDX = 8
R_HIP_IDX = 9
L_HIP_IDX = 12

# Confidence handling (weak filter)
DEFAULT_CONFIDENCE_CUTOFF = 0.05

# Weighted robust scale candidates
SHOULDER_WEIGHT = 0.50
TORSO_WEIGHT = 0.35
HIP_WEIGHT = 0.15

# Safety / smoothing constants
MIN_SCALE_EPS = 1e-3
SAFE_FALLBACK_SCALE = 100.0
SCALE_SMOOTH_ALPHA_OLD = 0.8
SCALE_SMOOTH_ALPHA_NEW = 0.2

# Suspicious frame thresholds
STABLE_JOINTS = [NECK_IDX, L_SHOULDER_IDX, R_SHOULDER_IDX, MID_HIP_IDX]
MULTI_JOINT_JUMP_SCALE_MULTIPLIER = 1.25
CENTER_JUMP_SCALE_MULTIPLIER = 1.50
MULTI_JOINT_MIN_COUNT = 2


@dataclass
class ParsedFrame:
    """Container for raw frame data before repair/final flattening."""

    xy: np.ndarray | None
    usable_mask: np.ndarray | None
    center_xy: np.ndarray | None
    current_scale: float | None
    is_bad: bool


@dataclass
class SampleResult:
    """Processed output for one take folder."""

    sample_90x50: np.ndarray
    num_raw_frames: int
    num_bad_frames: int
    used_future_fill_for_start: bool
    was_all_zero_sample: bool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_distance(xy: np.ndarray, usable_mask: np.ndarray, a_idx: int, b_idx: int) -> float | None:
    """Return Euclidean distance if both joints are usable, else None."""
    if not usable_mask[a_idx] or not usable_mask[b_idx]:
        return None
    dist = float(np.linalg.norm(xy[a_idx] - xy[b_idx]))
    if dist <= 0:
        return None
    return dist


def _choose_center(xy: np.ndarray, usable_mask: np.ndarray) -> np.ndarray | None:
    """Primary center: Neck. Fallback center: MidHip."""
    if usable_mask[NECK_IDX]:
        return xy[NECK_IDX]
    if usable_mask[MID_HIP_IDX]:
        return xy[MID_HIP_IDX]
    return None


def _compute_weighted_scale(xy: np.ndarray, usable_mask: np.ndarray) -> float | None:
    """Compute weighted scale from available body distances."""
    candidates: list[tuple[float, float]] = []

    shoulder = _safe_distance(xy, usable_mask, L_SHOULDER_IDX, R_SHOULDER_IDX)
    if shoulder is not None:
        candidates.append((shoulder, SHOULDER_WEIGHT))

    torso = _safe_distance(xy, usable_mask, NECK_IDX, MID_HIP_IDX)
    if torso is not None:
        candidates.append((torso, TORSO_WEIGHT))

    hip = _safe_distance(xy, usable_mask, L_HIP_IDX, R_HIP_IDX)
    if hip is not None:
        candidates.append((hip, HIP_WEIGHT))

    if not candidates:
        return None

    weighted_sum = sum(v * w for v, w in candidates)
    sum_weights = sum(w for _, w in candidates)
    return weighted_sum / sum_weights


def _is_suspicious_frame(
    current_xy: np.ndarray,
    current_center: np.ndarray,
    current_scale: float,
    current_usable: np.ndarray,
    prev_accepted_xy: np.ndarray | None,
    prev_accepted_center: np.ndarray | None,
    prev_accepted_usable: np.ndarray | None,
) -> bool:
    """Detect likely identity switch / catastrophic body jump.

    A frame is suspicious if:
    - at least two stable joints each jump > 1.25 * current_scale, OR
    - body center jump > 1.5 * current_scale.
    """
    if (
        prev_accepted_xy is None
        or prev_accepted_center is None
        or prev_accepted_usable is None
        or current_scale <= 0
    ):
        return False

    jump_threshold = MULTI_JOINT_JUMP_SCALE_MULTIPLIER * current_scale
    center_threshold = CENTER_JUMP_SCALE_MULTIPLIER * current_scale

    moved_count = 0
    for idx in STABLE_JOINTS:
        if current_usable[idx] and prev_accepted_usable[idx]:
            move_dist = float(np.linalg.norm(current_xy[idx] - prev_accepted_xy[idx]))
            if move_dist > jump_threshold:
                moved_count += 1

    if moved_count >= MULTI_JOINT_MIN_COUNT:
        return True

    center_jump = float(np.linalg.norm(current_center - prev_accepted_center))
    return center_jump > center_threshold


def _parse_frame(frame_json_path: Path, confidence_cutoff: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read one OpenPose frame and return (xy, usable_joint_mask).

    If no people exist, returns (None, None).
    """
    frame_data = _load_json(frame_json_path)
    people = frame_data.get("people", [])
    if not people:
        return None, None

    keypoints = people[0].get("pose_keypoints_2d", [])
    if len(keypoints) < NUM_JOINTS * 3:
        return None, None

    arr = np.asarray(keypoints, dtype=np.float32).reshape(NUM_JOINTS, 3)

    # Keep x/y only for model v1 (do not flatten yet).
    xy = arr[:, :2].copy()
    conf = arr[:, 2]

    # Weak confidence handling: only mark unusable when confidence is extremely low.
    usable_mask = conf >= confidence_cutoff
    return xy, usable_mask


def _collect_take_frame_paths(take_dir: Path) -> list[Path]:
    """Collect and lexicographically sort JSON frame files for one take."""
    return sorted(p for p in take_dir.iterdir() if p.is_file() and p.suffix == ".json")


def _build_sample_from_take(take_dir: Path, confidence_cutoff: float) -> SampleResult:
    """Convert one take folder into a fixed sample with shape (90, 50)."""
    json_paths = _collect_take_frame_paths(take_dir)
    num_raw_frames = len(json_paths)

    # Enforce exactly 90 timeline slots.
    if len(json_paths) >= SEQUENCE_LENGTH:
        timeline_paths: list[Path | None] = json_paths[:SEQUENCE_LENGTH]
    else:
        timeline_paths = json_paths + [None] * (SEQUENCE_LENGTH - len(json_paths))

    parsed_frames: list[ParsedFrame] = []

    prev_valid_scale: float | None = None
    prev_smoothed_scale: float | None = None

    prev_accepted_xy: np.ndarray | None = None
    prev_accepted_center: np.ndarray | None = None
    prev_accepted_usable: np.ndarray | None = None

    for frame_path in timeline_paths:
        if frame_path is None:
            parsed_frames.append(
                ParsedFrame(
                    xy=None,
                    usable_mask=None,
                    center_xy=None,
                    current_scale=None,
                    is_bad=True,
                )
            )
            continue

        xy, usable_mask = _parse_frame(frame_path, confidence_cutoff)
        if xy is None or usable_mask is None:
            parsed_frames.append(
                ParsedFrame(
                    xy=None,
                    usable_mask=None,
                    center_xy=None,
                    current_scale=None,
                    is_bad=True,
                )
            )
            continue

        center = _choose_center(xy, usable_mask)
        if center is None:
            parsed_frames.append(
                ParsedFrame(
                    xy=None,
                    usable_mask=None,
                    center_xy=None,
                    current_scale=None,
                    is_bad=True,
                )
            )
            continue

        computed_scale = _compute_weighted_scale(xy, usable_mask)
        if computed_scale is None:
            if prev_valid_scale is not None:
                current_scale = prev_valid_scale
            else:
                current_scale = SAFE_FALLBACK_SCALE
        else:
            current_scale = computed_scale

        current_scale = max(float(current_scale), MIN_SCALE_EPS)

        suspicious = _is_suspicious_frame(
            current_xy=xy,
            current_center=center,
            current_scale=current_scale,
            current_usable=usable_mask,
            prev_accepted_xy=prev_accepted_xy,
            prev_accepted_center=prev_accepted_center,
            prev_accepted_usable=prev_accepted_usable,
        )

        if suspicious:
            parsed_frames.append(
                ParsedFrame(
                    xy=None,
                    usable_mask=None,
                    center_xy=None,
                    current_scale=None,
                    is_bad=True,
                )
            )
            continue

        # Smooth scale only after the frame is accepted.
        if prev_smoothed_scale is None:
            smoothed_scale = current_scale
        else:
            smoothed_scale = (
                SCALE_SMOOTH_ALPHA_OLD * prev_smoothed_scale
                + SCALE_SMOOTH_ALPHA_NEW * current_scale
            )

        smoothed_scale = max(float(smoothed_scale), MIN_SCALE_EPS)

        normalized_xy = (xy - center[None, :]) / smoothed_scale

        # For extremely low-confidence joints, zero out coordinates.
        normalized_xy[~usable_mask] = 0.0

        parsed_frames.append(
            ParsedFrame(
                xy=normalized_xy,
                usable_mask=usable_mask,
                center_xy=center,
                current_scale=smoothed_scale,
                is_bad=False,
            )
        )

        prev_valid_scale = current_scale
        prev_smoothed_scale = smoothed_scale
        prev_accepted_xy = xy
        prev_accepted_center = center
        prev_accepted_usable = usable_mask

    # -----------------------------
    # Bad frame repair pass
    # -----------------------------
    good_indices = [idx for idx, fr in enumerate(parsed_frames) if not fr.is_bad and fr.xy is not None]

    if not good_indices:
        # Entire take is unusable.
        sample_90x50 = np.zeros((SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)
        return SampleResult(
            sample_90x50=sample_90x50,
            num_raw_frames=num_raw_frames,
            num_bad_frames=SEQUENCE_LENGTH,
            used_future_fill_for_start=False,
            was_all_zero_sample=True,
        )

    repaired_frames: list[np.ndarray | None] = [fr.xy.copy() if fr.xy is not None else None for fr in parsed_frames]

    first_good = good_indices[0]
    used_future_fill_for_start = first_good > 0

    # Fill bad leading frames with first future valid frame.
    if first_good > 0:
        first_good_frame = repaired_frames[first_good]
        assert first_good_frame is not None
        for idx in range(0, first_good):
            repaired_frames[idx] = first_good_frame.copy()

    # Copy-forward fill for all other bad frames.
    last_valid = repaired_frames[first_good]
    assert last_valid is not None
    for idx in range(first_good + 1, SEQUENCE_LENGTH):
        if repaired_frames[idx] is None:
            repaired_frames[idx] = last_valid.copy()
        else:
            last_valid = repaired_frames[idx]

    # Convert (90, 25, 2) -> (90, 50)
    sample_90x25x2 = np.stack(repaired_frames, axis=0).astype(np.float32)
    sample_90x50 = sample_90x25x2.reshape(SEQUENCE_LENGTH, FEATURES_PER_FRAME)

    num_bad_frames = sum(fr.is_bad for fr in parsed_frames)

    return SampleResult(
        sample_90x50=sample_90x50,
        num_raw_frames=num_raw_frames,
        num_bad_frames=num_bad_frames,
        used_future_fill_for_start=used_future_fill_for_start,
        was_all_zero_sample=False,
    )


def _discover_take_dirs(openpose_root: Path, allowed_gestures: list[str]) -> list[tuple[str, str, str, str, Path]]:
    """Find take directories in raw structure: gesture/person/session/take."""
    discovered: list[tuple[str, str, str, str, Path]] = []

    for gesture_dir in sorted(p for p in openpose_root.iterdir() if p.is_dir()):
        gesture = gesture_dir.name
        if gesture not in allowed_gestures:
            continue

        # Require full gesture/person/session/take hierarchy.
        for person_dir in sorted(p for p in gesture_dir.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in person_dir.iterdir() if p.is_dir()):
                for take_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
                    discovered.append(
                        (gesture, person_dir.name, session_dir.name, take_dir.name, take_dir)
                    )

    return discovered


def save_processed_outputs(
    x: np.ndarray,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    label_to_id: dict[str, int],
    output_dir: Path,
    target_mode: str,
) -> None:
    """Save arrays and metadata in the requested processed output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = output_dir / "X.npy"
    y_path = output_dir / "y.npy"
    meta_path = output_dir / "metadata.csv"
    label_map_path = output_dir / "label_map.json"

    np.save(x_path, x)
    np.save(y_path, y)
    metadata_df.to_csv(meta_path, index=False)

    payload = {
        "target_mode": target_mode,
        "label_to_id": label_to_id,
        "id_to_label": {str(v): k for k, v in label_to_id.items()},
    }
    with label_map_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def inspect_processed_sample(processed_dir: Path, sample_index: int = 0) -> None:
    """Small inspection helper for debugging before training."""
    x = np.load(processed_dir / "X.npy")
    y = np.load(processed_dir / "y.npy")
    meta_path = processed_dir / "metadata.csv"
    if meta_path.exists() and meta_path.stat().st_size > 0:
        meta = pd.read_csv(meta_path)
    else:
        meta = pd.DataFrame(columns=METADATA_COLUMNS)

    if len(x) == 0:
        print("Inspection: no samples found in processed outputs.")
        return

    if sample_index < 0 or sample_index >= len(x):
        raise IndexError(f"sample_index {sample_index} is out of range [0, {len(x) - 1}]")

    print(f"Inspection sample index: {sample_index}")
    print(f"Sample tensor shape: {x[sample_index].shape}")
    print(f"Frame tensor shape: {x[sample_index][0].shape}")
    print(f"Label id: {int(y[sample_index])}")
    print("Metadata row:")
    print(meta.iloc[sample_index])


def build_openpose_dataset(
    confidence_cutoff: float = DEFAULT_CONFIDENCE_CUTOFF,
    inspect_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Main preprocessing entry point for OpenPose raw JSON -> processed tensors."""
    paths_cfg = load_paths_config("configs/paths.yaml")
    openpose_root = resolve_path(paths_cfg["openpose_raw_dir"])
    processed_root = resolve_path(paths_cfg["processed_data_dir"])

    gestures_cfg_path = resolve_path("configs/gestures.yaml")
    # Load active gesture labels from target_mode using existing helper.
    active_labels = get_active_labels(gestures_cfg_path)

    # Also load target_mode string for label_map.json metadata.
    with gestures_cfg_path.open("r", encoding="utf-8") as f:
        gestures_cfg = yaml.safe_load(f)
    target_mode = str(gestures_cfg["target_mode"])

    label_to_id = {label: idx for idx, label in enumerate(active_labels)}

    takes = _discover_take_dirs(openpose_root, active_labels)
    print(f"[preprocess] Found {len(takes)} take folders under: {openpose_root}")

    samples_x: list[np.ndarray] = []
    labels_y: list[int] = []
    metadata_rows: list[dict[str, Any]] = []

    per_class_counts = {name: 0 for name in active_labels}
    total_bad_frames = 0

    for gesture, person, session, take, take_dir in takes:
        result = _build_sample_from_take(
            take_dir=take_dir,
            confidence_cutoff=confidence_cutoff,
        )

        samples_x.append(result.sample_90x50)
        labels_y.append(label_to_id[gesture])
        per_class_counts[gesture] += 1
        total_bad_frames += result.num_bad_frames

        metadata_rows.append(
            {
                "gesture": gesture,
                "person": person,
                "session": session,
                "take": take,
                "sample_path": str(take_dir.relative_to(resolve_path("."))),
                "num_raw_frames": result.num_raw_frames,
                "num_bad_frames": result.num_bad_frames,
                "used_future_fill_for_start": result.used_future_fill_for_start,
                "was_all_zero_sample": result.was_all_zero_sample,
            }
        )

    if samples_x:
        x = np.stack(samples_x, axis=0).astype(np.float32)
        y = np.asarray(labels_y, dtype=np.int32)
    else:
        x = np.zeros((0, SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int32)

    metadata_df = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)

    save_processed_outputs(
        x=x,
        y=y,
        metadata_df=metadata_df,
        label_to_id=label_to_id,
        output_dir=processed_root,
        target_mode=target_mode,
    )

    print("[preprocess] Samples per class:")
    for class_name in active_labels:
        print(f"  - {class_name}: {per_class_counts[class_name]}")

    print(f"[preprocess] Total bad frames repaired: {total_bad_frames}")
    print(f"[preprocess] Final X shape: {x.shape}")
    print(f"[preprocess] Final y shape: {y.shape}")
    print(f"[preprocess] Saved to: {processed_root}")

    if inspect_index is not None:
        inspect_processed_sample(processed_root, sample_index=inspect_index)

    return x, y, metadata_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed OpenPose dataset (v1 preprocessing, no training)."
    )
    parser.add_argument(
        "--confidence-cutoff",
        type=float,
        default=DEFAULT_CONFIDENCE_CUTOFF,
        help="Joint usability cutoff for very low confidence values.",
    )
    parser.add_argument(
        "--inspect-index",
        type=int,
        default=None,
        help="Optional sample index to inspect after writing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_openpose_dataset(
        confidence_cutoff=args.confidence_cutoff,
        inspect_index=args.inspect_index,
    )


if __name__ == "__main__":
    main()
