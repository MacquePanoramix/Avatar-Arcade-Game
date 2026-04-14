"""Build processed OpenPose dataset tensors using the runtime-causal preprocessing path.

This module intentionally focuses on:
1) Reading raw OpenPose JSON takes
2) Replaying each take through runtime-safe causal preprocessing rules
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
from src.preprocessing.preprocess_constants import (
    DEFAULT_CONFIDENCE_CUTOFF,
    FEATURES_PER_FRAME,
    SEQUENCE_LENGTH,
)
from src.preprocessing.runtime_preprocess import RuntimePreprocessor
from src.utils.paths import load_paths_config, resolve_path

METADATA_COLUMNS = [
    "gesture",
    "person",
    "session",
    "take",
    "sample_path",
    "num_raw_frames",
    "num_bad_frames",
    "num_joint_zero_fallbacks",
    "num_joint_interpolations",
    "num_joint_symmetric_fills",
    "used_future_fill_for_start",
    "was_all_zero_sample",
]

@dataclass
class SampleResult:
    """Processed output for one take folder."""

    sample_90x30: np.ndarray
    num_raw_frames: int
    num_bad_frames: int
    num_joint_zero_fallbacks: int
    num_joint_interpolations: int
    num_joint_symmetric_fills: int
    used_future_fill_for_start: bool
    was_all_zero_sample: bool


def _collect_take_frame_paths(take_dir: Path) -> list[Path]:
    """Collect and lexicographically sort JSON frame files for one take."""
    return sorted(p for p in take_dir.iterdir() if p.is_file() and p.suffix == ".json")


def _build_sample_from_take(take_dir: Path, confidence_cutoff: float) -> SampleResult:
    """Convert one take folder into a fixed sample with shape (90, 30).

    The take is replayed frame-by-frame through RuntimePreprocessor so dataset
    samples are built from the exact same causal preprocessing logic used live.
    """
    json_paths = _collect_take_frame_paths(take_dir)
    num_raw_frames = len(json_paths)
    runtime = RuntimePreprocessor(confidence_cutoff=confidence_cutoff)
    causal_frames: list[np.ndarray] = []

    num_bad_frames = 0
    for frame_path in json_paths[:SEQUENCE_LENGTH]:
        runtime_result = runtime.process_json_path(frame_path)
        causal_frames.append(runtime_result.features_30)
        if runtime_result.used_prev_frame_copy:
            # Keep frame-level bad count semantics as close as practical:
            # whole-frame fallbacks (no person / no center / suspicious jump).
            num_bad_frames += 1

    padding_count = max(0, SEQUENCE_LENGTH - len(causal_frames))
    if padding_count > 0:
        # Causal short-take padding: extend with the latest emitted runtime frame
        # (or zeros if no frame was ever emitted), never with future interpolation.
        if causal_frames:
            pad_frame = causal_frames[-1].copy()
        else:
            pad_frame = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
        for _ in range(padding_count):
            causal_frames.append(pad_frame.copy())
        num_bad_frames += padding_count

    sample_90x30 = np.stack(causal_frames, axis=0).astype(np.float32)

    return SampleResult(
        sample_90x30=sample_90x30,
        num_raw_frames=num_raw_frames,
        num_bad_frames=num_bad_frames,
        # Legacy metadata fields kept for compatibility; future-aware/interpolation
        # path is now retired in favor of runtime-causal replay.
        num_joint_zero_fallbacks=0,
        num_joint_interpolations=0,
        num_joint_symmetric_fills=0,
        used_future_fill_for_start=False,
        was_all_zero_sample=bool(np.all(sample_90x30 == 0.0)),
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

        samples_x.append(result.sample_90x30)
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
                "num_joint_zero_fallbacks": result.num_joint_zero_fallbacks,
                "num_joint_interpolations": result.num_joint_interpolations,
                "num_joint_symmetric_fills": result.num_joint_symmetric_fills,
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
        description="Build processed OpenPose dataset via runtime-causal preprocessing replay."
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
