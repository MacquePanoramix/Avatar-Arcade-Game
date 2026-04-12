"""Live debug classifier for OpenPose JSON streams.

Usage example:
    python -m src.inference.live_openpose_debug \
        --json-dir data/raw/live_openpose_json \
        --model-path models/checkpoints/best_mlp.keras
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow import keras

from src.preprocessing.build_openpose_dataset import FEATURES_PER_FRAME, SEQUENCE_LENGTH
from src.preprocessing.runtime_preprocess import RuntimePreprocessor
from src.utils.paths import load_paths_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live OpenPose JSON debug classifier (pose-only MLP).")
    parser.add_argument(
        "--json-dir",
        type=str,
        required=True,
        help="Directory continuously receiving OpenPose JSON frames.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/checkpoints/best_mlp.keras",
        help="Path to trained pose-only MLP .keras model (default: models/checkpoints/best_mlp.keras).",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default="data/processed/label_map.json",
        help="Label map JSON path (default: data/processed/label_map.json).",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default="",
        help="Optional CSV log path. If omitted, creates logs/inference/live_debug_<utc>.csv.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Directory polling interval in seconds.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.75,
        help="EMA carry factor for probability smoothing (0..1, higher = smoother).",
    )
    parser.add_argument(
        "--confidence-cutoff",
        type=float,
        default=0.05,
        help="Weak confidence cutoff for runtime joint usability.",
    )
    parser.add_argument(
        "--max-idle-polls",
        type=int,
        default=0,
        help="Optional exit after N consecutive polls with no new frames (0 = never exit).",
    )
    parser.add_argument(
        "--print-every-n",
        type=int,
        default=1,
        help="Print one status block every N processed frames (default: 1).",
    )
    parser.add_argument(
        "--quiet-warmup",
        action="store_true",
        help="Reduce warmup console output; still logs warmup frames to CSV.",
    )
    return parser.parse_args()


def load_label_map(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    id_to_label_raw = payload.get("id_to_label", {})
    if isinstance(id_to_label_raw, dict) and id_to_label_raw:
        return {int(k): str(v) for k, v in id_to_label_raw.items()}

    label_to_id = payload.get("label_to_id", {})
    if isinstance(label_to_id, dict):
        return {int(v): str(k) for k, v in label_to_id.items()}

    raise ValueError(f"Could not parse label mapping from {path}")


def default_log_path() -> Path:
    paths_cfg = load_paths_config()
    logs_root = resolve_path(paths_cfg.get("inference_logs_dir", "logs/inference"))
    logs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return logs_root / f"live_debug_{timestamp}.csv"


def summary_path_from_csv(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_summary.json")


def compact_top3(probs: np.ndarray, id_to_label: dict[int, str], top_k: int = 3) -> str:
    top_idx = np.argsort(probs)[-top_k:][::-1]
    return " | ".join(
        f"{id_to_label.get(int(class_idx), f'class_{int(class_idx)}')}={float(probs[class_idx]):.3f}"
        for class_idx in top_idx
    )


def main() -> None:
    args = parse_args()

    json_dir = Path(args.json_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    label_map_path = Path(args.label_map).expanduser().resolve()
    log_csv_path = Path(args.log_csv).expanduser().resolve() if args.log_csv else default_log_path()

    if not json_dir.exists() or not json_dir.is_dir():
        raise FileNotFoundError(f"JSON directory not found or not a directory: {json_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map path not found: {label_map_path}")

    id_to_label = load_label_map(label_map_path)
    num_classes = len(id_to_label)

    model = keras.models.load_model(model_path)
    preprocessor = RuntimePreprocessor(confidence_cutoff=args.confidence_cutoff)

    rolling: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    ema_probs: np.ndarray | None = None
    seen_files: set[str] = set()
    print_every_n = max(1, int(args.print_every_n))
    summary_path = summary_path_from_csv(log_csv_path)

    log_csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = log_csv_path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "timestamp_utc",
            "frame_file",
            "buffer_fill",
            "raw_prediction",
            "smoothed_prediction",
            "top1_label",
            "top1_prob",
            "top2_label",
            "top2_prob",
            "top3_label",
            "top3_prob",
            "had_joint_repair",
            "repaired_frame",
            "used_prev_frame_copy",
            "suspicious_jump",
            "missing_joint_count",
            "intended_label",
        ],
    )
    writer.writeheader()

    print("=== Live OpenPose Debug Classifier ===")
    print(f"JSON dir: {json_dir}")
    print(f"Model: {model_path}")
    print(f"Label map: {label_map_path}")
    print(f"CSV log: {log_csv_path}")
    print(f"Summary: {summary_path}")
    print("Press Ctrl+C to stop.\n")

    total_frames = 0
    warmup_frames = 0
    inference_frames = 0
    joint_repair_frames = 0
    prev_copy_frames = 0
    suspicious_frames = 0
    missing_joint_sum = 0
    min_missing_joints: int | None = None
    max_missing_joints = 0
    raw_class_counts: Counter[str] = Counter()
    smoothed_class_counts: Counter[str] = Counter()
    raw_smoothed_disagreements: Counter[str] = Counter()

    idle_polls = 0
    try:
        while True:
            frame_paths = sorted(p for p in json_dir.glob("*.json") if p.is_file())
            new_paths = [p for p in frame_paths if p.name not in seen_files]

            if not new_paths:
                idle_polls += 1
                if args.max_idle_polls > 0 and idle_polls >= args.max_idle_polls:
                    print("No new frames detected; exiting due to --max-idle-polls.")
                    break
                time.sleep(args.poll_interval)
                continue

            idle_polls = 0
            for frame_path in new_paths:
                # Mark seen first to avoid repeatedly trying a truncated file forever.
                seen_files.add(frame_path.name)

                try:
                    frame_result = preprocessor.process_json_path(frame_path)
                except json.JSONDecodeError:
                    # OpenPose may still be writing this file; skip safely.
                    continue

                total_frames += 1
                missing_joints = int(frame_result.missing_joint_count)
                missing_joint_sum += missing_joints
                min_missing_joints = (
                    missing_joints
                    if min_missing_joints is None
                    else min(min_missing_joints, missing_joints)
                )
                max_missing_joints = max(max_missing_joints, missing_joints)
                if frame_result.had_joint_repair:
                    joint_repair_frames += 1
                if frame_result.used_prev_frame_copy:
                    prev_copy_frames += 1
                if frame_result.suspicious_jump:
                    suspicious_frames += 1

                rolling.append(frame_result.features_30)
                fill = len(rolling)

                if fill < SEQUENCE_LENGTH:
                    warmup_frames += 1
                    should_print_warmup = (
                        not args.quiet_warmup
                        and (warmup_frames % print_every_n == 0 or fill == SEQUENCE_LENGTH - 1)
                    )
                    if should_print_warmup:
                        print(
                            f"[warmup {fill:>2}/{SEQUENCE_LENGTH}] frame={frame_path.name} "
                            f"| miss={missing_joints:>2} joint_fix={'Y' if frame_result.had_joint_repair else 'N'} "
                            f"prev_copy={'Y' if frame_result.used_prev_frame_copy else 'N'} "
                            f"susp={'Y' if frame_result.suspicious_jump else 'N'}"
                        )
                    writer.writerow(
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "frame_file": frame_path.name,
                            "buffer_fill": f"{fill}/{SEQUENCE_LENGTH}",
                            "raw_prediction": "",
                            "smoothed_prediction": "",
                            "top1_label": "",
                            "top1_prob": "",
                            "top2_label": "",
                            "top2_prob": "",
                            "top3_label": "",
                            "top3_prob": "",
                            "had_joint_repair": int(frame_result.had_joint_repair),
                            "repaired_frame": int(frame_result.was_repaired_frame),
                            "used_prev_frame_copy": int(frame_result.used_prev_frame_copy),
                            "suspicious_jump": int(frame_result.suspicious_jump),
                            "missing_joint_count": missing_joints,
                            "intended_label": "",
                        }
                    )
                    csv_file.flush()
                    continue

                x_window = np.stack(rolling, axis=0).reshape(1, SEQUENCE_LENGTH, FEATURES_PER_FRAME)
                x_flat = x_window.reshape(1, -1)

                raw_probs = model.predict(x_flat, verbose=0)[0].astype(np.float32)
                if raw_probs.shape[0] != num_classes:
                    raise ValueError(
                        f"Model output classes ({raw_probs.shape[0]}) does not match label map ({num_classes})"
                    )

                if ema_probs is None:
                    ema_probs = raw_probs.copy()
                else:
                    ema_probs = (args.smoothing_alpha * ema_probs) + ((1.0 - args.smoothing_alpha) * raw_probs)

                raw_idx = int(np.argmax(raw_probs))
                smoothed_idx = int(np.argmax(ema_probs))
                raw_label = id_to_label.get(raw_idx, f"class_{raw_idx}")
                smoothed_label = id_to_label.get(smoothed_idx, f"class_{smoothed_idx}")
                inference_frames += 1
                raw_class_counts[raw_label] += 1
                smoothed_class_counts[smoothed_label] += 1
                if raw_label != smoothed_label:
                    raw_smoothed_disagreements[f"{raw_label} -> {smoothed_label}"] += 1

                if inference_frames % print_every_n == 0:
                    severity = "!!" if frame_result.used_prev_frame_copy or frame_result.suspicious_jump else "--"
                    print(
                        f"{severity} frame={frame_path.name} | raw={raw_label} | smooth={smoothed_label} "
                        f"| top3: {compact_top3(raw_probs, id_to_label=id_to_label, top_k=3)}"
                    )
                    print(
                        "   status: "
                        f"joints_missing={missing_joints} "
                        f"joint_repair={'yes' if frame_result.had_joint_repair else 'no'} "
                        f"prev_copy={'yes' if frame_result.used_prev_frame_copy else 'no'} "
                        f"suspicious_jump={'yes' if frame_result.suspicious_jump else 'no'}"
                    )

                top3 = np.argsort(raw_probs)[-3:][::-1]
                writer.writerow(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "frame_file": frame_path.name,
                        "buffer_fill": f"{fill}/{SEQUENCE_LENGTH}",
                        "raw_prediction": raw_label,
                        "smoothed_prediction": smoothed_label,
                        "top1_label": id_to_label.get(int(top3[0]), f"class_{int(top3[0])}"),
                        "top1_prob": float(raw_probs[top3[0]]),
                        "top2_label": id_to_label.get(int(top3[1]), f"class_{int(top3[1])}"),
                        "top2_prob": float(raw_probs[top3[1]]),
                        "top3_label": id_to_label.get(int(top3[2]), f"class_{int(top3[2])}"),
                        "top3_prob": float(raw_probs[top3[2]]),
                        "had_joint_repair": int(frame_result.had_joint_repair),
                        "repaired_frame": int(frame_result.was_repaired_frame),
                        "used_prev_frame_copy": int(frame_result.used_prev_frame_copy),
                        "suspicious_jump": int(frame_result.suspicious_jump),
                        "missing_joint_count": missing_joints,
                        "intended_label": "",
                    }
                )
                csv_file.flush()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        csv_file.close()
        avg_missing_joints = (missing_joint_sum / total_frames) if total_frames > 0 else 0.0
        summary_payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "total_frames_processed": total_frames,
            "warmup_frames": warmup_frames,
            "inference_frames": inference_frames,
            "frames_with_joint_repair": joint_repair_frames,
            "frames_with_prev_frame_copy": prev_copy_frames,
            "frames_flagged_suspicious_jump": suspicious_frames,
            "missing_joints": {
                "average": avg_missing_joints,
                "min": 0 if min_missing_joints is None else min_missing_joints,
                "max": max_missing_joints,
            },
            "raw_prediction_class_counts": dict(raw_class_counts),
            "smoothed_prediction_class_counts": dict(smoothed_class_counts),
            "raw_to_smoothed_disagreements_top10": dict(raw_smoothed_disagreements.most_common(10)),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        print("\n=== Run Summary ===")
        print(f"frames_total={total_frames} warmup={warmup_frames} inference={inference_frames}")
        print(
            "repair_stats: "
            f"joint_repair={joint_repair_frames} "
            f"prev_copy={prev_copy_frames} "
            f"suspicious_jump={suspicious_frames}"
        )
        print(
            "missing_joints: "
            f"avg={avg_missing_joints:.2f} min={summary_payload['missing_joints']['min']} "
            f"max={summary_payload['missing_joints']['max']}"
        )
        if raw_class_counts:
            print(f"raw_counts: {dict(raw_class_counts)}")
        if smoothed_class_counts:
            print(f"smoothed_counts: {dict(smoothed_class_counts)}")
        if raw_smoothed_disagreements:
            print(f"raw->smoothed disagreements (top): {dict(raw_smoothed_disagreements.most_common(5))}")
        print(f"Log saved: {log_csv_path}")
        print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
