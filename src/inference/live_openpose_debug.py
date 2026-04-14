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

from src.preprocessing.preprocess_constants import FEATURES_PER_FRAME
from src.preprocessing.runtime_preprocess import RuntimePreprocessor
from src.preprocessing.temporal_resampling import (
    SOURCE_NOMINAL_FPS,
    TARGET_FPS,
    TARGET_SEQUENCE_LENGTH,
    resample_sequence_fixed_length,
    source_window_frames_for_target_span,
)
from src.utils.paths import load_paths_config, resolve_path

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


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
        "--tracking-mode",
        type=str,
        choices=["single_person", "two_player_left_right"],
        default="single_person",
        help="Runtime person assignment mode (default: single_person).",
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
    parser.add_argument(
        "--intended-label",
        type=str,
        default="",
        help=(
            "Optional run-level intended gesture label (for example: attack_earth). "
            "Written into every CSV row for downstream confidence analysis."
        ),
    )
    parser.add_argument(
        "--accept-threshold",
        type=float,
        default=0.80,
        help="Minimum top-1 confidence required to ACCEPT non-idle actions (default: 0.80).",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.20,
        help="Minimum (top1_prob - top2_prob) margin required to ACCEPT non-idle actions (default: 0.20).",
    )
    parser.add_argument(
        "--trigger-streak",
        type=int,
        default=3,
        help="Consecutive ACCEPT frames required to emit a TRIGGER action (default: 3).",
    )
    parser.add_argument(
        "--trigger-cooldown-frames",
        type=int,
        default=15,
        help="Inference-frame cooldown after each trigger where new triggers are suppressed (default: 15).",
    )
    parser.add_argument(
        "--overlay-mode",
        type=str,
        choices=["terminal", "window", "both", "none"],
        default="terminal",
        help="Live overlay mode (default: terminal).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable live overlay HUD (equivalent to --overlay-mode none).",
    )
    parser.add_argument(
        "--release-idle-frames",
        type=int,
        default=3,
        help=(
            "Unlock trigger hold protection after this many consecutive release frames. "
            "A release frame is any frame where decision=NO_ACTION, raw/smoothed label is idle, "
            "or smoothed gate top1 confidence is below accept threshold."
        ),
    )
    parser.add_argument(
        "--live-source-fps",
        type=float,
        default=11.0,
        help=(
            "Observed live source FPS used to size the rolling source window before resampling "
            "(default: 11.0)."
        ),
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


def normalize_overlay_mode(args: argparse.Namespace) -> str:
    if args.no_overlay:
        return "none"
    return str(args.overlay_mode)


def decide_action(
    *,
    top1_label: str,
    top1_prob: float,
    top2_prob: float,
    accept_threshold: float,
    margin_threshold: float,
) -> tuple[str, str, float]:
    margin = float(top1_prob - top2_prob)
    is_non_idle = top1_label != "idle"
    meets_conf = top1_prob >= accept_threshold
    meets_margin = margin >= margin_threshold
    if is_non_idle and meets_conf and meets_margin:
        return top1_label, "ACCEPT", margin
    return "NO_ACTION", "NO_ACTION", margin


def print_terminal_overlay(
    *,
    frame_file: str,
    buffer_fill: str,
    raw_label: str,
    smoothed_label: str,
    gate_top1_label: str,
    gate_top1_prob: float,
    gate_top2_label: str,
    gate_top2_prob: float,
    gate_margin: float,
    decision_status: str,
    decision_label: str,
    final_action_status: str,
    final_action_label: str,
    current_accept_streak: int,
    trigger_streak_required: int,
    current_cooldown_remaining: int,
    trigger_locked: bool,
    release_counter: int,
    release_idle_frames: int,
    detected_people_count: int,
    tracking_mode: str,
    selected_person_index: int | None,
    selected_left_person_index: int | None,
    selected_right_person_index: int | None,
    intended_label: str,
) -> None:
    line = (
        f"frame={frame_file} | fill={buffer_fill} | raw={raw_label} | smooth={smoothed_label} | "
        f"gate_top1={gate_top1_label}({gate_top1_prob:.2f}) | "
        f"gate_top2={gate_top2_label}({gate_top2_prob:.2f}) | margin={gate_margin:.2f} | "
        f"decision={decision_status}:{decision_label} | "
        f"trigger={final_action_status}:{final_action_label or '-'} "
        f"(streak={current_accept_streak}/{trigger_streak_required}, cd={current_cooldown_remaining}) | "
        f"lock={'Y' if trigger_locked else 'N'} release={release_counter}/{release_idle_frames} | "
        f"tracking={tracking_mode} "
        f"people={detected_people_count} sel={selected_person_index} "
        f"L={selected_left_person_index} R={selected_right_person_index} "
        f"| intended={intended_label or '-'}"
    )
    print(line, flush=True)


def draw_window_overlay(
    *,
    frame_file: str,
    final_action_status: str,
    final_action_label: str,
    decision_status: str,
    decision_label: str,
    raw_label: str,
    smoothed_label: str,
    gate_top1_label: str,
    gate_top1_prob: float,
    gate_top2_label: str,
    gate_top2_prob: float,
    gate_margin: float,
    current_accept_streak: int,
    trigger_streak_required: int,
    current_cooldown_remaining: int,
    trigger_locked: bool,
    release_counter: int,
    release_idle_frames: int,
    detected_people_count: int,
    selected_person_index: int | None,
    selected_left_person_index: int | None,
    selected_right_person_index: int | None,
    intended_label: str,
    severe_fallback: bool,
) -> None:
    if cv2 is None:
        return
    canvas = np.full((900, 1500, 3), 24, dtype=np.uint8)
    white = (235, 235, 235)
    gray = (170, 170, 170)
    red = (40, 40, 220)
    orange = (0, 180, 240)
    green = (60, 220, 60)
    final_color = green if final_action_status == "TRIGGER" else (orange if decision_status == "ACCEPT" else white)
    accent_color = red if severe_fallback else gray
    final_text = final_action_label if final_action_status == "TRIGGER" else "NO ACTION"

    cv2.putText(canvas, "FINAL ACTION", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, gray, 2, cv2.LINE_AA)
    cv2.putText(canvas, final_text, (40, 190), cv2.FONT_HERSHEY_DUPLEX, 2.6, final_color, 5, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{final_action_status} | decision={decision_status}:{decision_label}",
        (40, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        final_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, f"raw={raw_label}   smooth={smoothed_label}", (40, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.0, white, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        (
            f"gate_top1={gate_top1_label} ({gate_top1_prob:.2f})   "
            f"gate_top2={gate_top2_label} ({gate_top2_prob:.2f})   margin={gate_margin:.2f}"
        ),
        (40, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        white,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"streak={current_accept_streak}/{trigger_streak_required}   cooldown={current_cooldown_remaining}",
        (40, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        white,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"trigger_lock={'ON' if trigger_locked else 'OFF'}   release_counter={release_counter}/{release_idle_frames}",
        (40, 510),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        accent_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"people={detected_people_count}   selected={selected_person_index}   L={selected_left_person_index} R={selected_right_person_index}",
        (40, 570),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        white,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, f"intended={intended_label or '-'}", (40, 630), cv2.FONT_HERSHEY_SIMPLEX, 0.95, white, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"frame: {frame_file}", (40, 860), cv2.FONT_HERSHEY_SIMPLEX, 0.65, gray, 1, cv2.LINE_AA)
    cv2.imshow("Live Gesture HUD", canvas)
    cv2.waitKey(1)


def main() -> None:
    args = parse_args()
    overlay_mode = normalize_overlay_mode(args)
    if args.accept_threshold < 0 or args.accept_threshold > 1:
        raise ValueError("--accept-threshold must be in [0, 1].")
    if args.margin_threshold < 0 or args.margin_threshold > 1:
        raise ValueError("--margin-threshold must be in [0, 1].")
    if args.trigger_streak < 1:
        raise ValueError("--trigger-streak must be >= 1.")
    if args.trigger_cooldown_frames < 0:
        raise ValueError("--trigger-cooldown-frames must be >= 0.")
    if args.release_idle_frames < 1:
        raise ValueError("--release-idle-frames must be >= 1.")
    if args.live_source_fps <= 0:
        raise ValueError("--live-source-fps must be > 0.")

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
    intended_label = str(args.intended_label).strip()
    known_labels = set(id_to_label.values())
    if intended_label and intended_label not in known_labels:
        raise ValueError(
            f"--intended-label '{intended_label}' is not in label map labels: {sorted(known_labels)}"
        )

    model = keras.models.load_model(model_path)
    preprocessor = RuntimePreprocessor(
        confidence_cutoff=args.confidence_cutoff,
        tracking_mode=args.tracking_mode,
    )

    live_source_fps = float(args.live_source_fps)
    live_source_window_frames = source_window_frames_for_target_span(
        target_sequence_length=TARGET_SEQUENCE_LENGTH,
        source_nominal_fps=live_source_fps,
        target_fps=TARGET_FPS,
    )
    rolling: deque[np.ndarray] = deque(maxlen=live_source_window_frames)
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
            "detected_people_count",
            "tracking_mode",
            "selected_person_index",
            "selected_left_person_index",
            "selected_right_person_index",
            "tracking_note",
            "intended_label",
            "top1_margin",
            "smoothed_top1_label",
            "smoothed_top1_prob",
            "smoothed_top2_label",
            "smoothed_top2_prob",
            "smoothed_top1_margin",
            "decision_source",
            "decision_label",
            "decision_status",
            "accept_threshold",
            "margin_threshold",
            "trigger_streak_required",
            "current_accept_streak",
            "trigger_cooldown_frames",
            "current_cooldown_remaining",
            "final_action_status",
            "final_action_label",
            "trigger_locked",
            "release_counter",
            "reset_counter",
            "trigger_lock_was_off",
            "overlay_mode",
            "release_idle_frames",
            "live_source_fps",
            "live_source_window_frames",
        ],
    )
    writer.writeheader()

    print("=== Live OpenPose Debug Classifier ===")
    print(f"JSON dir: {json_dir}")
    print(f"Model: {model_path}")
    print(f"Label map: {label_map_path}")
    print(f"CSV log: {log_csv_path}")
    print(f"Summary: {summary_path}")
    print(f"Tracking mode: {args.tracking_mode}")
    effective_overlay_mode = overlay_mode
    if overlay_mode in {"window", "both"} and cv2 is None:
        print("WARNING: OpenCV (cv2) not found; window HUD disabled.")
        effective_overlay_mode = "terminal" if overlay_mode == "both" else "none"
    print(f"Overlay mode: {effective_overlay_mode} (requested: {overlay_mode})")
    print(f"Decision thresholds: accept>={args.accept_threshold:.2f}, margin>={args.margin_threshold:.2f}")
    print(
        "Trigger filtering: "
        f"streak>={args.trigger_streak} non-idle ACCEPT frames, cooldown={args.trigger_cooldown_frames} frames"
    )
    print(
        "Trigger hold lock release: "
        f"{args.release_idle_frames} consecutive release frames "
        "(NO_ACTION or idle raw/smoothed or smoothed gate confidence below threshold)"
    )
    print(f"Intended label: {intended_label or '(none)'}")
    print(
        "Temporal input policy: "
        f"target_fps={TARGET_FPS:.1f}, "
        f"live_source_fps={live_source_fps:.1f} "
        f"(shared nominal default={SOURCE_NOMINAL_FPS:.1f}), "
        f"source_window_frames={live_source_window_frames}, "
        f"target_sequence_length={TARGET_SEQUENCE_LENGTH}."
    )
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
    decision_status_counts: Counter[str] = Counter()
    decision_label_counts: Counter[str] = Counter()
    final_action_status_counts: Counter[str] = Counter()
    final_action_label_counts: Counter[str] = Counter()
    trigger_counts_by_label: Counter[str] = Counter()
    total_triggers = 0
    trigger_lock_was_off_count = 0
    current_accept_streak = 0
    current_accept_label = ""
    current_cooldown_remaining = 0
    trigger_locked = False
    release_counter = 0

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

                if fill < live_source_window_frames:
                    warmup_frames += 1
                    should_print_warmup = (
                        not args.quiet_warmup
                        and (warmup_frames % print_every_n == 0 or fill == live_source_window_frames - 1)
                    )
                    if should_print_warmup:
                        print(
                            f"[warmup {fill:>2}/{live_source_window_frames}] frame={frame_path.name} "
                            f"| miss={missing_joints:>2} joint_fix={'Y' if frame_result.had_joint_repair else 'N'} "
                            f"prev_copy={'Y' if frame_result.used_prev_frame_copy else 'N'} "
                            f"susp={'Y' if frame_result.suspicious_jump else 'N'} "
                            f"people={frame_result.detected_people_count:>2} "
                            f"sel={frame_result.selected_person_index}"
                        )
                    writer.writerow(
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "frame_file": frame_path.name,
                            "buffer_fill": f"{fill}/{live_source_window_frames}",
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
                            "detected_people_count": frame_result.detected_people_count,
                            "tracking_mode": frame_result.tracking_mode,
                            "selected_person_index": frame_result.selected_person_index,
                            "selected_left_person_index": frame_result.selected_left_person_index,
                            "selected_right_person_index": frame_result.selected_right_person_index,
                            "tracking_note": frame_result.tracking_note,
                            "intended_label": intended_label,
                            "top1_margin": "",
                            "smoothed_top1_label": "",
                            "smoothed_top1_prob": "",
                            "smoothed_top2_label": "",
                            "smoothed_top2_prob": "",
                            "smoothed_top1_margin": "",
                            "decision_source": "smoothed_probs",
                            "decision_label": "",
                            "decision_status": "",
                            "accept_threshold": args.accept_threshold,
                            "margin_threshold": args.margin_threshold,
                            "trigger_streak_required": args.trigger_streak,
                            "current_accept_streak": 0,
                            "trigger_cooldown_frames": args.trigger_cooldown_frames,
                            "current_cooldown_remaining": 0,
                            "final_action_status": "",
                            "final_action_label": "",
                            "trigger_locked": 0,
                            "release_counter": 0,
                            "reset_counter": 0,
                            "trigger_lock_was_off": "",
                            "overlay_mode": effective_overlay_mode,
                            "release_idle_frames": args.release_idle_frames,
                            "live_source_fps": live_source_fps,
                            "live_source_window_frames": live_source_window_frames,
                        }
                    )
                    csv_file.flush()
                    continue

                x_resampled = resample_sequence_fixed_length(
                    np.stack(rolling, axis=0),
                    target_sequence_length=TARGET_SEQUENCE_LENGTH,
                )
                x_window = x_resampled.reshape(1, TARGET_SEQUENCE_LENGTH, FEATURES_PER_FRAME)
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

                top3 = np.argsort(raw_probs)[-3:][::-1]
                raw_top1_idx = int(top3[0])
                raw_top2_idx = int(top3[1])
                raw_top1_label = id_to_label.get(raw_top1_idx, f"class_{raw_top1_idx}")
                raw_top2_label = id_to_label.get(raw_top2_idx, f"class_{raw_top2_idx}")
                raw_top1_prob = float(raw_probs[raw_top1_idx])
                raw_top2_prob = float(raw_probs[raw_top2_idx])
                raw_top1_margin = float(raw_top1_prob - raw_top2_prob)

                smoothed_top3 = np.argsort(ema_probs)[-3:][::-1]
                smoothed_top1_idx = int(smoothed_top3[0])
                smoothed_top2_idx = int(smoothed_top3[1])
                smoothed_top1_label = id_to_label.get(smoothed_top1_idx, f"class_{smoothed_top1_idx}")
                smoothed_top2_label = id_to_label.get(smoothed_top2_idx, f"class_{smoothed_top2_idx}")
                smoothed_top1_prob = float(ema_probs[smoothed_top1_idx])
                smoothed_top2_prob = float(ema_probs[smoothed_top2_idx])
                decision_label, decision_status, top1_margin = decide_action(
                    top1_label=smoothed_top1_label,
                    top1_prob=smoothed_top1_prob,
                    top2_prob=smoothed_top2_prob,
                    accept_threshold=float(args.accept_threshold),
                    margin_threshold=float(args.margin_threshold),
                )
                decision_status_counts[decision_status] += 1
                decision_label_counts[decision_label] += 1
                is_valid_accept = (
                    decision_status == "ACCEPT"
                    and decision_label not in {"", "idle", "NO_ACTION"}
                )
                release_frame = (
                    decision_status == "NO_ACTION"
                    or raw_label == "idle"
                    or smoothed_label == "idle"
                    or smoothed_top1_prob < args.accept_threshold
                )
                if trigger_locked:
                    if release_frame:
                        release_counter += 1
                    else:
                        release_counter = 0
                    if release_counter >= args.release_idle_frames:
                        trigger_locked = False
                        release_counter = 0
                        current_accept_streak = 0
                        current_accept_label = ""
                else:
                    release_counter = 0
                if is_valid_accept:
                    if decision_label == current_accept_label:
                        current_accept_streak += 1
                    else:
                        current_accept_label = decision_label
                        current_accept_streak = 1
                else:
                    current_accept_label = ""
                    current_accept_streak = 0

                final_action_status = "NO_TRIGGER"
                final_action_label = ""
                trigger_lock_was_off = ""
                if current_cooldown_remaining > 0:
                    current_cooldown_remaining -= 1
                elif (not trigger_locked) and current_accept_streak >= args.trigger_streak:
                    final_action_status = "TRIGGER"
                    final_action_label = current_accept_label
                    total_triggers += 1
                    trigger_lock_was_off = 1
                    trigger_lock_was_off_count += 1
                    trigger_counts_by_label[final_action_label] += 1
                    current_cooldown_remaining = args.trigger_cooldown_frames
                    current_accept_streak = 0
                    current_accept_label = ""
                    trigger_locked = True
                    release_counter = 0

                final_action_status_counts[final_action_status] += 1
                final_action_label_counts[final_action_label or "NO_ACTION"] += 1

                severe_fallback = bool(frame_result.used_prev_frame_copy or frame_result.suspicious_jump)
                if effective_overlay_mode in {"terminal", "both"}:
                    print_terminal_overlay(
                        frame_file=frame_path.name,
                        buffer_fill=f"{fill}/{live_source_window_frames}",
                        raw_label=raw_label,
                        smoothed_label=smoothed_label,
                        gate_top1_label=smoothed_top1_label,
                        gate_top1_prob=smoothed_top1_prob,
                        gate_top2_label=smoothed_top2_label,
                        gate_top2_prob=smoothed_top2_prob,
                        gate_margin=top1_margin,
                        decision_status=decision_status,
                        decision_label=decision_label,
                        final_action_status=final_action_status,
                        final_action_label=final_action_label,
                        current_accept_streak=current_accept_streak,
                        trigger_streak_required=args.trigger_streak,
                        current_cooldown_remaining=current_cooldown_remaining,
                        trigger_locked=trigger_locked,
                        release_counter=release_counter,
                        release_idle_frames=args.release_idle_frames,
                        detected_people_count=frame_result.detected_people_count,
                        tracking_mode=frame_result.tracking_mode,
                        selected_person_index=frame_result.selected_person_index,
                        selected_left_person_index=frame_result.selected_left_person_index,
                        selected_right_person_index=frame_result.selected_right_person_index,
                        intended_label=intended_label,
                    )
                if effective_overlay_mode in {"window", "both"}:
                    draw_window_overlay(
                        frame_file=frame_path.name,
                        final_action_status=final_action_status,
                        final_action_label=final_action_label,
                        decision_status=decision_status,
                        decision_label=decision_label,
                        raw_label=raw_label,
                        smoothed_label=smoothed_label,
                        gate_top1_label=smoothed_top1_label,
                        gate_top1_prob=smoothed_top1_prob,
                        gate_top2_label=smoothed_top2_label,
                        gate_top2_prob=smoothed_top2_prob,
                        gate_margin=top1_margin,
                        current_accept_streak=current_accept_streak,
                        trigger_streak_required=args.trigger_streak,
                        current_cooldown_remaining=current_cooldown_remaining,
                        trigger_locked=trigger_locked,
                        release_counter=release_counter,
                        release_idle_frames=args.release_idle_frames,
                        detected_people_count=frame_result.detected_people_count,
                        selected_person_index=frame_result.selected_person_index,
                        selected_left_person_index=frame_result.selected_left_person_index,
                        selected_right_person_index=frame_result.selected_right_person_index,
                        intended_label=intended_label,
                        severe_fallback=severe_fallback,
                    )

                if inference_frames % print_every_n == 0:
                    severity = "!!" if frame_result.used_prev_frame_copy or frame_result.suspicious_jump else "--"
                    print(
                        f"{severity} frame={frame_path.name} | raw={raw_label} | smooth={smoothed_label} "
                        f"| top3: {compact_top3(raw_probs, id_to_label=id_to_label, top_k=3)} "
                        f"| raw_top1={raw_top1_label}:{raw_top1_prob:.3f} "
                        f"raw_top2={raw_top2_label}:{raw_top2_prob:.3f} "
                        f"raw_margin={raw_top1_margin:.3f} "
                        f"| gate_top1={smoothed_top1_label}:{smoothed_top1_prob:.3f} "
                        f"gate_top2={smoothed_top2_label}:{smoothed_top2_prob:.3f} "
                        f"gate_margin={top1_margin:.3f} "
                        f"decision={decision_status}:{decision_label} "
                        f"trigger={final_action_status}:{final_action_label or '-'} "
                        f"streak={current_accept_streak}/{args.trigger_streak} "
                        f"cooldown={current_cooldown_remaining} "
                        f"lock={'Y' if trigger_locked else 'N'} "
                        f"release={release_counter}/{args.release_idle_frames} "
                        f"| people={frame_result.detected_people_count} "
                        f"sel={frame_result.selected_person_index} "
                        f"L={frame_result.selected_left_person_index} "
                        f"R={frame_result.selected_right_person_index}"
                    )
                    print(
                        "   status: "
                        f"joints_missing={missing_joints} "
                        f"joint_repair={'yes' if frame_result.had_joint_repair else 'no'} "
                        f"prev_copy={'yes' if frame_result.used_prev_frame_copy else 'no'} "
                        f"suspicious_jump={'yes' if frame_result.suspicious_jump else 'no'} "
                        f"tracking_note={frame_result.tracking_note}"
                    )

                writer.writerow(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "frame_file": frame_path.name,
                        "buffer_fill": f"{fill}/{live_source_window_frames}",
                        "raw_prediction": raw_label,
                        "smoothed_prediction": smoothed_label,
                        "top1_label": raw_top1_label,
                        "top1_prob": raw_top1_prob,
                        "top2_label": raw_top2_label,
                        "top2_prob": raw_top2_prob,
                        "top3_label": id_to_label.get(int(top3[2]), f"class_{int(top3[2])}"),
                        "top3_prob": float(raw_probs[top3[2]]),
                        "had_joint_repair": int(frame_result.had_joint_repair),
                        "repaired_frame": int(frame_result.was_repaired_frame),
                        "used_prev_frame_copy": int(frame_result.used_prev_frame_copy),
                        "suspicious_jump": int(frame_result.suspicious_jump),
                        "missing_joint_count": missing_joints,
                        "detected_people_count": frame_result.detected_people_count,
                        "tracking_mode": frame_result.tracking_mode,
                        "selected_person_index": frame_result.selected_person_index,
                        "selected_left_person_index": frame_result.selected_left_person_index,
                        "selected_right_person_index": frame_result.selected_right_person_index,
                        "tracking_note": frame_result.tracking_note,
                        "intended_label": intended_label,
                        "top1_margin": raw_top1_margin,
                        "smoothed_top1_label": smoothed_top1_label,
                        "smoothed_top1_prob": smoothed_top1_prob,
                        "smoothed_top2_label": smoothed_top2_label,
                        "smoothed_top2_prob": smoothed_top2_prob,
                        "smoothed_top1_margin": top1_margin,
                        "decision_source": "smoothed_probs",
                        "decision_label": decision_label,
                        "decision_status": decision_status,
                        "accept_threshold": args.accept_threshold,
                        "margin_threshold": args.margin_threshold,
                        "trigger_streak_required": args.trigger_streak,
                        "current_accept_streak": current_accept_streak,
                        "trigger_cooldown_frames": args.trigger_cooldown_frames,
                        "current_cooldown_remaining": current_cooldown_remaining,
                        "final_action_status": final_action_status,
                        "final_action_label": final_action_label,
                        "trigger_locked": int(trigger_locked),
                        "release_counter": release_counter,
                        "reset_counter": release_counter,
                        "trigger_lock_was_off": trigger_lock_was_off,
                        "overlay_mode": effective_overlay_mode,
                        "release_idle_frames": args.release_idle_frames,
                        "live_source_fps": live_source_fps,
                        "live_source_window_frames": live_source_window_frames,
                    }
                )
                csv_file.flush()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if cv2 is not None and effective_overlay_mode in {"window", "both"}:
            cv2.destroyAllWindows()
        if effective_overlay_mode in {"terminal", "both"}:
            print()
        csv_file.close()
        avg_missing_joints = (missing_joint_sum / total_frames) if total_frames > 0 else 0.0
        summary_payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tracking_mode": args.tracking_mode,
            "intended_label": intended_label,
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
            "decision_status_counts": dict(decision_status_counts),
            "decision_label_counts": dict(decision_label_counts),
            "final_action_status_counts": dict(final_action_status_counts),
            "final_action_label_counts": dict(final_action_label_counts),
            "total_triggers": total_triggers,
            "trigger_counts_by_label": dict(trigger_counts_by_label),
            "trigger_lock_was_off_count": trigger_lock_was_off_count,
            "overlay_mode": effective_overlay_mode,
            "trigger_locked_final": trigger_locked,
            "release_counter_final": release_counter,
            "reset_counter_final": release_counter,
            "accept_threshold": args.accept_threshold,
            "margin_threshold": args.margin_threshold,
            "trigger_streak_required": args.trigger_streak,
            "trigger_cooldown_frames": args.trigger_cooldown_frames,
            "release_idle_frames": args.release_idle_frames,
            "trigger_lock_enabled": True,
            "target_fps": TARGET_FPS,
            "target_sequence_length": TARGET_SEQUENCE_LENGTH,
            "live_source_fps": live_source_fps,
            "live_source_window_frames": live_source_window_frames,
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
