"""Summarize live debug prediction confidence from a CSV log.

Usage example:
    python -m src.analysis.analyze_live_debug_confidence \
        --csv logs/inference/live_debug_20260412_120000.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze confidence behavior for a live_openpose_debug CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to live debug CSV log.")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.50, 0.70, 0.80, 0.90],
        help="Confidence thresholds to summarize (default: 0.50 0.70 0.80 0.90).",
    )
    return parser.parse_args()


def _clean_counts(series: pd.Series) -> dict[str, int]:
    valid = series.fillna("").astype(str).str.strip()
    valid = valid[valid != ""]
    if valid.empty:
        return {}
    return {str(k): int(v) for k, v in valid.value_counts().items()}


def _detect_intended_label(df: pd.DataFrame) -> str:
    if "intended_label" not in df.columns:
        return ""
    values = df["intended_label"].fillna("").astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return ""
    return str(values.mode().iloc[0])


def build_summary(df: pd.DataFrame, thresholds: list[float]) -> dict[str, Any]:
    out = df.copy()
    out["raw_prediction"] = out.get("raw_prediction", "").fillna("").astype(str).str.strip()
    out["smoothed_prediction"] = out.get("smoothed_prediction", "").fillna("").astype(str).str.strip()
    out["top1_prob"] = pd.to_numeric(out.get("top1_prob", pd.Series(dtype=float)), errors="coerce")

    inference_mask = out["raw_prediction"] != ""
    inference_df = out.loc[inference_mask].copy()
    intended_label = _detect_intended_label(out)

    avg_by_raw = (
        inference_df.groupby("raw_prediction")["top1_prob"].mean().dropna().sort_values(ascending=False)
    )
    max_by_raw = (
        inference_df.groupby("raw_prediction")["top1_prob"].max().dropna().sort_values(ascending=False)
    )

    threshold_counts: dict[str, int] = {}
    for thr in sorted(set(float(t) for t in thresholds)):
        threshold_counts[f">={thr:.2f}"] = int((inference_df["top1_prob"] >= thr).sum())

    summary: dict[str, Any] = {
        "total_frames": int(len(out)),
        "total_inference_frames": int(len(inference_df)),
        "intended_label": intended_label,
        "counts_by_raw_prediction": _clean_counts(out["raw_prediction"]),
        "counts_by_smoothed_prediction": _clean_counts(out["smoothed_prediction"]),
        "average_top1_confidence_overall": (
            float(inference_df["top1_prob"].mean()) if not inference_df.empty else None
        ),
        "average_top1_confidence_by_raw_prediction": {k: float(v) for k, v in avg_by_raw.items()},
        "max_top1_confidence_by_raw_prediction": {k: float(v) for k, v in max_by_raw.items()},
        "frames_above_thresholds": threshold_counts,
    }

    if intended_label:
        match_mask = inference_df["raw_prediction"] == intended_label
        non_match_mask = ~match_mask
        competitors = inference_df.loc[non_match_mask, "raw_prediction"]
        summary["intended_label_analysis"] = {
            "average_top1_confidence_when_predicted_intended": (
                float(inference_df.loc[match_mask, "top1_prob"].mean()) if match_mask.any() else None
            ),
            "average_top1_confidence_when_not_predicted_intended": (
                float(inference_df.loc[non_match_mask, "top1_prob"].mean()) if non_match_mask.any() else None
            ),
            "frames_with_intended_as_top1": int(match_mask.sum()),
            "frames_with_other_top1": int(non_match_mask.sum()),
            "top_competing_predictions": _clean_counts(competitors),
        }

    return summary


def print_terminal_summary(summary: dict[str, Any]) -> None:
    print("=== Live Debug Confidence Summary ===")
    print(
        f"frames_total={summary['total_frames']} "
        f"inference_frames={summary['total_inference_frames']} "
        f"intended_label={summary.get('intended_label') or '(none)'}"
    )
    print(f"avg_top1_conf_overall={summary.get('average_top1_confidence_overall')}")
    print(f"frames_above_thresholds={summary['frames_above_thresholds']}")
    print(f"raw_counts={summary['counts_by_raw_prediction']}")
    print(f"smoothed_counts={summary['counts_by_smoothed_prediction']}")
    intended = summary.get("intended_label_analysis")
    if intended:
        print(
            "intended_analysis: "
            f"frames_with_intended={intended['frames_with_intended_as_top1']} "
            f"frames_with_other={intended['frames_with_other_top1']} "
            f"avg_conf_intended={intended['average_top1_confidence_when_predicted_intended']} "
            f"avg_conf_other={intended['average_top1_confidence_when_not_predicted_intended']}"
        )
        print(f"top_competitors={intended['top_competing_predictions']}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    summary = build_summary(df=df, thresholds=list(args.thresholds))
    summary_path = csv_path.with_name(f"{csv_path.stem}_confidence_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_terminal_summary(summary)
    print(f"Saved summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
