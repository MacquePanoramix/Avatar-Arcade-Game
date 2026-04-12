"""Analyze misclassified test samples for a completed experiment run.

Usage examples:

- python -m src.analysis.analyze_misclassifications --run-dir models/experiment_runs/20260412_120000/full_mlp
- python -m src.analysis.analyze_misclassifications --suite-dir models/experiment_runs/20260412_120000
- python -m src.analysis.analyze_misclassifications --latest-suite-dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


TRACEABILITY_COLUMNS = ["gesture", "person", "session", "take", "sample_path", "original_sample_path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze misclassifications for one run folder.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Path to an experiment run folder (for example .../<timestamp>/full_mlp).",
    )
    parser.add_argument(
        "--suite-dir",
        type=str,
        default="",
        help=(
            "Optional convenience input: suite folder path (.../experiment_runs/<timestamp>). "
            "When used without --run-dir, the script analyzes <suite-dir>/<experiment-name>."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="full_mlp",
        help="Experiment subfolder name used with --suite-dir (default: full_mlp).",
    )
    parser.add_argument(
        "--latest-suite-dir",
        action="store_true",
        help=(
            "Convenience flag: use the newest timestamped suite folder under models/experiment_runs "
            "and analyze its full_mlp subfolder (or --experiment-name if provided)."
        ),
    )
    return parser.parse_args()


def find_latest_suite_dir(experiment_runs_root: Path) -> Path:
    if not experiment_runs_root.exists():
        raise FileNotFoundError(f"Missing suite root: {experiment_runs_root}")

    candidates = [p for p in experiment_runs_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No suite directories found under: {experiment_runs_root}")

    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()

    if args.latest_suite_dir:
        suite_dir = find_latest_suite_dir(Path("models/experiment_runs"))
        return (suite_dir / args.experiment_name).resolve()

    if args.suite_dir:
        suite_dir = Path(args.suite_dir).expanduser().resolve()
        return (suite_dir / args.experiment_name).resolve()

    raise ValueError("Provide --run-dir, or --suite-dir, or --latest-suite-dir.")


def load_predictions(run_dir: Path) -> pd.DataFrame:
    """Load best available predictions artifact from a run folder."""
    candidate_names = [
        "predictions.csv",
        "mlp_test_predictions.csv",
        "test_predictions.csv",
        "mlp_motion_test_predictions.csv",
        "lstm_motion_test_predictions.csv",
        "gru_motion_test_predictions.csv",
    ]

    for name in candidate_names:
        path = run_dir / name
        if path.exists():
            print(f"[analysis] Using predictions file: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError(
        "No predictions file found in run folder. Expected one of: "
        f"{', '.join(candidate_names)}"
    )


def normalize_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Support both legacy and richer prediction export schemas."""
    df = predictions.copy()

    rename_map = {
        "y_true": "true_label_id",
        "y_pred": "predicted_label_id",
        "true_label": "true_label_name",
        "pred_label": "predicted_label_name",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "sample_index" not in df.columns:
        print("[analysis][warn] sample_index column missing; using row order as fallback.")
        df["sample_index"] = range(len(df))
    if "split" not in df.columns:
        df["split"] = "test"

    required_names = ["true_label_name", "predicted_label_name"]
    missing_name_cols = [c for c in required_names if c not in df.columns]
    if missing_name_cols:
        raise ValueError(
            "Predictions file is missing required label-name columns: "
            f"{missing_name_cols}."
        )

    if "is_correct" not in df.columns:
        df["is_correct"] = df["true_label_name"] == df["predicted_label_name"]

    for col in ["confidence_of_predicted_class", "confidence_of_true_class", "top2_predicted_confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recover confidence values from per-class probabilities when available.
    if "confidence_of_predicted_class" not in df.columns:
        df["confidence_of_predicted_class"] = pd.NA
    if "confidence_of_true_class" not in df.columns:
        df["confidence_of_true_class"] = pd.NA

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if prob_cols:
        probs = df[prob_cols].apply(pd.to_numeric, errors="coerce")
        missing_pred_conf = df["confidence_of_predicted_class"].isna()
        if missing_pred_conf.any():
            df.loc[missing_pred_conf, "confidence_of_predicted_class"] = probs.max(axis=1)[missing_pred_conf]

        if "predicted_label_name" in df.columns:
            for row_idx, row in df.loc[missing_pred_conf, ["predicted_label_name"]].iterrows():
                pred_name = str(row["predicted_label_name"]).lower().replace(" ", "_")
                pred_prob_col = f"prob_{pred_name}"
                if pred_prob_col in probs.columns:
                    df.at[row_idx, "confidence_of_predicted_class"] = probs.at[row_idx, pred_prob_col]

        missing_true_conf = df["confidence_of_true_class"].isna()
        if missing_true_conf.any() and "true_label_name" in df.columns:
            for row_idx, row in df.loc[missing_true_conf, ["true_label_name"]].iterrows():
                true_name = str(row["true_label_name"]).lower().replace(" ", "_")
                true_prob_col = f"prob_{true_name}"
                if true_prob_col in probs.columns:
                    df.at[row_idx, "confidence_of_true_class"] = probs.at[row_idx, true_prob_col]

        # Derive top-2 info if missing and probabilities exist.
        if "top2_predicted_label_name" not in df.columns:
            df["top2_predicted_label_name"] = pd.NA
        if "top2_predicted_confidence" not in df.columns:
            df["top2_predicted_confidence"] = pd.NA

        if len(prob_cols) >= 2:
            top2_indices = np.argsort(probs.to_numpy(), axis=1)[:, -2]
            top2_conf = probs.to_numpy()[np.arange(len(df)), top2_indices]
            top2_labels = [prob_cols[idx].removeprefix("prob_") for idx in top2_indices]
            top2_missing = df["top2_predicted_confidence"].isna()
            df.loc[top2_missing, "top2_predicted_confidence"] = top2_conf[top2_missing]
            df.loc[df["top2_predicted_label_name"].isna(), "top2_predicted_label_name"] = top2_labels

    if "original_sample_path" not in df.columns and "sample_path" in df.columns:
        df["original_sample_path"] = df["sample_path"]

    return df


def expected_test_size(run_dir: Path) -> int | None:
    """Load expected test split size from run-local or global split artifacts."""
    candidates = [
        run_dir / "splits" / "test_indices.npy",
        Path("data/splits/test_indices.npy"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return int(len(np.load(path)))
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[analysis][warn] Could not read split file {path}: {exc}")
                return None
    print("[analysis][warn] No test split index file found; skipping test-size validation.")
    return None


def load_metadata_candidates() -> pd.DataFrame | None:
    """Load processed metadata used for traceability joins if available."""
    metadata_path = Path("data/processed/metadata.csv")
    if not metadata_path.exists():
        print("[analysis][warn] data/processed/metadata.csv not found; metadata join skipped.")
        return None

    metadata_df = pd.read_csv(metadata_path).reset_index(drop=True)
    if "sample_index" not in metadata_df.columns:
        metadata_df.insert(0, "sample_index", np.arange(len(metadata_df)))
    if "original_sample_path" not in metadata_df.columns and "sample_path" in metadata_df.columns:
        metadata_df["original_sample_path"] = metadata_df["sample_path"]

    duplicate_count = int(metadata_df["sample_index"].duplicated().sum())
    if duplicate_count > 0:
        print(
            "[analysis][warn] metadata.csv has duplicate sample_index values; "
            "keeping first row for each index."
        )
        metadata_df = metadata_df.drop_duplicates(subset=["sample_index"], keep="first")

    return metadata_df


def ensure_traceability_columns(df: pd.DataFrame, metadata_df: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Make sure traceability fields are available, joining from metadata.csv when needed."""
    out = df.copy()
    stats = {
        "rows_missing_traceability_before_join": 0,
        "rows_missing_traceability_after_join": 0,
        "rows_with_person": 0,
        "rows_with_session": 0,
        "rows_with_take": 0,
        "rows_with_original_sample_path": 0,
    }

    for col in TRACEABILITY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    core_traceability = ["person", "session", "take", "original_sample_path"]
    stats["rows_missing_traceability_before_join"] = int(out[core_traceability].isna().all(axis=1).sum())

    if metadata_df is not None:
        metadata_fields = ["sample_index", *[c for c in TRACEABILITY_COLUMNS if c in metadata_df.columns]]
        metadata_subset = metadata_df[metadata_fields].copy()
        merged = out.merge(metadata_subset, on="sample_index", how="left", suffixes=("", "_meta"))

        for col in TRACEABILITY_COLUMNS:
            meta_col = f"{col}_meta"
            if meta_col in merged.columns:
                merged[col] = merged[col].combine_first(merged[meta_col])
                merged = merged.drop(columns=[meta_col])
        out = merged

    stats["rows_missing_traceability_after_join"] = int(out[core_traceability].isna().all(axis=1).sum())
    stats["rows_with_person"] = int(out["person"].notna().sum())
    stats["rows_with_session"] = int(out["session"].notna().sum())
    stats["rows_with_take"] = int(out["take"].notna().sum())
    stats["rows_with_original_sample_path"] = int(out["original_sample_path"].notna().sum())

    return out, stats


def run_validations(df: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    """Run non-fatal validations and return stats for summary outputs."""
    validation: dict[str, Any] = {}

    duplicates = int(df["sample_index"].duplicated().sum())
    validation["duplicate_sample_index_count"] = duplicates
    if duplicates > 0:
        print("[analysis][warn] Duplicate sample_index values found in predictions data.")

    expected = expected_test_size(run_dir)
    validation["expected_test_size"] = expected
    validation["actual_prediction_rows"] = int(len(df))
    if expected is not None and expected != len(df):
        print(
            "[analysis][warn] Prediction row count does not match test split size: "
            f"expected={expected}, actual={len(df)}"
        )

    pred_conf_valid = int(pd.to_numeric(df["confidence_of_predicted_class"], errors="coerce").notna().sum())
    true_conf_valid = int(pd.to_numeric(df["confidence_of_true_class"], errors="coerce").notna().sum())
    validation["valid_confidence_of_predicted_class_count"] = pred_conf_valid
    validation["valid_confidence_of_true_class_count"] = true_conf_valid

    if pred_conf_valid == 0:
        print(
            "[analysis][warn] No valid confidence_of_predicted_class values found. "
            "This run likely needs re-training with richer prediction export enabled."
        )
    if true_conf_valid == 0:
        print(
            "[analysis][warn] No valid confidence_of_true_class values found. "
            "True-class confidence analysis is unavailable for this run."
        )

    nan_pred_conf = int(df["confidence_of_predicted_class"].isna().sum())
    nan_true_conf = int(df["confidence_of_true_class"].isna().sum())
    validation["nan_confidence_of_predicted_class_count"] = nan_pred_conf
    validation["nan_confidence_of_true_class_count"] = nan_true_conf
    if nan_pred_conf > 0:
        print(f"[analysis][warn] confidence_of_predicted_class has {nan_pred_conf} NaN rows.")
    if nan_true_conf > 0:
        print(f"[analysis][warn] confidence_of_true_class has {nan_true_conf} NaN rows.")

    return validation


def build_confusions_by_pair(df: pd.DataFrame) -> pd.DataFrame:
    mis = df[~df["is_correct"]].copy()
    if mis.empty:
        return pd.DataFrame(columns=["true_label", "predicted_label", "count"])

    pair_counts = (
        mis.groupby(["true_label_name", "predicted_label_name"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"true_label_name": "true_label", "predicted_label_name": "predicted_label"})
        .sort_values("count", ascending=False)
    )
    return pair_counts


def summarize_by_class(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_name in sorted(df["true_label_name"].dropna().unique()):
        subset = df[df["true_label_name"] == class_name]
        total = int(len(subset))
        mistakes = int((~subset["is_correct"]).sum())
        rows.append(
            {
                "class_name": class_name,
                "total_samples": total,
                "mistake_count": mistakes,
                "mistake_rate": float(mistakes / total) if total else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values(["mistake_count", "class_name"], ascending=[False, True])


def generate_plots(analysis_dir: Path, class_summary: pd.DataFrame, pair_summary: pd.DataFrame) -> None:
    if not class_summary.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(class_summary["class_name"], class_summary["mistake_count"])
        plt.title("Per-class misclassification counts")
        plt.xlabel("True class")
        plt.ylabel("Number of mistakes")
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(analysis_dir / "per_class_error_bar.png", dpi=150)
        plt.close()

    if not pair_summary.empty:
        top_pairs = pair_summary.head(10).copy()
        labels = [f"{t} → {p}" for t, p in zip(top_pairs["true_label"], top_pairs["predicted_label"])]
        plt.figure(figsize=(11, 5))
        plt.bar(labels, top_pairs["count"])
        plt.title("Top confusion pairs")
        plt.xlabel("True → Predicted")
        plt.ylabel("Count")
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(analysis_dir / "top_confusion_pairs_bar.png", dpi=150)
        plt.close()


def maybe_classification_report(df: pd.DataFrame) -> dict[str, Any]:
    try:
        report = classification_report(
            df["true_label_name"],
            df["predicted_label_name"],
            output_dict=True,
            zero_division=0,
        )
        return report
    except Exception:
        return {}


def write_summary_markdown(
    analysis_dir: Path,
    run_dir: Path,
    df: pd.DataFrame,
    pair_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    clf_report: dict[str, Any],
    validation: dict[str, Any],
    traceability_stats: dict[str, int],
) -> None:
    total = len(df)
    incorrect = int((~df["is_correct"]).sum())
    correct = total - incorrect
    accuracy = (correct / total) if total else 0.0

    pred_conf_count = int(validation.get("valid_confidence_of_predicted_class_count", 0))
    true_conf_count = int(validation.get("valid_confidence_of_true_class_count", 0))
    has_conf = pred_conf_count > 0 and true_conf_count > 0

    rows_with_person = int(traceability_stats.get("rows_with_person", 0))
    rows_with_session = int(traceability_stats.get("rows_with_session", 0))
    rows_with_take = int(traceability_stats.get("rows_with_take", 0))
    rows_with_path = int(traceability_stats.get("rows_with_original_sample_path", 0))
    has_traceability = all(v > 0 for v in [rows_with_person, rows_with_session, rows_with_take, rows_with_path])

    lines = [
        "# Misclassification Analysis",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Total test samples: **{total}**",
        f"- Correct: **{correct}**",
        f"- Incorrect: **{incorrect}**",
        f"- Test accuracy from predictions: **{accuracy:.4f}**",
        "",
        "## Data availability checks",
        "",
        f"- confidence values available: **{'yes' if has_conf else 'no'}**",
        f"  - valid `confidence_of_predicted_class`: {pred_conf_count}/{total}",
        f"  - valid `confidence_of_true_class`: {true_conf_count}/{total}",
        f"- metadata traceability available: **{'yes' if has_traceability else 'no'}**",
        f"  - valid `person`: {rows_with_person}/{total}",
        f"  - valid `session`: {rows_with_session}/{total}",
        f"  - valid `take`: {rows_with_take}/{total}",
        f"  - valid `original_sample_path`: {rows_with_path}/{total}",
    ]

    if not has_conf:
        lines.append(
            "- Missing confidence values mean this run may need re-training to export richer prediction probabilities."
        )
    if not has_traceability:
        lines.append(
            "- Missing traceability means person/session/take clustering checks are incomplete."
        )

    lines.extend(["", "## Top confusion pairs", ""])

    if pair_summary.empty:
        lines.append("No misclassifications were found in this predictions file.")
    else:
        top_n = pair_summary.head(10)
        for _, row in top_n.iterrows():
            lines.append(f"- `{row['true_label']}` → `{row['predicted_label']}`: {int(row['count'])}")

    lines.extend(["", "## Class-by-class mistake breakdown", ""])
    for _, row in class_summary.iterrows():
        lines.append(
            f"- `{row['class_name']}`: mistakes={int(row['mistake_count'])} / total={int(row['total_samples'])} "
            f"(rate={float(row['mistake_rate']):.3f})"
        )

    lines.extend(["", "## Short interpretation notes", ""])
    lines.append("- Use `highest_confidence_errors.csv` to inspect likely label/representation mismatches.")
    lines.append("- Use `hardest_correct_samples.csv` to inspect borderline-but-correct examples.")
    lines.append(
        "- Use metadata fields (`person`, `session`, `take`, `original_sample_path`) to check if confusions cluster by take."
    )

    if clf_report:
        lines.extend(["", "## Precision / recall snapshot", ""])
        for class_name, metrics in clf_report.items():
            if not isinstance(metrics, dict):
                continue
            if class_name in {"macro avg", "weighted avg"}:
                continue
            precision = float(metrics.get("precision", 0.0))
            recall = float(metrics.get("recall", 0.0))
            support = int(metrics.get("support", 0))
            lines.append(
                f"- `{class_name}`: precision={precision:.3f}, recall={recall:.3f}, support={support}"
            )

    metadata_cols = [c for c in ["person", "session", "take", "original_sample_path"] if c in df.columns]
    if metadata_cols and not pair_summary.empty:
        lines.extend(["", "## Example take references for top confusion pairs", ""])
        for _, pair in pair_summary.head(5).iterrows():
            subset = df[
                (~df["is_correct"])
                & (df["true_label_name"] == pair["true_label"])
                & (df["predicted_label_name"] == pair["predicted_label"])
            ].head(3)
            lines.append(f"- `{pair['true_label']}` → `{pair['predicted_label']}`:")
            if subset.empty:
                lines.append("  - (no rows)")
                continue
            for _, row in subset.iterrows():
                reference_bits = []
                for col in metadata_cols:
                    value = row.get(col)
                    if pd.notna(value):
                        reference_bits.append(f"{col}={value}")
                if reference_bits:
                    lines.append(f"  - {', '.join(reference_bits)}")

    (analysis_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    print(f"[analysis] Run directory: {run_dir}")
    predictions = load_predictions(run_dir)
    df = normalize_prediction_columns(predictions)

    metadata_df = load_metadata_candidates()
    df, traceability_stats = ensure_traceability_columns(df, metadata_df)

    validation = run_validations(df, run_dir)

    analysis_dir = run_dir / "misclassification_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    total = len(df)
    incorrect_mask = ~df["is_correct"]
    incorrect = int(incorrect_mask.sum())
    correct = total - incorrect
    accuracy = float(correct / total) if total else 0.0

    print(f"[analysis] Total test rows: {total}")
    print(f"[analysis] Incorrect rows: {incorrect}")

    pair_summary = build_confusions_by_pair(df)
    class_summary = summarize_by_class(df)

    output_columns = [
        "sample_index",
        "split",
        "true_label_id",
        "true_label_name",
        "predicted_label_id",
        "predicted_label_name",
        "confidence_of_predicted_class",
        "confidence_of_true_class",
        "top2_predicted_label_name",
        "top2_predicted_confidence",
        "gesture",
        "person",
        "session",
        "take",
        "sample_path",
        "original_sample_path",
        "is_correct",
    ]
    final_output_cols = [c for c in output_columns if c in df.columns]

    misclassified = df[incorrect_mask].copy().sort_values(
        by=["confidence_of_predicted_class", "sample_index"],
        ascending=[False, True],
    )
    misclassified[final_output_cols].to_csv(analysis_dir / "misclassified_samples.csv", index=False)

    highest_conf_errors = misclassified.head(50)
    highest_conf_errors[final_output_cols].to_csv(analysis_dir / "highest_confidence_errors.csv", index=False)

    hardest_correct = (
        df[df["is_correct"]]
        .copy()
        .sort_values(by=["confidence_of_predicted_class", "sample_index"], ascending=[True, True])
        .head(50)
    )
    hardest_correct[final_output_cols].to_csv(analysis_dir / "hardest_correct_samples.csv", index=False)

    pair_summary.to_csv(analysis_dir / "confusions_by_pair.csv", index=False)

    clf_report = maybe_classification_report(df)
    per_class_precision_recall: dict[str, dict[str, float]] = {}
    for label_name, metrics in clf_report.items():
        if not isinstance(metrics, dict):
            continue
        if label_name in {"macro avg", "weighted avg"}:
            continue
        per_class_precision_recall[label_name] = {
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "support": float(metrics.get("support", 0.0)),
        }

    summary_payload = {
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "total_test_samples": int(total),
        "total_correct": int(correct),
        "total_incorrect": int(incorrect),
        "overall_test_accuracy": accuracy,
        "most_common_confusion_pairs": pair_summary.head(10).to_dict(orient="records"),
        "per_class_mistake_counts": class_summary[["class_name", "mistake_count"]].to_dict(orient="records"),
        "per_class_precision_recall": per_class_precision_recall,
        "source_predictions_columns": list(df.columns),
        "confidence_values_available": bool(validation.get("valid_confidence_of_predicted_class_count", 0) > 0),
        "metadata_traceability_available": bool(traceability_stats.get("rows_with_person", 0) > 0),
        "validation": validation,
        "traceability": traceability_stats,
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    write_summary_markdown(
        analysis_dir=analysis_dir,
        run_dir=run_dir,
        df=df,
        pair_summary=pair_summary,
        class_summary=class_summary,
        clf_report=clf_report,
        validation=validation,
        traceability_stats=traceability_stats,
    )

    generate_plots(analysis_dir=analysis_dir, class_summary=class_summary, pair_summary=pair_summary)

    print(f"[analysis] Wrote outputs to: {analysis_dir}")


if __name__ == "__main__":
    main()
