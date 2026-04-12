"""Run reproducible training experiment suites with isolated per-run outputs.

This module orchestrates multiple calls to the existing training entrypoint
(`python -m src.training.train_lstm`) using subprocesses.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExperimentSpec:
    """A lightweight experiment definition for one subprocess run."""

    name: str
    args: list[str]
    model_type: str
    tiny_overfit: bool


SUITES: dict[str, list[ExperimentSpec]] = {
    "all": [
        ExperimentSpec("full_mlp", ["--model-type", "mlp"], model_type="mlp", tiny_overfit=False),
        ExperimentSpec("full_lstm", ["--model-type", "lstm"], model_type="lstm", tiny_overfit=False),
        ExperimentSpec(
            "full_lstm_motion",
            ["--model-type", "lstm_motion"],
            model_type="lstm_motion",
            tiny_overfit=False,
        ),
        ExperimentSpec(
            "tiny_overfit_lstm",
            ["--tiny-overfit", "--tiny-model-type", "lstm"],
            model_type="lstm",
            tiny_overfit=True,
        ),
        ExperimentSpec(
            "tiny_overfit_mlp",
            ["--tiny-overfit", "--tiny-model-type", "mlp"],
            model_type="mlp",
            tiny_overfit=True,
        ),
    ],
    "motion_followup": [
        ExperimentSpec("full_mlp", ["--model-type", "mlp"], model_type="mlp", tiny_overfit=False),
        ExperimentSpec(
            "full_mlp_motion",
            ["--model-type", "mlp_motion"],
            model_type="mlp_motion",
            tiny_overfit=False,
        ),
        ExperimentSpec(
            "full_lstm_motion_valacc",
            ["--model-type", "lstm_motion", "--checkpoint-monitor", "val_accuracy"],
            model_type="lstm_motion",
            tiny_overfit=False,
        ),
        ExperimentSpec(
            "full_gru_motion",
            ["--model-type", "gru_motion"],
            model_type="gru_motion",
            tiny_overfit=False,
        ),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a timestamped experiment suite.")
    parser.add_argument("--suite", choices=sorted(SUITES.keys()), default="all")
    parser.add_argument(
        "--output-root",
        type=str,
        default="models/experiment_runs",
        help="Parent directory for timestamped suite folders.",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Regenerate full-dataset split once at suite start, then reuse for full runs.",
    )
    return parser.parse_args()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def get_split_summary(y: np.ndarray, split_paths: dict[str, Path], label_map_path: Path) -> dict[str, Any]:
    label_map: dict[str, Any] = {}
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    id_to_name = {int(v): k for k, v in label_map.get("label_to_id", {}).items()}

    summary: dict[str, Any] = {"sizes": {}, "label_distribution": {}}
    for split_name, split_path in split_paths.items():
        if not split_path.exists():
            continue
        split_idx = np.load(split_path)
        summary["sizes"][split_name] = int(len(split_idx))

        y_split = y[split_idx]
        labels, counts = np.unique(y_split, return_counts=True)
        distribution: dict[str, int] = {}
        for label_id, count in zip(labels, counts):
            label_name = id_to_name.get(int(label_id), f"class_{int(label_id)}")
            distribution[label_name] = int(count)
        summary["label_distribution"][split_name] = distribution
    return summary


def build_notes(row: dict[str, Any]) -> str:
    if row.get("status") != "success":
        return row.get("error", "run failed")
    if row.get("tiny_overfit"):
        train_acc = row.get("final_train_accuracy")
        if isinstance(train_acc, (int, float)):
            return f"Tiny overfit train accuracy={train_acc:.4f}"
        return "Tiny overfit run completed"

    test_acc = row.get("final_test_accuracy")
    if isinstance(test_acc, (int, float)):
        if test_acc < 0.2:
            return "Near chance-level test accuracy"
        if test_acc > 0.85:
            return "Strong baseline performance"
    return "Run completed"


def write_summary_files(suite_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = suite_root / "experiment_summary.csv"
    json_path = suite_root / "experiment_summary.json"
    md_path = suite_root / "experiment_summary.md"

    fieldnames = [
        "experiment_name",
        "status",
        "model_type",
        "input_representation",
        "tiny_overfit",
        "dataset_shape",
        "split_sizes",
        "checkpoint_monitor",
        "best_epoch_by_monitor",
        "epochs_requested",
        "epochs_run",
        "final_train_accuracy",
        "final_val_accuracy",
        "final_test_accuracy",
        "final_train_loss",
        "final_val_loss",
        "final_test_loss",
        "best_checkpoint_path",
        "history_path",
        "metrics_path",
        "confusion_matrix_path",
        "notes",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {k: row.get(k) for k in fieldnames}
            for key in ["dataset_shape", "split_sizes"]:
                if isinstance(csv_row.get(key), (dict, list)):
                    csv_row[key] = json.dumps(csv_row[key], sort_keys=True)
            writer.writerow(csv_row)

    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    successful = [r for r in rows if r.get("status") == "success" and not r.get("tiny_overfit")]
    best = None
    if successful:
        best = max(successful, key=lambda r: float(r.get("final_test_accuracy") or -1.0))

    chance_like = [
        r for r in rows if r.get("status") != "success" or (isinstance(r.get("final_test_accuracy"), (int, float)) and r["final_test_accuracy"] < 0.2)
    ]

    lines = ["# Experiment Suite Summary", "", "## Run Outcomes", ""]
    for row in rows:
        lines.append(
            f"- **{row['experiment_name']}**: {row['status']} | "
            f"test_acc={row.get('final_test_accuracy')} | notes={row.get('notes', '')}"
        )

    lines.extend(["", "## Brief Comparison", ""])
    if best is not None:
        lines.append(
            "- Current best baseline in this suite: "
            f"**{best['experiment_name']}** ({best.get('model_type')}) "
            f"with test accuracy {best.get('final_test_accuracy')}."
        )
    else:
        lines.append("- No successful full-dataset runs were available to identify a baseline.")

    if chance_like:
        lines.append("- Failed or near-chance runs:")
        for row in chance_like:
            lines.append(f"  - {row['experiment_name']}: {row['status']} ({row.get('notes', '')})")

    lines.append("- Interpretation: the MLP remains the strongest valid baseline unless a sequence model exceeds it.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    suite_name = args.suite
    experiments = SUITES[suite_name]

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suite_root = Path(args.output_root) / timestamp
    suite_root.mkdir(parents=True, exist_ok=True)

    print(f"[suite] Starting suite '{suite_name}'")
    print(f"[suite] Output root: {suite_root}")

    split_paths = {
        "train": Path("data/splits/train_indices.npy"),
        "val": Path("data/splits/val_indices.npy"),
        "test": Path("data/splits/test_indices.npy"),
    }

    manifest: dict[str, Any] = {
        "suite": suite_name,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "suite_root": str(suite_root),
        "force_resplit": bool(args.force_resplit),
        "experiments": [e.name for e in experiments],
        "split_info": {
            "mode": "fresh_then_reuse" if args.force_resplit else "reuse_or_create",
            "files": {k: str(v) for k, v in split_paths.items()},
        },
    }

    y_path = Path("data/processed/y.npy")
    label_map_path = Path("data/processed/label_map.json")
    y_data = np.load(y_path) if y_path.exists() else None

    rows: list[dict[str, Any]] = []
    full_run_counter = 0
    for spec in experiments:
        run_dir = suite_root / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            "src.training.train_lstm",
            "--run-name",
            spec.name,
            "--run-dir",
            str(run_dir),
        ]
        command.extend(spec.args)

        if not spec.tiny_overfit:
            command.append("--save-split-copy")
            if args.force_resplit and full_run_counter == 0:
                command.append("--force-resplit")
            full_run_counter += 1

        (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        console_log = run_dir / "console.log"
        stderr_log = run_dir / "stderr.log"

        print(f"[suite] Running {spec.name} ...")
        with console_log.open("w", encoding="utf-8") as out_file, stderr_log.open("w", encoding="utf-8") as err_file:
            completed = subprocess.run(command, stdout=out_file, stderr=err_file, check=False)

        metrics = read_json_if_exists(run_dir / "metrics.json")
        row: dict[str, Any] = {
            "experiment_name": spec.name,
            "status": "success" if completed.returncode == 0 else "failed",
            "model_type": spec.model_type,
            "input_representation": metrics.get("input_representation"),
            "tiny_overfit": spec.tiny_overfit,
            "dataset_shape": metrics.get("dataset_shape"),
            "split_sizes": metrics.get("split_sizes"),
            "checkpoint_monitor": metrics.get("checkpoint_monitor"),
            "best_epoch_by_monitor": metrics.get("best_epoch_by_monitor"),
            "epochs_requested": metrics.get("epochs_requested"),
            "epochs_run": metrics.get("epochs_run"),
            "final_train_accuracy": metrics.get("final_train_accuracy"),
            "final_val_accuracy": metrics.get("final_val_accuracy"),
            "final_test_accuracy": metrics.get("final_test_accuracy"),
            "final_train_loss": metrics.get("final_train_loss"),
            "final_val_loss": metrics.get("final_val_loss"),
            "final_test_loss": metrics.get("final_test_loss"),
            "best_checkpoint_path": metrics.get("best_checkpoint_path"),
            "history_path": metrics.get("history_csv") or metrics.get("history_png"),
            "metrics_path": str(run_dir / "metrics.json"),
            "confusion_matrix_path": metrics.get("confusion_matrix_png") or metrics.get("confusion_matrix_csv"),
            "return_code": int(completed.returncode),
        }

        if completed.returncode != 0:
            row["error"] = f"Subprocess returned non-zero exit code {completed.returncode}"

        row["notes"] = build_notes(row)
        rows.append(row)

    if y_data is not None:
        manifest["split_info"].update(get_split_summary(y_data, split_paths=split_paths, label_map_path=label_map_path))

    manifest["results"] = [
        {
            "experiment_name": row["experiment_name"],
            "status": row["status"],
            "return_code": row.get("return_code"),
            "metrics_path": row.get("metrics_path"),
        }
        for row in rows
    ]
    (suite_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    write_summary_files(suite_root, rows)
    print("[suite] Completed. Summary files written:")
    print(f"  - {suite_root / 'suite_manifest.json'}")
    print(f"  - {suite_root / 'experiment_summary.csv'}")
    print(f"  - {suite_root / 'experiment_summary.json'}")
    print(f"  - {suite_root / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
