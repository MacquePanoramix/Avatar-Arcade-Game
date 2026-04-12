"""Beginner-friendly baseline LSTM training script for processed OpenPose data.

This script expects preprocessing outputs to already exist in ``data/processed``:
- X.npy
- y.npy
- metadata.csv
- label_map.json

It performs a simple stratified random split, trains a small Keras LSTM model,
and saves baseline reports/artifacts for a first training run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config if available; return empty config when missing."""
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    return loaded if isinstance(loaded, dict) else {}


def get_training_value(config: dict[str, Any], key: str, default: Any) -> Any:
    """Read a value from config['training']; fallback to default if missing."""
    training_cfg = config.get("training", {}) if isinstance(config, dict) else {}
    return training_cfg.get(key, default)


def has_training_value(config: dict[str, Any], key: str) -> bool:
    """Return True when config['training'] explicitly provides a given key."""
    training_cfg = config.get("training", {}) if isinstance(config, dict) else {}
    return key in training_cfg


def load_processed_data(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load model inputs/targets + label mapping information."""
    x_path = processed_dir / "X.npy"
    y_path = processed_dir / "y.npy"
    label_map_path = processed_dir / "label_map.json"

    if not x_path.exists() or not y_path.exists() or not label_map_path.exists():
        raise FileNotFoundError(
            "Missing processed files. Expected: data/processed/X.npy, data/processed/y.npy, "
            "and data/processed/label_map.json."
        )

    x = np.load(x_path)
    y = np.load(y_path)

    with label_map_path.open("r", encoding="utf-8") as file:
        label_map = json.load(file)

    return x, y, label_map


def print_dataset_summary(x: np.ndarray, y: np.ndarray, label_map: dict[str, Any]) -> None:
    """Print beginner-friendly dataset summary in console."""
    unique_classes, class_counts = np.unique(y, return_counts=True)

    print("\n=== Dataset Summary ===")
    print(f"X shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of classes in y: {len(unique_classes)}")

    id_to_name = {int(v): k for k, v in label_map.get("label_to_id", {}).items()}
    print("Class counts:")
    for class_id, count in zip(unique_classes, class_counts):
        class_name = id_to_name.get(int(class_id), f"class_{int(class_id)}")
        print(f"  - {class_id} ({class_name}): {count}")


def split_data_stratified(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create 70/15/15 train/val/test sample indices via stratified random split."""
    all_indices = np.arange(len(y))

    # First split: 70% train, 30% temporary pool.
    train_idx, temp_idx, _, y_temp = train_test_split(
        all_indices,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    # Second split: split temp equally => 15% validation, 15% test overall.
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    return train_idx, val_idx, test_idx


def print_split_summary(name: str, y_subset: np.ndarray, label_map: dict[str, Any]) -> None:
    """Print class balance for one split."""
    unique_classes, class_counts = np.unique(y_subset, return_counts=True)
    id_to_name = {int(v): k for k, v in label_map.get("label_to_id", {}).items()}

    print(f"\n{name} split class balance:")
    for class_id, count in zip(unique_classes, class_counts):
        class_name = id_to_name.get(int(class_id), f"class_{int(class_id)}")
        print(f"  - {class_id} ({class_name}): {count}")


def build_lstm_model(
    input_shape: tuple[int, int],
    num_classes: int,
    lstm_units: int,
    learning_rate: float,
    use_dropout: bool = True,
    use_masking: bool = True,
) -> keras.Model:
    """Build the requested simple baseline LSTM classifier."""
    model_layers: list[keras.layers.Layer] = [keras.layers.Input(shape=input_shape)]

    # IMPORTANT:
    # - Masking is useful only when a special value (like 0.0) encodes "missing/padded" frames.
    # - After z-score standardization, 0.0 is simply "mean value", not "missing".
    # - Therefore, standardized sequence paths should disable masking (use_masking=False).
    if use_masking:
        model_layers.append(keras.layers.Masking(mask_value=0.0))

    model_layers.append(keras.layers.LSTM(lstm_units))

    # In normal training we keep dropout for regularization.
    # In tiny-overfit mode we can disable dropout to maximize memorization power.
    if use_dropout:
        model_layers.append(keras.layers.Dropout(0.3))

    model_layers.extend(
        [
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model = keras.Sequential(model_layers)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_motion_aware_sequences(x: np.ndarray) -> np.ndarray:
    """Create motion-aware inputs by concatenating position and frame deltas.

    Delta rule used:
    - delta[t] = x[t] - x[t-1]
    - delta[0] = 0 (all-zero vector for the first frame)

    For an original per-sample shape of (timesteps, features) = (90, 30),
    this returns (90, 60) after concatenating [position, delta] per frame.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array (samples, timesteps, features), got {x.shape}")

    deltas = np.zeros_like(x)
    deltas[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
    return np.concatenate([x, deltas], axis=-1)


def build_lstm_motion_model(
    input_shape: tuple[int, int],
    num_classes: int,
    learning_rate: float,
) -> keras.Model:
    """Build the motion-aware LSTM baseline architecture."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(128),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru_motion_model(
    input_shape: tuple[int, int],
    num_classes: int,
    learning_rate: float,
) -> keras.Model:
    """Build the requested GRU motion-aware architecture."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.GRU(128),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mlp_motion_model(
    input_dim: int,
    num_classes: int,
    learning_rate: float,
) -> keras.Model:
    """Build an MLP baseline on flattened motion-aware sequence features."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_sequence_normalization_stats(x_train_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std from training split only for 3D sequence tensors.

    Expected input shape:
    - (samples, timesteps, features)

    Reduction axes for train-only statistics:
    - across samples axis (axis=0)
    - across timesteps axis (axis=1)
    - keep feature axis separate
    """
    if x_train_seq.ndim != 3:
        raise ValueError(
            f"Expected 3D sequence array (samples, timesteps, features), got {x_train_seq.shape}"
        )
    feature_mean = np.mean(x_train_seq, axis=(0, 1), keepdims=True)
    feature_std = np.std(x_train_seq, axis=(0, 1), keepdims=True)
    return feature_mean, feature_std


def standardize_sequence_data(
    x_data: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    """Apply feature-wise standardization: X_std = (X - mean) / (std + 1e-8)."""
    return (x_data - feature_mean) / (feature_std + 1e-8)


def save_sequence_normalization_stats(
    reports_dir: Path,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> tuple[Path, Path]:
    """Persist train-only sequence normalization stats for reproducible future runs."""
    mean_path = reports_dir / "sequence_feature_mean.npy"
    std_path = reports_dir / "sequence_feature_std.npy"
    np.save(mean_path, feature_mean)
    np.save(std_path, feature_std)
    return mean_path, std_path


def save_tiny_overfit_history(history: keras.callbacks.History, reports_dir: Path) -> None:
    """Save tiny-overfit history using dedicated filenames."""
    save_tiny_overfit_history_for_model(history=history, reports_dir=reports_dir, tiny_model_type="lstm")


def save_tiny_overfit_history_for_model(
    history: keras.callbacks.History,
    reports_dir: Path,
    tiny_model_type: str,
) -> None:
    """Save tiny-overfit history with model-specific filenames (lstm/mlp)."""
    history_df = pd.DataFrame(history.history)
    history_csv_path = reports_dir / f"tiny_overfit_{tiny_model_type}_history.csv"
    history_df.to_csv(history_csv_path, index=False)

    # In tiny-overfit mode we only train on one tiny subset (no validation split),
    # so plotting training-only curves keeps the figure easy to understand.
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("loss", []), label="loss")
    plt.title("Tiny Overfit Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("accuracy", []), label="accuracy")
    plt.title("Tiny Overfit Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(reports_dir / f"tiny_overfit_{tiny_model_type}_history.png", dpi=150)
    plt.close()


def build_tiny_mlp_model(
    input_dim: int,
    num_classes: int,
    learning_rate: float,
) -> keras.Model:
    """Build a simple MLP for tiny-overfit diagnostics on flattened inputs."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_tiny_balanced_subset(
    x: np.ndarray,
    y: np.ndarray,
    max_samples_per_class: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a tiny, roughly balanced subset with up to N samples per class."""
    rng = np.random.default_rng(random_seed)
    chosen_indices: list[int] = []

    # We explicitly loop class-by-class to keep class balance simple and clear.
    for class_id in sorted(np.unique(y)):
        class_indices = np.where(y == class_id)[0]
        rng.shuffle(class_indices)
        selected_for_class = class_indices[:max_samples_per_class]
        chosen_indices.extend(selected_for_class.tolist())

    # Shuffle one final time so tiny training batches are not grouped by class.
    chosen_indices_array = np.array(chosen_indices, dtype=int)
    rng.shuffle(chosen_indices_array)

    return x[chosen_indices_array], y[chosen_indices_array]


def parse_args() -> argparse.Namespace:
    """Parse optional CLI flags for training modes."""
    parser = argparse.ArgumentParser(description="Train baseline LSTM for Avatar Arcade data.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["lstm", "mlp", "mlp_motion", "lstm_motion", "gru_motion"],
        help=(
            "Model architecture for normal full-dataset training mode. "
            "Use 'lstm' (default), 'mlp', 'mlp_motion', 'lstm_motion', or 'gru_motion'."
        ),
    )
    parser.add_argument(
        "--checkpoint-monitor",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_accuracy"],
        help=(
            "Validation metric monitored by EarlyStopping and ModelCheckpoint. "
            "Use 'val_accuracy' for motion-sequence reruns when needed."
        ),
    )
    parser.add_argument(
        "--tiny-overfit",
        action="store_true",
        help=(
            "Run a tiny diagnostic training mode (up to 2 samples per class, no split) "
            "to check if the pipeline can memorize a very small dataset."
        ),
    )
    parser.add_argument(
        "--tiny-model-type",
        type=str,
        default="lstm",
        choices=["lstm", "mlp"],
        help=(
            "Model architecture to use in --tiny-overfit mode. "
            "Use 'lstm' (default) or 'mlp'."
        ),
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Always regenerate train/val/test split indices and overwrite saved split files.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help=(
            "Optional descriptive run name stored in config/metrics outputs. "
            "Does not change behavior when omitted."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help=(
            "Optional output folder for this run. When provided, training artifacts are written "
            "inside that folder instead of default models/reports + models/checkpoints paths."
        ),
    )
    parser.add_argument(
        "--save-split-copy",
        action="store_true",
        help=(
            "When used with --run-dir in full-dataset mode, copy train/val/test split index files "
            "into <run-dir>/splits for experiment tracking."
        ),
    )
    return parser.parse_args()


def compute_label_distribution(y_values: np.ndarray, label_map: dict[str, Any]) -> dict[str, int]:
    """Build a JSON-friendly class-count mapping using label names."""
    id_to_name = {int(v): k for k, v in label_map.get("label_to_id", {}).items()}
    unique_classes, class_counts = np.unique(y_values, return_counts=True)
    distribution: dict[str, int] = {}
    for class_id, count in zip(unique_classes, class_counts):
        class_name = id_to_name.get(int(class_id), f"class_{int(class_id)}")
        distribution[class_name] = int(count)
    return distribution


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty JSON to disk with stable key ordering."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_test_reports(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
    label_map: dict[str, Any],
    reports_dir: Path,
    test_indices: np.ndarray | None = None,
    metadata_df: pd.DataFrame | None = None,
    filename_prefix: str = "",
) -> None:
    """Save classification report, confusion matrix, and per-sample predictions."""
    id_to_name = {int(v): k for k, v in label_map.get("label_to_id", {}).items()}

    labels_sorted = sorted(np.unique(np.concatenate([y_test, y_pred])))
    target_names = [id_to_name.get(int(label), f"class_{int(label)}") for label in labels_sorted]

    report_text = classification_report(
        y_test,
        y_pred,
        labels=labels_sorted,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    (reports_dir / f"{filename_prefix}classification_report.txt").write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(reports_dir / f"{filename_prefix}confusion_matrix.csv")
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_positions = np.arange(len(target_names))
    plt.xticks(tick_positions, target_names, rotation=45, ha="right")
    plt.yticks(tick_positions, target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{filename_prefix}confusion_matrix.png", dpi=150)
    plt.close()

    if y_pred_probs.ndim != 2:
        raise ValueError(
            f"Expected y_pred_probs shape (num_samples, num_classes), got {y_pred_probs.shape}"
        )
    if y_pred_probs.shape[0] != len(y_test):
        raise ValueError(
            "Predicted probability row count does not match y_test length: "
            f"{y_pred_probs.shape[0]} vs {len(y_test)}"
        )

    class_ids_sorted = sorted(id_to_name.keys())
    predicted_confidence = np.max(y_pred_probs, axis=1)
    true_confidence = y_pred_probs[np.arange(len(y_test)), y_test]
    top2_class_ids = np.argsort(y_pred_probs, axis=1)[:, -2]
    top2_confidence = y_pred_probs[np.arange(len(y_test)), top2_class_ids]

    predictions_df = pd.DataFrame(
        {
            "sample_index": test_indices if test_indices is not None else np.arange(len(y_test)),
            "split": "test",
            "true_label_id": y_test,
            "true_label_name": [id_to_name.get(int(v), f"class_{int(v)}") for v in y_test],
            "predicted_label_id": y_pred,
            "predicted_label_name": [id_to_name.get(int(v), f"class_{int(v)}") for v in y_pred],
            "confidence_of_predicted_class": predicted_confidence,
            "confidence_of_true_class": true_confidence,
            "top2_predicted_label_name": [
                id_to_name.get(int(v), f"class_{int(v)}") for v in top2_class_ids
            ],
            "top2_predicted_confidence": top2_confidence,
            "is_correct": y_test == y_pred,
        }
    )

    # Add one probability column per class for richer downstream error analysis.
    for class_id in class_ids_sorted:
        class_name = id_to_name.get(int(class_id), f"class_{int(class_id)}")
        safe_name = class_name.lower().replace(" ", "_")
        predictions_df[f"prob_{safe_name}"] = y_pred_probs[:, int(class_id)]

    # Add traceability fields from preprocessing metadata (if available).
    if metadata_df is not None and "sample_index" in metadata_df.columns:
        metadata_fields = [
            "gesture",
            "person",
            "session",
            "take",
            "sample_path",
            "original_sample_path",
        ]
        metadata_subset_cols = [c for c in metadata_fields if c in metadata_df.columns]
        if metadata_subset_cols:
            metadata_subset = metadata_df[["sample_index", *metadata_subset_cols]].copy()
            duplicate_count = int(metadata_subset["sample_index"].duplicated().sum())
            if duplicate_count > 0:
                print(
                    "Warning: metadata contains duplicate sample_index values; "
                    "dropping duplicates before prediction join."
                )
                metadata_subset = metadata_subset.drop_duplicates(subset=["sample_index"], keep="first")
            predictions_df = predictions_df.merge(metadata_subset, on="sample_index", how="left")
            if "sample_path" in predictions_df.columns and "original_sample_path" not in predictions_df.columns:
                predictions_df["original_sample_path"] = predictions_df["sample_path"]

            missing_metadata_rows = int(predictions_df["gesture"].isna().sum()) if "gesture" in predictions_df.columns else 0
            if missing_metadata_rows > 0:
                print(
                    f"Warning: metadata join could not find rows for {missing_metadata_rows} "
                    "test samples. Check sample_index alignment."
                )

    # Backward-compatible artifact name.
    predictions_df.to_csv(reports_dir / f"{filename_prefix}test_predictions.csv", index=False)
    # Canonical name for analysis workflows.
    predictions_df.to_csv(reports_dir / "predictions.csv", index=False)


def save_history(
    history: keras.callbacks.History,
    reports_dir: Path,
    filename_prefix: str = "",
) -> None:
    """Save training history to CSV and PNG plot."""
    history_df = pd.DataFrame(history.history)
    history_csv_path = reports_dir / f"{filename_prefix}training_history.csv"
    history_df.to_csv(history_csv_path, index=False)

    plt.figure(figsize=(10, 4))

    # Left plot: loss curves.
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("loss", []), label="loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Right plot: accuracy curves.
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("accuracy", []), label="accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(reports_dir / f"{filename_prefix}training_history.png", dpi=150)
    plt.close()


def build_mlp_model(
    input_dim: int,
    num_classes: int,
    learning_rate: float,
) -> keras.Model:
    """Build a simple full-dataset MLP baseline on flattened sequence features."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_or_create_split_indices(
    y: np.ndarray,
    splits_dir: Path,
    random_state: int,
    force_resplit: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load valid split indices when available; otherwise create and save them."""
    train_idx_path = splits_dir / "train_indices.npy"
    val_idx_path = splits_dir / "val_indices.npy"
    test_idx_path = splits_dir / "test_indices.npy"

    split_files_exist = train_idx_path.exists() and val_idx_path.exists() and test_idx_path.exists()
    dataset_size = len(y)

    def validate_saved_splits(
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple[bool, str]:
        split_names_and_indices = [("train", train_idx), ("validation", val_idx), ("test", test_idx)]

        for split_name, split_idx in split_names_and_indices:
            if len(split_idx) == 0:
                return False, f"{split_name} split is empty"
            if np.any(split_idx < 0):
                return False, f"{split_name} split contains negative indices"
            if np.any(split_idx >= dataset_size):
                return False, f"{split_name} split contains out-of-bounds indices for dataset size {dataset_size}"

        combined = np.concatenate([train_idx, val_idx, test_idx])
        unique_indices = np.unique(combined)

        if len(unique_indices) != dataset_size:
            return (
                False,
                f"unique index count across splits ({len(unique_indices)}) does not match dataset size ({dataset_size})",
            )
        if len(combined) != len(unique_indices):
            return False, "duplicate indices found across train/validation/test splits"

        return True, ""

    def create_and_save_new_splits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # We pass x as a placeholder array because the split helper uses y-based stratification.
        x_placeholder = np.zeros((len(y), 1))
        train_idx_new, val_idx_new, test_idx_new = split_data_stratified(
            x=x_placeholder,
            y=y,
            random_state=random_state,
        )
        np.save(train_idx_path, train_idx_new)
        np.save(val_idx_path, val_idx_new)
        np.save(test_idx_path, test_idx_new)
        return train_idx_new, val_idx_new, test_idx_new

    if split_files_exist and not force_resplit:
        train_idx = np.load(train_idx_path)
        val_idx = np.load(val_idx_path)
        test_idx = np.load(test_idx_path)

        is_valid, reason = validate_saved_splits(train_idx, val_idx, test_idx)
        if is_valid:
            print("\nUsing valid existing split indices")
        else:
            print(f"\nWarning: {reason}")
            print("Saved splits are stale/invalid; regenerating new splits")
            train_idx, val_idx, test_idx = create_and_save_new_splits()
    else:
        if force_resplit and split_files_exist:
            print("\n--force-resplit provided; regenerating train/val/test splits")
        else:
            print("\nNo saved split indices found. Creating new 70/15/15 stratified split.")
        train_idx, val_idx, test_idx = create_and_save_new_splits()

    return train_idx, val_idx, test_idx

def main() -> None:
    """Run full first-baseline training flow."""
    args = parse_args()

    config = load_config(Path("configs/config.yaml"))

    random_seed = int(get_training_value(config, "random_seed", 42))
    batch_size = int(get_training_value(config, "batch_size", 32))
    learning_rate = float(get_training_value(config, "learning_rate", 0.001))
    lstm_units = int(get_training_value(config, "lstm_units", 64))

    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    processed_dir = Path("data/processed")
    splits_dir = Path("data/splits")
    default_checkpoints_dir = Path("models/checkpoints")
    default_reports_dir = Path("models/reports")
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    run_name = args.run_name.strip() if args.run_name else ""

    splits_dir.mkdir(parents=True, exist_ok=True)
    if run_dir is None:
        checkpoints_dir = default_checkpoints_dir
        reports_dir = default_reports_dir
    else:
        checkpoints_dir = run_dir / "checkpoints"
        reports_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    x, y, label_map = load_processed_data(processed_dir)
    metadata_df: pd.DataFrame | None = None
    metadata_path = processed_dir / "metadata.csv"
    if metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path).reset_index(drop=True)
        if "sample_index" not in metadata_df.columns:
            metadata_df.insert(0, "sample_index", np.arange(len(metadata_df)))
        if "sample_path" in metadata_df.columns and "original_sample_path" not in metadata_df.columns:
            metadata_df["original_sample_path"] = metadata_df["sample_path"]
        duplicate_metadata_indices = int(metadata_df["sample_index"].duplicated().sum())
        if duplicate_metadata_indices > 0:
            print(
                "\nWarning: metadata.csv has duplicate sample_index values; "
                "keeping the first row for each sample_index."
            )
            metadata_df = metadata_df.drop_duplicates(subset=["sample_index"], keep="first")
        if len(metadata_df) != len(y):
            print(
                "\nWarning: metadata.csv row count does not match dataset size. "
                "Skipping metadata join for predictions export."
            )
            metadata_df = None
    else:
        print("\nNote: metadata.csv not found in data/processed; predictions export will omit traceability fields.")
    print_dataset_summary(x, y, label_map)

    if x.ndim != 3:
        raise ValueError(f"Expected X shape (samples, timesteps, features), got {x.shape}")

    config_snapshot = {
        "run_name": run_name if run_name else None,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "argv": sys.argv,
        "args": vars(args),
        "random_seed": random_seed,
        "training_config": config.get("training", {}),
        "processed_dir": str(processed_dir),
        "splits_dir": str(splits_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "reports_dir": str(reports_dir),
        "dataset_shape": list(x.shape),
        "label_distribution": compute_label_distribution(y, label_map),
    }
    write_json(reports_dir / "config.json", config_snapshot)

    if args.tiny_overfit:
        print("\n=== Tiny Overfit Diagnostic Mode (enabled) ===")
        print("This mode trains on a tiny balanced subset only (no train/val/test split).")
        print(f"Tiny model type: {args.tiny_model_type}")

        # Build a tiny subset with up to 2 samples per class.
        x_tiny, y_tiny = build_tiny_balanced_subset(
            x=x,
            y=y,
            max_samples_per_class=2,
            random_seed=random_seed,
        )

        print(f"Tiny subset shape: X={x_tiny.shape}, y={y_tiny.shape}")
        print("Expected target is ~18 samples for 9 classes (if all classes have >= 2 samples).")

        # Tiny-overfit configuration is intentionally aggressive so we can test
        # "can this representation be memorized at all?" in a very direct way.
        tiny_epochs = 300
        tiny_batch_size = 1

        if args.tiny_model_type == "lstm":
            # Tiny LSTM mode:
            # - bigger LSTM (128 units instead of normal default 64)
            # - dropout disabled to remove regularization during memorization test
            model = build_lstm_model(
                input_shape=(x.shape[1], x.shape[2]),
                num_classes=9,
                lstm_units=128,
                learning_rate=learning_rate,
                use_dropout=False,
            )
            x_tiny_train = x_tiny
        else:
            # Tiny MLP mode:
            # - flatten each sequence from (90, 30) to 2700 features
            # - no dropout layers
            x_tiny_train = x_tiny.reshape(x_tiny.shape[0], -1)
            model = build_tiny_mlp_model(
                input_dim=x_tiny_train.shape[1],
                num_classes=9,
                learning_rate=learning_rate,
            )

        print("\n=== Model Summary ===")
        model.summary()

        # Diagnostic behavior:
        # - 300 epochs to give model enough time to memorize tiny data
        # - batch size 1 for strongest per-sample fitting pressure
        # - no early stopping (we want to observe full memorization trajectory)
        # - verbose=1 prints loss/accuracy every epoch
        history = model.fit(
            x_tiny_train,
            y_tiny,
            epochs=tiny_epochs,
            batch_size=tiny_batch_size,
            verbose=1,
        )

        save_tiny_overfit_history_for_model(
            history=history,
            reports_dir=reports_dir,
            tiny_model_type=args.tiny_model_type,
        )

        final_train_loss = float(history.history.get("loss", [np.nan])[-1])
        final_train_acc = float(history.history.get("accuracy", [np.nan])[-1])
        tiny_unique, tiny_counts = np.unique(y_tiny, return_counts=True)
        id_to_name = {int(v): k for k, v in label_map.get("label_to_id", {}).items()}

        print("\n=== Tiny Overfit Summary ===")
        print(f"Subset size: {len(y_tiny)}")
        print("Class counts in tiny subset:")
        for class_id, count in zip(tiny_unique, tiny_counts):
            class_name = id_to_name.get(int(class_id), f"class_{int(class_id)}")
            print(f"  - {class_id} ({class_name}): {count}")
        print(f"Final training accuracy: {final_train_acc:.4f}")
        print(f"Final training loss: {final_train_loss:.4f}")
        print(f"Model type used: {args.tiny_model_type}")

        print("\nInterpretation hint:")
        if final_train_acc >= 0.90:
            print(
                "- Very high final training accuracy suggests the current pipeline can "
                "memorize a tiny dataset."
            )
        else:
            print(
                "- Low final training accuracy on this tiny set suggests a likely issue in "
                "preprocessing, representation, or model setup."
            )

        print("\n=== Saved Tiny Overfit Outputs ===")
        print(
            f"- Tiny history CSV: "
            f"{reports_dir / f'tiny_overfit_{args.tiny_model_type}_history.csv'}"
        )
        print(
            f"- Tiny history plot: "
            f"{reports_dir / f'tiny_overfit_{args.tiny_model_type}_history.png'}"
        )

        tiny_metrics = {
            "status": "success",
            "run_name": run_name if run_name else None,
            "experiment_name": run_name if run_name else None,
            "tiny_overfit": True,
            "tiny_model_type": args.tiny_model_type,
            "model_type": args.tiny_model_type,
            "dataset_shape": list(x.shape),
            "tiny_subset_shape": {"x": list(x_tiny.shape), "y": list(y_tiny.shape)},
            "tiny_subset_distribution": compute_label_distribution(y_tiny, label_map),
            "epochs_requested": tiny_epochs,
            "epochs_run": len(history.history.get("loss", [])),
            "final_train_accuracy": final_train_acc,
            "final_train_loss": final_train_loss,
            "history_csv": str(reports_dir / f"tiny_overfit_{args.tiny_model_type}_history.csv"),
            "history_png": str(reports_dir / f"tiny_overfit_{args.tiny_model_type}_history.png"),
        }
        write_json(reports_dir / "metrics.json", tiny_metrics)
        return

    print("\n=== Full Dataset Training Mode ===")
    print(f"Model type used: {args.model_type}")

    train_idx, val_idx, test_idx = load_or_create_split_indices(
        y=y,
        splits_dir=splits_dir,
        random_state=random_seed,
        force_resplit=args.force_resplit,
    )

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print("\n=== Split Sizes ===")
    print(f"Train: {len(train_idx)}")
    print(f"Validation: {len(val_idx)}")
    print(f"Test: {len(test_idx)}")

    print_split_summary("Train", y_train, label_map)
    print_split_summary("Validation", y_val, label_map)
    print_split_summary("Test", y_test, label_map)
    split_copy_paths: dict[str, str] = {}
    if args.save_split_copy and run_dir is not None:
        split_copy_dir = run_dir / "splits"
        split_copy_dir.mkdir(parents=True, exist_ok=True)
        source_split_paths = {
            "train": splits_dir / "train_indices.npy",
            "val": splits_dir / "val_indices.npy",
            "test": splits_dir / "test_indices.npy",
        }
        for split_name, src_path in source_split_paths.items():
            dst_path = split_copy_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            split_copy_paths[split_name] = str(dst_path)

    checkpoint_monitor = args.checkpoint_monitor
    monitor_mode = "max" if checkpoint_monitor.endswith("accuracy") else "min"
    checkpoint_callback: keras.callbacks.ModelCheckpoint | None = None
    input_representation = "pose_only"
    model_dataset_shape: list[int] = list(x.shape)

    if args.model_type == "lstm":
        epochs = int(get_training_value(config, "epochs", 20))

        # Sequence-model defaults are intentionally more conservative/stable.
        # "Unless explicitly configured otherwise":
        # - prefer sequence_* keys when present
        # - otherwise fallback to generic training keys when present
        # - otherwise use sequence-safe defaults (batch=16, lr=1e-4)
        if has_training_value(config, "sequence_batch_size"):
            sequence_batch_size = int(get_training_value(config, "sequence_batch_size", 16))
        elif has_training_value(config, "batch_size"):
            sequence_batch_size = int(get_training_value(config, "batch_size", 16))
        else:
            sequence_batch_size = 16

        if has_training_value(config, "sequence_learning_rate"):
            sequence_learning_rate = float(get_training_value(config, "sequence_learning_rate", 1e-4))
        elif has_training_value(config, "learning_rate"):
            sequence_learning_rate = float(get_training_value(config, "learning_rate", 1e-4))
        else:
            sequence_learning_rate = 1e-4

        # Standardize sequence tensors using train-split-only statistics.
        feature_mean, feature_std = compute_sequence_normalization_stats(x_train)
        x_train_model = standardize_sequence_data(x_train, feature_mean, feature_std)
        x_val_model = standardize_sequence_data(x_val, feature_mean, feature_std)
        x_test_model = standardize_sequence_data(x_test, feature_mean, feature_std)
        model_dataset_shape = [int(x.shape[0]), int(x_train_model.shape[1]), int(x_train_model.shape[2])]
        mean_path, std_path = save_sequence_normalization_stats(reports_dir, feature_mean, feature_std)

        model = build_lstm_model(
            input_shape=(x.shape[1], x.shape[2]),
            num_classes=9,
            lstm_units=lstm_units,
            learning_rate=sequence_learning_rate,
            use_masking=False,
        )
        checkpoint_path = checkpoints_dir / "best_lstm.keras"
        filename_prefix = ""
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=checkpoint_monitor,
            mode=monitor_mode,
            save_best_only=True,
        )
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=checkpoint_monitor,
                mode=monitor_mode,
                patience=10,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            ),
            checkpoint_callback,
        ]
        batch_size = sequence_batch_size

        print("\n=== Sequence Model Details ===")
        print("Model type: lstm")
        print("Representation used: original sequence positions")
        print("Standardization applied: yes (train split only)")
        print(f"Original input shape: {x.shape[1:]}")
        print(f"Motion-aware shape: not used for this mode")
        print(f"Sequence batch size: {sequence_batch_size}")
        print(f"Sequence learning rate: {sequence_learning_rate}")
        print(f"Normalization stats saved to: {mean_path} and {std_path}")
        input_representation = "pose_only"
        model_dataset_shape = [int(x.shape[0]), int(x_train_model.shape[1]), int(x_train_model.shape[2])]
    elif args.model_type == "mlp":
        epochs = int(get_training_value(config, "epochs", 100))
        # MLP baseline:
        # - flatten each sample from (90, 30) into 2700 features
        # - train and evaluate with the exact same split indices as LSTM
        x_train_model = x_train.reshape(x_train.shape[0], -1)
        x_val_model = x_val.reshape(x_val.shape[0], -1)
        x_test_model = x_test.reshape(x_test.shape[0], -1)
        model_dataset_shape = [int(x.shape[0]), int(x_train_model.shape[1])]
        print(
            f"MLP flattening: each sample {x.shape[1:]} -> "
            f"{x_train_model.shape[1]} features"
        )
        model = build_mlp_model(
            input_dim=x_train_model.shape[1],
            num_classes=9,
            learning_rate=learning_rate,
        )
        checkpoint_path = checkpoints_dir / "best_mlp.keras"
        filename_prefix = "mlp_"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=checkpoint_monitor,
            mode=monitor_mode,
            save_best_only=True,
        )
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=checkpoint_monitor,
                mode=monitor_mode,
                patience=10,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            ),
            checkpoint_callback,
        ]
    elif args.model_type == "mlp_motion":
        epochs = int(get_training_value(config, "epochs", 100))
        x_motion = build_motion_aware_sequences(x)
        x_train_motion = x_motion[train_idx]
        x_val_motion = x_motion[val_idx]
        x_test_motion = x_motion[test_idx]

        feature_mean, feature_std = compute_sequence_normalization_stats(x_train_motion)
        x_train_motion = standardize_sequence_data(x_train_motion, feature_mean, feature_std)
        x_val_motion = standardize_sequence_data(x_val_motion, feature_mean, feature_std)
        x_test_motion = standardize_sequence_data(x_test_motion, feature_mean, feature_std)
        mean_path, std_path = save_sequence_normalization_stats(reports_dir, feature_mean, feature_std)

        x_train_model = x_train_motion.reshape(x_train_motion.shape[0], -1)
        x_val_model = x_val_motion.reshape(x_val_motion.shape[0], -1)
        x_test_model = x_test_motion.reshape(x_test_motion.shape[0], -1)
        input_representation = "pose_plus_delta"
        model_dataset_shape = [int(x.shape[0]), int(x_train_model.shape[1])]

        print("\n=== Input Representation ===")
        print("Representation: position + delta (motion-aware), then flattened for MLP")
        print(f"Original per-sample shape: {x.shape[1:]}")
        print(f"Motion-aware per-sample shape: {x_train_motion.shape[1:]}")
        print(f"Flattened per-sample features: {x_train_model.shape[1]}")
        print("Standardization applied: yes (train split only)")
        print(f"Normalization stats saved to: {mean_path} and {std_path}")

        model = build_mlp_motion_model(
            input_dim=x_train_model.shape[1],
            num_classes=9,
            learning_rate=learning_rate,
        )
        checkpoint_path = checkpoints_dir / "best_mlp_motion.keras"
        filename_prefix = "mlp_motion_"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=checkpoint_monitor,
            mode=monitor_mode,
            save_best_only=True,
        )
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=checkpoint_monitor,
                mode=monitor_mode,
                patience=10,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            ),
            checkpoint_callback,
        ]
    else:
        # Motion-aware sequence baseline (LSTM or GRU):
        # - keeps original processed X.npy untouched
        # - derives explicit motion (frame-to-frame deltas) during training
        # - standardizes using train-only stats
        epochs = 100
        x_motion = build_motion_aware_sequences(x)
        x_train_model = x_motion[train_idx]
        x_val_model = x_motion[val_idx]
        x_test_model = x_motion[test_idx]

        # Sequence-model defaults are intentionally more conservative/stable.
        if has_training_value(config, "sequence_batch_size"):
            sequence_batch_size = int(get_training_value(config, "sequence_batch_size", 16))
        elif has_training_value(config, "batch_size"):
            sequence_batch_size = int(get_training_value(config, "batch_size", 16))
        else:
            sequence_batch_size = 16

        if has_training_value(config, "sequence_learning_rate"):
            sequence_learning_rate = float(get_training_value(config, "sequence_learning_rate", 1e-4))
        elif has_training_value(config, "learning_rate"):
            sequence_learning_rate = float(get_training_value(config, "learning_rate", 1e-4))
        else:
            sequence_learning_rate = 1e-4

        feature_mean, feature_std = compute_sequence_normalization_stats(x_train_model)
        x_train_model = standardize_sequence_data(x_train_model, feature_mean, feature_std)
        x_val_model = standardize_sequence_data(x_val_model, feature_mean, feature_std)
        x_test_model = standardize_sequence_data(x_test_model, feature_mean, feature_std)
        mean_path, std_path = save_sequence_normalization_stats(reports_dir, feature_mean, feature_std)

        print("\n=== Input Representation ===")
        print("Representation: position + delta (motion-aware)")
        print(f"Original per-sample shape: {x.shape[1:]}")
        print(f"Motion-aware per-sample shape: {x_train_model.shape[1:]}")
        print("Standardization applied: yes (train split only)")
        print(f"Sequence batch size: {sequence_batch_size}")
        print(f"Sequence learning rate: {sequence_learning_rate}")
        print(f"Normalization stats saved to: {mean_path} and {std_path}")
        input_representation = "pose_plus_delta"
        model_dataset_shape = [int(x.shape[0]), int(x_train_model.shape[1]), int(x_train_model.shape[2])]

        if args.model_type == "lstm_motion":
            model = build_lstm_motion_model(
                input_shape=(x_train_model.shape[1], x_train_model.shape[2]),
                num_classes=9,
                learning_rate=sequence_learning_rate,
            )
            checkpoint_path = checkpoints_dir / "best_lstm_motion.keras"
            filename_prefix = "lstm_motion_"
        else:
            model = build_gru_motion_model(
                input_shape=(x_train_model.shape[1], x_train_model.shape[2]),
                num_classes=9,
                learning_rate=sequence_learning_rate,
            )
            checkpoint_path = checkpoints_dir / "best_gru_motion.keras"
            filename_prefix = "gru_motion_"

        print(f"Model type: {args.model_type}")
        print("Masking used: no (after standardization, zero no longer means missing)")

        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=checkpoint_monitor,
            mode=monitor_mode,
            save_best_only=True,
        )
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor=checkpoint_monitor,
                mode=monitor_mode,
                patience=10,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-6,
            ),
            checkpoint_callback,
        ]
        batch_size = sequence_batch_size

    print("\n=== Model Summary ===")
    model.summary()
    print("\n=== Training Configuration ===")
    print(f"Max epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Checkpoint / EarlyStopping monitor: {checkpoint_monitor} (mode={monitor_mode})")
    print("EarlyStopping restore_best_weights: True")

    history = model.fit(
        x_train_model,
        y_train,
        validation_data=(x_val_model, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    epochs_ran = len(history.history.get("loss", []))
    early_stopped = epochs_ran < epochs
    val_loss_history = history.history.get("val_loss", [])
    val_acc_history = history.history.get("val_accuracy", [])
    best_val_loss_epoch = int(np.argmin(val_loss_history) + 1) if val_loss_history else None
    best_val_acc_epoch = int(np.argmax(val_acc_history) + 1) if val_acc_history else None
    monitor_history = history.history.get(checkpoint_monitor, [])
    if monitor_history:
        best_monitor_epoch = (
            int(np.argmax(monitor_history) + 1)
            if monitor_mode == "max"
            else int(np.argmin(monitor_history) + 1)
        )
    else:
        best_monitor_epoch = None

    restored_from_checkpoint = False
    restored_checkpoint_source = "early_stopping_in_memory_best_weights"
    if checkpoint_path.exists():
        model = keras.models.load_model(checkpoint_path)
        restored_from_checkpoint = True
        restored_checkpoint_source = str(checkpoint_path)

    train_loss, train_acc = model.evaluate(x_train_model, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(x_val_model, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(x_test_model, y_test, verbose=0)

    y_pred_probs = model.predict(x_test_model, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    save_history(history, reports_dir, filename_prefix=filename_prefix)
    save_test_reports(
        y_test=y_test,
        y_pred=y_pred,
        y_pred_probs=y_pred_probs,
        label_map=label_map,
        reports_dir=reports_dir,
        test_indices=test_idx,
        metadata_df=metadata_df,
        filename_prefix=filename_prefix,
    )

    print("\n=== Final Metrics ===")
    print(f"Validation accuracy: {val_acc:.4f} (loss: {val_loss:.4f})")
    print(f"Test accuracy: {test_acc:.4f} (loss: {test_loss:.4f})")
    print("\n=== Training Run Summary ===")
    print(f"Epochs actually run: {epochs_ran}/{epochs}")
    print(f"EarlyStopping triggered: {'yes' if early_stopped else 'no'}")
    if best_val_loss_epoch is not None:
        print(f"Best val_loss epoch: {best_val_loss_epoch}")
    if best_val_acc_epoch is not None:
        print(f"Best val_accuracy epoch: {best_val_acc_epoch}")
    if best_monitor_epoch is not None:
        print(f"Best {checkpoint_monitor} epoch: {best_monitor_epoch}")
    print(f"Final restored checkpoint source: {restored_checkpoint_source}")

    print("\n=== Saved Outputs ===")
    print(f"- Best checkpoint: {checkpoint_path}")
    print(f"- Training history CSV: {reports_dir / f'{filename_prefix}training_history.csv'}")
    print(f"- Training history plot: {reports_dir / f'{filename_prefix}training_history.png'}")
    print(f"- Classification report: {reports_dir / f'{filename_prefix}classification_report.txt'}")
    print(f"- Confusion matrix CSV: {reports_dir / f'{filename_prefix}confusion_matrix.csv'}")
    print(f"- Confusion matrix PNG: {reports_dir / f'{filename_prefix}confusion_matrix.png'}")
    print(f"- Test predictions CSV: {reports_dir / f'{filename_prefix}test_predictions.csv'}")
    print(f"- Split indices: {splits_dir / 'train_indices.npy'}, {splits_dir / 'val_indices.npy'}, {splits_dir / 'test_indices.npy'}")
    if split_copy_paths:
        print(f"- Split copy dir: {run_dir / 'splits'}")

    metrics_payload = {
        "status": "success",
        "run_name": run_name if run_name else None,
        "experiment_name": run_name if run_name else None,
        "tiny_overfit": False,
        "model_type": args.model_type,
        "input_representation": input_representation,
        "dataset_shape": model_dataset_shape,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "split_label_distribution": {
            "train": compute_label_distribution(y_train, label_map),
            "val": compute_label_distribution(y_val, label_map),
            "test": compute_label_distribution(y_test, label_map),
        },
        "checkpoint_monitor": checkpoint_monitor,
        "checkpoint_monitor_mode": monitor_mode,
        "early_stopping_restore_best_weights": True,
        "best_epoch_by_monitor": best_monitor_epoch,
        "best_epoch_by_val_loss": best_val_loss_epoch,
        "best_epoch_by_val_accuracy": best_val_acc_epoch,
        "restored_from_checkpoint_file": restored_from_checkpoint,
        "restored_checkpoint_source": restored_checkpoint_source,
        "epochs_requested": int(epochs),
        "epochs_run": int(epochs_ran),
        "final_train_accuracy": float(train_acc),
        "final_val_accuracy": float(val_acc),
        "final_test_accuracy": float(test_acc),
        "final_train_loss": float(train_loss),
        "final_val_loss": float(val_loss),
        "final_test_loss": float(test_loss),
        "best_checkpoint_path": str(checkpoint_path),
        "history_csv": str(reports_dir / f"{filename_prefix}training_history.csv"),
        "history_png": str(reports_dir / f"{filename_prefix}training_history.png"),
        "classification_report": str(reports_dir / f"{filename_prefix}classification_report.txt"),
        "confusion_matrix_csv": str(reports_dir / f"{filename_prefix}confusion_matrix.csv"),
        "confusion_matrix_png": str(reports_dir / f"{filename_prefix}confusion_matrix.png"),
        "predictions_csv": str(reports_dir / f"{filename_prefix}test_predictions.csv"),
        "split_copy_paths": split_copy_paths,
    }
    write_json(reports_dir / "metrics.json", metrics_payload)


if __name__ == "__main__":
    main()
