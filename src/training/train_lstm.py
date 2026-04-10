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

import json
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
) -> keras.Model:
    """Build the requested simple baseline LSTM classifier."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Masking(mask_value=0.0),
            keras.layers.LSTM(lstm_units),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_history(history: keras.callbacks.History, reports_dir: Path) -> None:
    """Save training history to CSV and PNG plot."""
    history_df = pd.DataFrame(history.history)
    history_csv_path = reports_dir / "training_history.csv"
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
    plt.savefig(reports_dir / "training_history.png", dpi=150)
    plt.close()


def save_test_reports(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict[str, Any],
    reports_dir: Path,
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
    (reports_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(reports_dir / "confusion_matrix.csv")

    predictions_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "true_label": [id_to_name.get(int(v), f"class_{int(v)}") for v in y_test],
            "pred_label": [id_to_name.get(int(v), f"class_{int(v)}") for v in y_pred],
        }
    )
    predictions_df.to_csv(reports_dir / "test_predictions.csv", index=False)


def main() -> None:
    """Run full first-baseline training flow."""
    config = load_config(Path("configs/config.yaml"))

    random_seed = int(get_training_value(config, "random_seed", 42))
    epochs = int(get_training_value(config, "epochs", 20))
    batch_size = int(get_training_value(config, "batch_size", 32))
    learning_rate = float(get_training_value(config, "learning_rate", 0.001))
    lstm_units = int(get_training_value(config, "lstm_units", 64))

    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    processed_dir = Path("data/processed")
    splits_dir = Path("data/splits")
    checkpoints_dir = Path("models/checkpoints")
    reports_dir = Path("models/reports")

    splits_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    x, y, label_map = load_processed_data(processed_dir)
    print_dataset_summary(x, y, label_map)

    if x.ndim != 3:
        raise ValueError(f"Expected X shape (samples, timesteps, features), got {x.shape}")

    train_idx, val_idx, test_idx = split_data_stratified(x, y, random_state=random_seed)

    np.save(splits_dir / "train_indices.npy", train_idx)
    np.save(splits_dir / "val_indices.npy", val_idx)
    np.save(splits_dir / "test_indices.npy", test_idx)

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

    # Baseline architecture requested in the task.
    model = build_lstm_model(
        input_shape=(x.shape[1], x.shape[2]),
        num_classes=9,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
    )

    print("\n=== Model Summary ===")
    model.summary()

    checkpoint_path = checkpoints_dir / "best_lstm.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    save_history(history, reports_dir)
    save_test_reports(y_test=y_test, y_pred=y_pred, label_map=label_map, reports_dir=reports_dir)

    print("\n=== Final Metrics ===")
    print(f"Validation accuracy: {val_acc:.4f} (loss: {val_loss:.4f})")
    print(f"Test accuracy: {test_acc:.4f} (loss: {test_loss:.4f})")

    print("\n=== Saved Outputs ===")
    print(f"- Best checkpoint: {checkpoint_path}")
    print(f"- Training history CSV: {reports_dir / 'training_history.csv'}")
    print(f"- Training history plot: {reports_dir / 'training_history.png'}")
    print(f"- Classification report: {reports_dir / 'classification_report.txt'}")
    print(f"- Confusion matrix CSV: {reports_dir / 'confusion_matrix.csv'}")
    print(f"- Test predictions CSV: {reports_dir / 'test_predictions.csv'}")
    print(f"- Split indices: {splits_dir / 'train_indices.npy'}, {splits_dir / 'val_indices.npy'}, {splits_dir / 'test_indices.npy'}")


if __name__ == "__main__":
    main()
