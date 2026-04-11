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
    use_dropout: bool = True,
) -> keras.Model:
    """Build the requested simple baseline LSTM classifier."""
    model_layers: list[keras.layers.Layer] = [
        keras.layers.Input(shape=input_shape),
        keras.layers.Masking(mask_value=0.0),
        keras.layers.LSTM(lstm_units),
    ]

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
        choices=["lstm", "mlp"],
        help=(
            "Model architecture for normal full-dataset training mode. "
            "Use 'lstm' (default) or 'mlp'."
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
    return parser.parse_args()


def save_test_reports(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict[str, Any],
    reports_dir: Path,
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

    predictions_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "true_label": [id_to_name.get(int(v), f"class_{int(v)}") for v in y_test],
            "pred_label": [id_to_name.get(int(v), f"class_{int(v)}") for v in y_pred],
        }
    )
    predictions_df.to_csv(reports_dir / f"{filename_prefix}test_predictions.csv", index=False)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load existing split indices when available; otherwise create and save them."""
    train_idx_path = splits_dir / "train_indices.npy"
    val_idx_path = splits_dir / "val_indices.npy"
    test_idx_path = splits_dir / "test_indices.npy"

    split_files_exist = train_idx_path.exists() and val_idx_path.exists() and test_idx_path.exists()

    if split_files_exist:
        print("\nUsing existing saved split indices from data/splits for fair comparison.")
        train_idx = np.load(train_idx_path)
        val_idx = np.load(val_idx_path)
        test_idx = np.load(test_idx_path)
    else:
        print("\nNo saved split indices found. Creating new 70/15/15 stratified split.")
        # We pass x as a placeholder array because the split helper uses y-based stratification.
        x_placeholder = np.zeros((len(y), 1))
        train_idx, val_idx, test_idx = split_data_stratified(
            x=x_placeholder,
            y=y,
            random_state=random_state,
        )

        np.save(train_idx_path, train_idx)
        np.save(val_idx_path, val_idx)
        np.save(test_idx_path, test_idx)

    return train_idx, val_idx, test_idx

def main() -> None:
    """Run full first-baseline training flow."""
    args = parse_args()

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
        return

    print("\n=== Full Dataset Training Mode ===")
    print(f"Model type used: {args.model_type}")

    train_idx, val_idx, test_idx = load_or_create_split_indices(
        y=y,
        splits_dir=splits_dir,
        random_state=random_seed,
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

    if args.model_type == "lstm":
        # Keep existing LSTM behavior unchanged in normal mode.
        model = build_lstm_model(
            input_shape=(x.shape[1], x.shape[2]),
            num_classes=9,
            lstm_units=lstm_units,
            learning_rate=learning_rate,
        )
        x_train_model, x_val_model, x_test_model = x_train, x_val, x_test
        checkpoint_path = checkpoints_dir / "best_lstm.keras"
        filename_prefix = ""
    else:
        # MLP baseline:
        # - flatten each sample from (90, 30) into 2700 features
        # - train and evaluate with the exact same split indices as LSTM
        x_train_model = x_train.reshape(x_train.shape[0], -1)
        x_val_model = x_val.reshape(x_val.shape[0], -1)
        x_test_model = x_test.reshape(x_test.shape[0], -1)
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

    print("\n=== Model Summary ===")
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        x_train_model,
        y_train,
        validation_data=(x_val_model, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(x_val_model, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(x_test_model, y_test, verbose=0)

    y_pred_probs = model.predict(x_test_model, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    save_history(history, reports_dir, filename_prefix=filename_prefix)
    save_test_reports(
        y_test=y_test,
        y_pred=y_pred,
        label_map=label_map,
        reports_dir=reports_dir,
        filename_prefix=filename_prefix,
    )

    print("\n=== Final Metrics ===")
    print(f"Validation accuracy: {val_acc:.4f} (loss: {val_loss:.4f})")
    print(f"Test accuracy: {test_acc:.4f} (loss: {test_loss:.4f})")

    print("\n=== Saved Outputs ===")
    print(f"- Best checkpoint: {checkpoint_path}")
    print(f"- Training history CSV: {reports_dir / f'{filename_prefix}training_history.csv'}")
    print(f"- Training history plot: {reports_dir / f'{filename_prefix}training_history.png'}")
    print(f"- Classification report: {reports_dir / f'{filename_prefix}classification_report.txt'}")
    print(f"- Confusion matrix CSV: {reports_dir / f'{filename_prefix}confusion_matrix.csv'}")
    print(f"- Test predictions CSV: {reports_dir / f'{filename_prefix}test_predictions.csv'}")
    print(f"- Split indices: {splits_dir / 'train_indices.npy'}, {splits_dir / 'val_indices.npy'}, {splits_dir / 'test_indices.npy'}")


if __name__ == "__main__":
    main()
