"""Minimal LSTM training scaffold for gesture sequence classification."""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras

from src.preprocessing.build_dataset import build_dataset


def load_config(config_path: Path) -> dict[str, Any]:
    """Load top-level training config from YAML."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_lstm_model(
    input_shape: Tuple[int, int], num_classes: int, lstm_units: int = 64, learning_rate: float = 0.001
) -> keras.Model:
    """Create a simple LSTM classifier model."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(lstm_units),
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


def load_dataset_placeholder() -> Tuple[np.ndarray, np.ndarray]:
    """Load training data via preprocessing placeholder pipeline."""
    return build_dataset()


def run_training(config: dict[str, Any]) -> keras.Model:
    """Run placeholder training flow and return the trained model.

    TODO: Replace placeholder dataset and add real train/val splits.
    """
    x, y = load_dataset_placeholder()
    if x.ndim != 3:
        raise ValueError("Expected x to have shape (samples, timesteps, features).")

    num_classes = int(max(y) + 1) if y.size else 1
    input_shape = (x.shape[1], x.shape[2])

    t_cfg = config.get("training", {})
    model = build_lstm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        lstm_units=int(t_cfg.get("lstm_units", 64)),
        learning_rate=float(t_cfg.get("learning_rate", 0.001)),
    )

    model.fit(
        x,
        y,
        epochs=int(t_cfg.get("epochs", 1)),
        batch_size=int(t_cfg.get("batch_size", 8)),
        verbose=1,
    )
    return model


def main() -> None:
    """CLI entrypoint for scaffold training run."""
    cfg = load_config(Path("configs/config.yaml"))
    tf.random.set_seed(int(cfg.get("training", {}).get("random_seed", 42)))
    model = run_training(cfg)
    model.summary()


if __name__ == "__main__":
    main()
