"""Model export helpers for deployment to inference/Unity bridge."""

from pathlib import Path


def export_placeholder(model_path: Path) -> None:
    """Placeholder export function.

    TODO: Support SavedModel/Keras export and metadata packaging.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[TODO] Export model artifact to {model_path}")
