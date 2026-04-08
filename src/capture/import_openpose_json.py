"""Utilities for importing OpenPose JSON files."""

from pathlib import Path
from typing import Any


def load_openpose_frame(json_path: Path) -> dict[str, Any]:
    """Load one OpenPose JSON frame from disk.

    TODO: Parse and validate keypoint schema used by this project.
    """
    return {"path": str(json_path), "people": []}
