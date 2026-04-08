"""Project-relative path helpers."""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return absolute path to repository root."""
    return PROJECT_ROOT


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a path relative to project root."""
    return PROJECT_ROOT / Path(relative_path)


def load_paths_config(config_file: str = "configs/paths.yaml") -> dict[str, Any]:
    """Load paths configuration YAML."""
    cfg_path = resolve_path(config_file)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
