"""Helpers for gesture label configuration and mappings."""

from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_gesture_config(config_path: Path) -> dict:
    """Load gesture YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_active_labels(config_path: Path) -> List[str]:
    """Return labels for the active target gesture mode."""
    cfg = load_gesture_config(config_path)
    target_mode = cfg["target_mode"]
    return list(cfg["gesture_sets"][target_mode])


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build name->id and id->name mappings from ordered labels."""
    name_to_id = {name: idx for idx, name in enumerate(labels)}
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    return name_to_id, id_to_name
