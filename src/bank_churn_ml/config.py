"""Configuration loading utilities for the churn project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path | str) -> dict[str, Any]:
    """Load YAML configuration from disk.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    # Convert incoming string/path-like value into a Path object.
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Parse YAML into native Python structures (dict/list/etc.).
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Enforce the expected top-level config shape.
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level mapping")

    # Return validated config so callers can rely on dictionary access patterns.
    return config
