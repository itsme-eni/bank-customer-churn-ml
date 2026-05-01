"""Common utility helpers such as logging setup and directory creation."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: str = "INFO", fmt: str | None = None) -> None:
    """Configure application-level logging.

    Args:
        level: Logging level (for example: INFO, DEBUG).
        fmt: Optional custom log format.
    """
    # Normalize level text and fall back to INFO if value is invalid.
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt or "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Directory path object.
    """
    # Create parent directories as needed; no-op if directory already exists.
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
