"""Common utility helpers such as logging setup and directory creation."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None


def setup_logging(level: str = "INFO", fmt: str | None = None) -> None:
    """Configure application-level logging.

    Args:
        level: Logging level (for example: INFO, DEBUG).
        fmt: Optional custom log format.
    """
    # Normalize level text and fall back to INFO if value is invalid.
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Prefer RichHandler for colorized logs; use plain StreamHandler as fallback.
    if RichHandler is not None:
        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        log_format = fmt or "%(message)s"
    else:
        handler = logging.StreamHandler()
        log_format = fmt or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    root_logger.setLevel(resolved_level)
    root_logger.addHandler(handler)


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
