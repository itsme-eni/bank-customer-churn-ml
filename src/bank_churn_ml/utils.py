"""Common utility helpers such as logging setup and directory creation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None


def setup_logging(
    level: str = "INFO",
    fmt: str | None = None,
    script_name: str = "app",
    tag: str = "",
    log_dir: Path | str | None = None,
) -> tuple[logging.Logger, Console]:
    """Configure logging with rich console output and file output.

    Args:
        level: Logging level (for example: INFO, DEBUG).
        fmt: Optional custom log format.
        script_name: Base script name used for log file naming.
        tag: Optional suffix appended to the log file name.
        log_dir: Optional log directory path.

    Returns:
        Tuple of configured logger and rich console.
    """
    # Normalize level text and fall back to INFO if value is invalid.
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    # Clear handlers so repeated script runs do not duplicate log lines.
    root_logger.handlers.clear()

    # Use a predictable default logs directory when caller does not provide one.
    if log_dir is None:
        log_dir_path = Path("reports/logs")
    else:
        log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Build file name like prepare_data.log or prepare_data_debug.log.
    suffix = f"_{tag}" if tag else ""
    log_file = log_dir_path / f"{script_name}{suffix}.log"

    console = Console()

    # Prefer RichHandler for colorized logs; use plain StreamHandler as fallback.
    if RichHandler is not None:
        handler: logging.Handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=False,
            show_path=False,
        )
        log_format = fmt or "%(message)s"
    else:
        handler = logging.StreamHandler()
        log_format = fmt or "%(message)s"

    # File handler keeps a durable plain-text log for debugging after runs finish.
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")

    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # Attach both console and file handlers to the root logger.
    root_logger.setLevel(resolved_level)
    root_logger.addHandler(handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(script_name), console


def dataframe_to_rich_table(dataframe: pd.DataFrame, title: str, max_rows: int = 20) -> Table:
    """Convert a pandas DataFrame to a rich Table for terminal display."""
    # Create a terminal-friendly table with column names from the DataFrame.
    table = Table(title=title)
    for column in dataframe.columns:
        table.add_column(str(column), overflow="fold")

    # Limit rows so large DataFrames do not flood the console.
    display_df = dataframe.head(max_rows)
    for _, row in display_df.iterrows():
        table.add_row(*[str(value) for value in row.tolist()])

    return table


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
