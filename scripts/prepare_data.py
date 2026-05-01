"""Prepare raw data for EDA and modeling."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Resolve project root dynamically so script can be run from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Add src to import path for local package imports without installation.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bank_churn_ml.config import load_config
from bank_churn_ml.data_loading import EXPECTED_COLUMNS, load_raw_data, save_dataframe
from bank_churn_ml.features import add_engineered_features
from bank_churn_ml.utils import setup_logging
from bank_churn_ml.validation import (
    generate_data_quality_report,
    validate_binary_target,
    validate_required_columns,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare raw churn dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    """Run data preparation pipeline entry point."""
    args = parse_args()
    config = load_config(args.config)
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        fmt=config.get("logging", {}).get("format"),
    )

    logger = logging.getLogger(__name__)

    # Read key pipeline settings from YAML config.
    raw_data_path = Path(config["paths"]["raw_data"])
    processed_data_path = Path(config["paths"]["processed_data"])
    target_column = config["project"]["target_column"]

    logger.info("Loading raw dataset from %s", raw_data_path)
    dataframe = load_raw_data(raw_data_path)

    # Validate schema and target assumptions before feature creation.
    validate_required_columns(dataframe, EXPECTED_COLUMNS)
    validate_binary_target(dataframe, target_column)

    quality_report = generate_data_quality_report(dataframe)
    logger.info(
        "Raw data loaded: rows=%s cols=%s duplicates=%s",
        quality_report.row_count,
        quality_report.column_count,
        quality_report.duplicate_rows,
    )

    # Add engineered features and persist final processed table.
    engineered_dataframe = add_engineered_features(dataframe)
    saved_path = save_dataframe(engineered_dataframe, processed_data_path)
    logger.info("Saved processed dataset to %s", saved_path)


if __name__ == "__main__":
    main()
