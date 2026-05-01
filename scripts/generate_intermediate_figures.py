"""Generate initial EDA figures into reports/figures/intermediate."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Resolve project root dynamically so script runs from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Add src folder to Python import path for local package imports.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bank_churn_ml.config import load_config
from bank_churn_ml.data_loading import load_raw_data
from bank_churn_ml.features import add_engineered_features
from bank_churn_ml.utils import setup_logging
from bank_churn_ml.visualization import generate_initial_eda_figures


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate intermediate EDA figures")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--use-processed",
        action="store_true",
        help="Use processed dataset if available instead of raw dataset",
    )
    return parser.parse_args()


def main() -> None:
    """Run figure generation pipeline."""
    args = parse_args()
    config = load_config(args.config)
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        fmt=config.get("logging", {}).get("format"),
    )
    logger = logging.getLogger(__name__)

    # Choose source table and output figure folder from config.
    raw_data_path = PROJECT_ROOT / Path(config["paths"]["raw_data"])
    processed_data_path = PROJECT_ROOT / Path(config["paths"]["processed_data"])
    figures_root = PROJECT_ROOT / Path(config["paths"]["figures_dir"])
    intermediate_dir = figures_root / "intermediate"

    source_path = processed_data_path if args.use_processed and processed_data_path.exists() else raw_data_path
    logger.info("Loading dataset for plotting from %s", source_path)

    dataframe = load_raw_data(source_path)

    # If plotting from raw data, add engineered columns for richer initial plots.
    if source_path == raw_data_path:
        dataframe = add_engineered_features(dataframe)

    saved_paths = generate_initial_eda_figures(dataframe, intermediate_dir)

    logger.info("Generated %s initial figures in %s", len(saved_paths), intermediate_dir)
    for path in saved_paths:
        logger.info("Saved figure: %s", path)


if __name__ == "__main__":
    main()
