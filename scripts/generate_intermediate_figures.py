"""Generate initial EDA figures into reports/figures/intermediate."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

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


def save_eda_summary_tables(dataframe: pd.DataFrame, metrics_dir: Path) -> list[Path]:
    """Build and save compact EDA summary tables for reproducibility."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # One-row overview table with key health metrics.
    overview = pd.DataFrame(
        [
            {
                "rows": int(dataframe.shape[0]),
                "columns": int(dataframe.shape[1]),
                "duplicate_rows": int(dataframe.duplicated().sum()),
                "churn_rate": float(dataframe["churn"].mean()),
            }
        ]
    )
    overview_path = metrics_dir / "eda_overview.csv"
    overview.to_csv(overview_path, index=False)
    saved.append(overview_path)

    # Missing-value counts by column.
    missing = dataframe.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing_path = metrics_dir / "eda_missing_values.csv"
    missing.to_csv(missing_path, index=False)
    saved.append(missing_path)

    # Summary stats for numeric features.
    numeric_summary = dataframe.select_dtypes(include=["number"]).describe().T.reset_index()
    numeric_summary = numeric_summary.rename(columns={"index": "feature"})
    numeric_path = metrics_dir / "eda_numeric_summary.csv"
    numeric_summary.to_csv(numeric_path, index=False)
    saved.append(numeric_path)

    # Counts and churn rates for core categorical features.
    categorical_rows: list[pd.DataFrame] = []
    for column in ["country", "gender", "credit_card", "active_member", "products_number"]:
        grouped = dataframe.groupby(column, as_index=False).agg(
            customer_count=("churn", "size"),
            churn_rate=("churn", "mean"),
        )
        grouped.insert(0, "feature", column)
        grouped = grouped.rename(columns={column: "category_value"})
        categorical_rows.append(grouped)

    categorical_summary = pd.concat(categorical_rows, ignore_index=True)
    categorical_path = metrics_dir / "eda_categorical_summary.csv"
    categorical_summary.to_csv(categorical_path, index=False)
    saved.append(categorical_path)

    return saved


def main() -> None:
    """Run figure generation pipeline."""
    args = parse_args()
    config = load_config(args.config)
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        fmt=config.get("logging", {}).get("format"),
    )
    logger = logging.getLogger(__name__)
    console = Console()

    # Choose source table and output figure folder from config.
    raw_data_path = PROJECT_ROOT / Path(config["paths"]["raw_data"])
    processed_data_path = PROJECT_ROOT / Path(config["paths"]["processed_data"])
    figures_root = PROJECT_ROOT / Path(config["paths"]["figures_dir"])
    metrics_dir = PROJECT_ROOT / Path(config["paths"]["metrics_dir"])
    intermediate_dir = figures_root / "intermediate"

    source_path = processed_data_path if args.use_processed and processed_data_path.exists() else raw_data_path
    logger.info("Loading dataset for plotting from %s", source_path)

    dataframe = load_raw_data(source_path)

    # If plotting from raw data, add engineered columns for richer initial plots.
    if source_path == raw_data_path:
        dataframe = add_engineered_features(dataframe)

    saved_paths = generate_initial_eda_figures(dataframe, intermediate_dir)
    summary_paths = save_eda_summary_tables(dataframe, metrics_dir)

    logger.info("Generated %s initial figures in %s", len(saved_paths), intermediate_dir)
    for path in saved_paths:
        logger.info("Saved figure: %s", path)
    for path in summary_paths:
        logger.info("Saved summary table: %s", path)

    # Print compact artifact summary in a rich table.
    figure_table = Table(title="Intermediate Figures")
    figure_table.add_column("#", justify="right", style="cyan")
    figure_table.add_column("File", style="green")
    for index, path in enumerate(saved_paths, start=1):
        figure_table.add_row(str(index), str(path.relative_to(PROJECT_ROOT)))
    console.print(figure_table)

    summary_table = Table(title="EDA Summary Tables")
    summary_table.add_column("#", justify="right", style="cyan")
    summary_table.add_column("File", style="green")
    for index, path in enumerate(summary_paths, start=1):
        summary_table.add_row(str(index), str(path.relative_to(PROJECT_ROOT)))
    console.print(summary_table)


if __name__ == "__main__":
    main()
