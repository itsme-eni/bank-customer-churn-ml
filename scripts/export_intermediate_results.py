"""Export intermediate artifacts for completed project phases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rich.table import Table

# Resolve project root dynamically so script runs from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Add src folder to Python import path for local package imports.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bank_churn_ml.config import load_config
from bank_churn_ml.data_loading import EXPECTED_COLUMNS, load_raw_data
from bank_churn_ml.features import add_engineered_features
from bank_churn_ml.preprocessing import prepare_training_matrices
from bank_churn_ml.utils import setup_logging
from bank_churn_ml.validation import generate_data_quality_report, validate_binary_target, validate_required_columns


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    # Single config flag keeps this script easy to run from terminal.
    parser = argparse.ArgumentParser(description="Export intermediate results for completed phases")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args()


def _save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> Path:
    """Save DataFrame and ensure parent directory exists."""
    # Utility wrapper used to standardize folder creation + CSV writing.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path


def export_intermediate_results(config: dict) -> list[Path]:
    """Generate artifacts for completed phases (1 to 4)."""
    # Read key config values used by multiple phases.
    raw_data_path = PROJECT_ROOT / Path(config["paths"]["raw_data"])
    processed_data_path = PROJECT_ROOT / Path(config["paths"]["processed_data"])
    metrics_dir = PROJECT_ROOT / Path(config["paths"]["metrics_dir"])
    intermediate_dir = metrics_dir / "intermediate"
    target_column = config["project"]["target_column"]
    id_column = config["project"]["id_column"]

    split_cfg = config.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    stratify = bool(split_cfg.get("stratify", True))
    random_state = int(config["project"].get("random_state", 42))

    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: data loading and validation artifacts.
    raw_df = load_raw_data(raw_data_path)
    validate_required_columns(raw_df, EXPECTED_COLUMNS)
    validate_binary_target(raw_df, target_column)
    quality = generate_data_quality_report(raw_df)

    phase1_summary = {
        "phase": "phase_1_data_loading_validation",
        "rows": quality.row_count,
        "columns": quality.column_count,
        "duplicate_rows": quality.duplicate_rows,
        "missing_cells_total": int(sum(quality.missing_by_column.values())),
        "target_column": target_column,
        "target_positive_rate": float(raw_df[target_column].mean()),
    }

    saved_paths: list[Path] = []

    phase1_json = intermediate_dir / "phase_1_validation_summary.json"
    phase1_json.write_text(json.dumps(phase1_summary, indent=2), encoding="utf-8")
    saved_paths.append(phase1_json)

    phase1_missing = pd.DataFrame(
        [{"column": key, "missing_count": value} for key, value in quality.missing_by_column.items()]
    )
    saved_paths.append(_save_dataframe(phase1_missing, intermediate_dir / "phase_1_missing_by_column.csv"))

    # Phase 2: EDA overview artifacts (reproducible tables).
    working_df = load_raw_data(processed_data_path if processed_data_path.exists() else raw_data_path)
    if "balance_to_salary_ratio" not in working_df.columns:
        working_df = add_engineered_features(working_df)

    phase2_overview = pd.DataFrame(
        [
            {
                "phase": "phase_2_eda_overview",
                "rows": int(working_df.shape[0]),
                "columns": int(working_df.shape[1]),
                "duplicate_rows": int(working_df.duplicated().sum()),
                "churn_rate": float(working_df[target_column].mean()),
            }
        ]
    )
    saved_paths.append(_save_dataframe(phase2_overview, intermediate_dir / "phase_2_eda_overview.csv"))

    # Phase 3: feature engineering artifact coverage.
    engineered_columns = [
        "balance_to_salary_ratio",
        "products_per_tenure",
        "age_group",
        "tenure_group",
        "high_value_customer",
    ]
    phase3_rows: list[dict[str, object]] = []
    for column in engineered_columns:
        if column in working_df.columns:
            phase3_rows.append(
                {
                    "feature": column,
                    "dtype": str(working_df[column].dtype),
                    "missing_count": int(working_df[column].isna().sum()),
                    "unique_values": int(working_df[column].nunique(dropna=True)),
                }
            )
    phase3_df = pd.DataFrame(phase3_rows)
    saved_paths.append(_save_dataframe(phase3_df, intermediate_dir / "phase_3_feature_engineering_summary.csv"))

    # Phase 4: preprocessing and split summary artifacts.
    prepared = prepare_training_matrices(
        working_df,
        target_column=target_column,
        id_column=id_column,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    phase4_summary = pd.DataFrame(
        [
            {
                "phase": "phase_4_preprocessing",
                "x_train_rows": int(prepared["X_train"].shape[0]),
                "x_test_rows": int(prepared["X_test"].shape[0]),
                "x_train_columns": int(prepared["X_train"].shape[1]),
                "categorical_feature_count": int(len(prepared["categorical_features"])),
                "numerical_feature_count": int(len(prepared["numerical_features"])),
                "customer_id_present_in_features": str(id_column in prepared["X_train"].columns),
                "train_churn_rate": float(prepared["y_train"].mean()),
                "test_churn_rate": float(prepared["y_test"].mean()),
            }
        ]
    )
    saved_paths.append(_save_dataframe(phase4_summary, intermediate_dir / "phase_4_preprocessing_summary.csv"))

    # Create a compact markdown report for quick review in GitHub.
    report_lines = [
        "# Intermediate Results Report",
        "",
        "Generated artifacts for completed phases (1 to 4).",
        "",
        "## Files",
    ]
    for path in saved_paths:
        rel = path.relative_to(PROJECT_ROOT)
        report_lines.append(f"- {rel.as_posix()}")

    report_path = intermediate_dir / "INTERMEDIATE_RESULTS.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    saved_paths.append(report_path)

    return saved_paths


def main() -> None:
    """Run intermediate export pipeline."""
    # 1) Parse options and load config.
    args = parse_args()
    config = load_config(args.config)
    logger, console = setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        fmt=config.get("logging", {}).get("format"),
        script_name="export_intermediate_results",
        log_dir=PROJECT_ROOT / Path(config.get("paths", {}).get("logs_dir", "reports/logs")),
    )

    # 2) Generate intermediate artifacts and summarize in console table.
    logger.info("Exporting intermediate artifacts for completed phases")
    saved_paths = export_intermediate_results(config)

    table = Table(title="Intermediate Phase Artifacts")
    table.add_column("#", justify="right", style="cyan")
    table.add_column("Artifact", style="green")
    for idx, path in enumerate(saved_paths, start=1):
        table.add_row(str(idx), str(path.relative_to(PROJECT_ROOT)))

    console.print(table)
    logger.info("Saved %s artifacts", len(saved_paths))


if __name__ == "__main__":
    main()
