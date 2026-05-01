"""Train and compare candidate churn models, then save best model artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from rich.table import Table

# Resolve project root dynamically so script runs from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Add src folder to Python import path for local package imports.
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from bank_churn_ml.config import load_config
from bank_churn_ml.data_loading import load_raw_data
from bank_churn_ml.evaluation import build_model_comparison_table, calculate_classification_metrics
from bank_churn_ml.features import add_engineered_features
from bank_churn_ml.modeling import train_and_compare_models
from bank_churn_ml.preprocessing import prepare_training_matrices
from bank_churn_ml.utils import ensure_directory, setup_logging


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Train and compare churn models")
	parser.add_argument(
		"--config",
		type=Path,
		default=Path("config/config.yaml"),
		help="Path to YAML config file",
	)
	return parser.parse_args()


def _build_comparison_records(
	training_results,
	X_test: pd.DataFrame,
	y_test: pd.Series,
) -> list[dict[str, float | str]]:
	"""Create comparison rows from trained model results."""
	records: list[dict[str, float | str]] = []

	for result in training_results:
		y_pred = pd.Series(result.pipeline.predict(X_test), index=y_test.index)
		y_prob = pd.Series(result.pipeline.predict_proba(X_test)[:, 1], index=y_test.index)

		metrics = calculate_classification_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
		row: dict[str, float | str] = {
			"model_name": result.model_name,
			"cv_roc_auc": float(result.cv_roc_auc_mean),
			"test_accuracy": metrics["accuracy"],
			"test_precision": metrics["precision"],
			"test_recall": metrics["recall"],
			"test_f1_score": metrics["f1_score"],
			"test_roc_auc": metrics["roc_auc"],
		}
		records.append(row)

	return records


def main() -> None:
	"""Run model training pipeline for Phase 5."""
	args = parse_args()
	config = load_config(args.config)
	logger, console = setup_logging(
		level=config.get("logging", {}).get("level", "INFO"),
		fmt=config.get("logging", {}).get("format"),
		script_name="train_model",
		log_dir=PROJECT_ROOT / Path(config.get("paths", {}).get("logs_dir", "reports/logs")),
	)

	raw_data_path = PROJECT_ROOT / Path(config["paths"]["raw_data"])
	processed_data_path = PROJECT_ROOT / Path(config["paths"]["processed_data"])
	best_model_path = PROJECT_ROOT / Path(config["paths"]["best_model"])
	metrics_dir = PROJECT_ROOT / Path(config["paths"]["metrics_dir"])

	target_column = config["project"]["target_column"]
	id_column = config["project"]["id_column"]
	random_state = int(config["project"]["random_state"])

	test_size = float(config["split"]["test_size"])
	stratify = bool(config["split"]["stratify"])
	cv_folds = int(config["modeling"]["cross_validation_folds"])

	# Prefer processed dataset; fall back to raw + feature engineering.
	source_path = processed_data_path if processed_data_path.exists() else raw_data_path
	logger.info("Loading dataset for training from %s", source_path)
	dataframe = load_raw_data(source_path)

	if "balance_to_salary_ratio" not in dataframe.columns:
		dataframe = add_engineered_features(dataframe)

	prepared = prepare_training_matrices(
		dataframe=dataframe,
		target_column=target_column,
		id_column=id_column,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify,
	)

	logger.info("Training candidate models with %s-fold cross validation", cv_folds)
	training_results = train_and_compare_models(
		X_train=prepared["X_train"],
		y_train=prepared["y_train"],
		preprocessor=prepared["preprocessor"],
		random_state=random_state,
		cv_folds=cv_folds,
	)

	comparison_records = _build_comparison_records(
		training_results=training_results,
		X_test=prepared["X_test"],
		y_test=prepared["y_test"],
	)

	comparison_table = build_model_comparison_table(comparison_records)

	ensure_directory(metrics_dir)
	comparison_path = metrics_dir / "model_comparison.csv"
	comparison_table.to_csv(comparison_path, index=False)
	logger.info("Saved model comparison to %s", comparison_path)

	# Save best model pipeline.
	best_result = training_results[0]
	ensure_directory(best_model_path.parent)
	joblib.dump(best_result.pipeline, best_model_path)
	logger.info("Saved best model '%s' to %s", best_result.model_name, best_model_path)

	# Print compact summary table in terminal.
	rich_table = Table(title="Phase 5 Model Comparison")
	rich_table.add_column("Model", style="cyan")
	rich_table.add_column("CV ROC-AUC", justify="right", style="magenta")
	rich_table.add_column("Test ROC-AUC", justify="right", style="green")
	rich_table.add_column("Test Recall", justify="right", style="yellow")

	for _, row in comparison_table.iterrows():
		rich_table.add_row(
			str(row["model_name"]),
			f"{float(row['cv_roc_auc']):.4f}",
			f"{float(row['test_roc_auc']):.4f}",
			f"{float(row['test_recall']):.4f}",
		)

	console.print(rich_table)
	console.print(f"Best model: [bold]{best_result.model_name}[/bold]")
	console.print(f"Saved model: [green]{best_model_path.relative_to(PROJECT_ROOT)}[/green]")
	console.print(f"Saved comparison: [green]{comparison_path.relative_to(PROJECT_ROOT)}[/green]")


if __name__ == "__main__":
	main()
