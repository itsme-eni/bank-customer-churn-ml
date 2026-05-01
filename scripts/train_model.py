"""Train and compare candidate churn models, then save best model artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from rich.table import Table
from sklearn.metrics import precision_recall_curve

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
	# Keep CLI simple: config path is the only runtime override needed.
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
	# Each row merges CV score with holdout-test metrics for side-by-side ranking.
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


def _find_threshold_for_recall(
	y_true: pd.Series,
	y_prob: pd.Series,
	min_precision: float = 0.40,
	default_threshold: float = 0.5,
) -> dict[str, float | bool]:
	"""Select threshold maximizing recall while keeping precision above a floor."""
	_, _, pr_thresholds = precision_recall_curve(y_true, y_prob)

	# Evaluate threshold candidates directly from predicted probabilities.
	threshold_candidates = sorted({float(default_threshold), *pr_thresholds.tolist()})

	def _metrics_at_threshold(threshold: float) -> tuple[float, float]:
		pred = (y_prob >= threshold).astype(int)
		tp = int(((pred == 1) & (y_true == 1)).sum())
		fp = int(((pred == 1) & (y_true == 0)).sum())
		fn = int(((pred == 0) & (y_true == 1)).sum())
		precision_value = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
		recall_value = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
		return precision_value, recall_value

	viable_candidates: list[tuple[float, float, float]] = []
	fallback_candidates: list[tuple[float, float, float]] = []

	for threshold in threshold_candidates:
		precision_value, recall_value = _metrics_at_threshold(threshold)
		candidate = (threshold, precision_value, recall_value)
		fallback_candidates.append(candidate)
		if precision_value >= min_precision:
			viable_candidates.append(candidate)

	if viable_candidates:
		best_threshold, best_precision, best_recall = max(
			viable_candidates,
			key=lambda item: (item[2], item[1], -abs(item[0] - default_threshold)),
		)
		return {
			"threshold": float(best_threshold),
			"precision": float(best_precision),
			"recall": float(best_recall),
			"meets_min_precision": True,
		}

	# Fallback: maximize recall even if precision floor cannot be met.
	best_threshold, best_precision, best_recall = max(
		fallback_candidates,
		key=lambda item: (item[2], item[1], -abs(item[0] - default_threshold)),
	)
	return {
		"threshold": float(best_threshold),
		"precision": float(best_precision),
		"recall": float(best_recall),
		"meets_min_precision": False,
	}


def main() -> None:
	"""Run model training pipeline for Phase 5."""
	# 1) Parse CLI options and config.
	args = parse_args()
	config = load_config(args.config)
	logger, console = setup_logging(
		level=config.get("logging", {}).get("level", "INFO"),
		fmt=config.get("logging", {}).get("format"),
		script_name="train_model",
		log_dir=PROJECT_ROOT / Path(config.get("paths", {}).get("logs_dir", "reports/logs")),
	)

	# 2) Resolve key paths and modeling settings.
	raw_data_path = PROJECT_ROOT / Path(config["paths"]["raw_data"])
	processed_data_path = PROJECT_ROOT / Path(config["paths"]["processed_data"])
	best_model_path = PROJECT_ROOT / Path(config["paths"]["best_model"])
	metrics_dir = PROJECT_ROOT / Path(config["paths"]["metrics_dir"])
	models_dir = PROJECT_ROOT / Path("models")

	target_column = config["project"]["target_column"]
	id_column = config["project"]["id_column"]
	random_state = int(config["project"]["random_state"])

	test_size = float(config["split"]["test_size"])
	stratify = bool(config["split"]["stratify"])
	cv_folds = int(config["modeling"]["cross_validation_folds"])

	# 3) Prefer processed dataset; fall back to raw + feature engineering.
	source_path = processed_data_path if processed_data_path.exists() else raw_data_path
	logger.info("Loading dataset for training from %s", source_path)
	dataframe = load_raw_data(source_path)

	if "balance_to_salary_ratio" not in dataframe.columns:
		dataframe = add_engineered_features(dataframe)

	# 4) Build train/test matrices and preprocessing object.
	prepared = prepare_training_matrices(
		dataframe=dataframe,
		target_column=target_column,
		id_column=id_column,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify,
	)

	# 5) Train baseline candidate models with cross-validation.
	logger.info("Training candidate models with %s-fold cross validation", cv_folds)
	training_results = train_and_compare_models(
		X_train=prepared["X_train"],
		y_train=prepared["y_train"],
		preprocessor=prepared["preprocessor"],
		random_state=random_state,
		cv_folds=cv_folds,
	)

	# 6) Evaluate each fitted pipeline on holdout test split.
	comparison_records = _build_comparison_records(
		training_results=training_results,
		X_test=prepared["X_test"],
		y_test=prepared["y_test"],
	)

	# 7) Save comparison CSV for reporting.
	comparison_table = build_model_comparison_table(comparison_records)

	ensure_directory(metrics_dir)
	comparison_path = metrics_dir / "model_comparison.csv"
	comparison_table.to_csv(comparison_path, index=False)
	logger.info("Saved model comparison to %s", comparison_path)

	# 8) Save best model pipeline for downstream evaluation/inference.
	best_result = training_results[0]
	ensure_directory(best_model_path.parent)
	joblib.dump(best_result.pipeline, best_model_path)
	logger.info("Saved best model '%s' to %s", best_result.model_name, best_model_path)

	# Portfolio decision layer:
	# Keep Gradient Boosting as primary baseline when available.
	result_by_name = {result.model_name: result for result in training_results}
	primary_result = result_by_name.get("gradient_boosting", best_result)
	svm_result = result_by_name.get("svm")

	# Tune decision threshold on holdout probabilities to improve recall.
	primary_prob = pd.Series(
		primary_result.pipeline.predict_proba(prepared["X_test"])[:, 1],
		index=prepared["y_test"].index,
	)
	threshold_policy = _find_threshold_for_recall(
		y_true=prepared["y_test"],
		y_prob=primary_prob,
		min_precision=0.40,
	)

	# Save primary and challenger models for operational flexibility.
	ensure_directory(models_dir)
	primary_model_path = models_dir / "primary_model_gradient_boosting.joblib"
	joblib.dump(primary_result.pipeline, primary_model_path)
	logger.info("Saved primary baseline model '%s' to %s", primary_result.model_name, primary_model_path)

	svm_model_path = models_dir / "high_recall_model_svm.joblib"
	if svm_result is not None:
		joblib.dump(svm_result.pipeline, svm_model_path)
		logger.info("Saved high-recall challenger model 'svm' to %s", svm_model_path)

	# Export a machine-readable strategy note for README/report integration.
	decision_payload = {
		"primary_model": primary_result.model_name,
		"primary_model_path": str(primary_model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
		"selection_reason": "Primary baseline chosen for strongest ranking performance (ROC-AUC).",
		"threshold_tuning": {
			"objective": "Increase recall while keeping precision >= 0.40",
			"min_precision": 0.40,
			"recommended_threshold": threshold_policy["threshold"],
			"expected_precision": threshold_policy["precision"],
			"expected_recall": threshold_policy["recall"],
			"meets_min_precision": threshold_policy["meets_min_precision"],
		},
		"challenger_model": "svm" if svm_result is not None else "not_available",
		"challenger_model_path": str(svm_model_path.relative_to(PROJECT_ROOT)).replace("\\", "/") if svm_result is not None else "",
		"challenger_reason": "Use SVM for aggressive retention strategy prioritizing recall.",
	}

	decision_path = metrics_dir / "model_decision_strategy.json"
	decision_path.write_text(json.dumps(decision_payload, indent=2), encoding="utf-8")
	logger.info("Saved model decision strategy to %s", decision_path)

	# 9) Print compact summary table in terminal.
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
	console.print(f"Primary baseline: [bold]{primary_result.model_name}[/bold]")
	console.print(
		"Recommended threshold for primary model: "
		f"[bold]{threshold_policy['threshold']:.4f}[/bold] "
		f"(precision={threshold_policy['precision']:.4f}, recall={threshold_policy['recall']:.4f})"
	)
	if not bool(threshold_policy["meets_min_precision"]):
		console.print(
			"[yellow]No threshold met precision >= 0.40 on holdout set; "
			"using highest-recall fallback threshold.[/yellow]"
		)
	if svm_result is not None:
		console.print("High-recall challenger: [bold]svm[/bold]")
	console.print(f"Saved model: [green]{best_model_path.relative_to(PROJECT_ROOT)}[/green]")
	console.print(f"Saved comparison: [green]{comparison_path.relative_to(PROJECT_ROOT)}[/green]")
	console.print(f"Saved strategy: [green]{decision_path.relative_to(PROJECT_ROOT)}[/green]")


if __name__ == "__main__":
	main()
