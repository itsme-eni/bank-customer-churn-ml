"""Evaluate trained churn models and export Phase 6 metrics/figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.table import Table
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# Resolve project root dynamically so script runs from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Add src folder to Python import path for local package imports.
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from bank_churn_ml.config import load_config
from bank_churn_ml.data_loading import load_raw_data
from bank_churn_ml.evaluation import calculate_classification_metrics, get_confusion_matrix
from bank_churn_ml.features import add_engineered_features
from bank_churn_ml.preprocessing import prepare_training_matrices
from bank_churn_ml.utils import ensure_directory, setup_logging


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Evaluate trained churn model artifacts")
	parser.add_argument(
		"--config",
		type=Path,
		default=Path("config/config.yaml"),
		help="Path to YAML config file",
	)
	return parser.parse_args()


def _load_evaluation_dataset(config: dict[str, Any], logger) -> dict[str, Any]:
	"""Load processed/raw data and rebuild deterministic holdout split."""
	raw_data_path = PROJECT_ROOT / Path(config["paths"]["raw_data"])
	processed_data_path = PROJECT_ROOT / Path(config["paths"]["processed_data"])

	target_column = config["project"]["target_column"]
	id_column = config["project"]["id_column"]
	random_state = int(config["project"]["random_state"])
	test_size = float(config["split"]["test_size"])
	stratify = bool(config["split"]["stratify"])

	source_path = processed_data_path if processed_data_path.exists() else raw_data_path
	logger.info("Loading dataset for evaluation from %s", source_path)
	dataframe = load_raw_data(source_path)

	if "balance_to_salary_ratio" not in dataframe.columns:
		dataframe = add_engineered_features(dataframe)

	return prepare_training_matrices(
		dataframe=dataframe,
		target_column=target_column,
		id_column=id_column,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify,
	)


def _find_model_label_from_path(path: Path) -> str:
	"""Infer human-readable model label from artifact file name."""
	name = path.stem.lower()
	if "gradient_boosting" in name:
		return "gradient_boosting"
	if "svm" in name:
		return "svm"
	if "random_forest" in name:
		return "random_forest"
	if "logistic" in name:
		return "logistic_regression"
	if "best_model" in name:
		return "best_model"
	return name


def _evaluate_at_threshold(
	y_true: pd.Series,
	y_prob: pd.Series,
	threshold: float,
) -> tuple[dict[str, float], pd.Series]:
	"""Compute metrics and predictions for a specific probability threshold."""
	y_pred = (y_prob >= threshold).astype(int)
	metrics = calculate_classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
	return metrics, y_pred


def _plot_roc_curve(records: list[dict[str, Any]], output_path: Path) -> None:
	"""Plot ROC curve for all evaluated model strategies."""
	sns.set_theme(style="whitegrid")
	plt.figure(figsize=(9, 6))

	for record in records:
		fpr, tpr, _ = roc_curve(record["y_true"], record["y_prob"])
		curve_auc = auc(fpr, tpr)
		label = f"{record['model_label']} [{record['strategy_label']}] (AUC={curve_auc:.3f})"
		plt.plot(fpr, tpr, linewidth=2, label=label)

	plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Random")
	plt.title("ROC Curves - Phase 6 Evaluation")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.legend(loc="lower right", fontsize=9)
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close()


def _plot_pr_curve(records: list[dict[str, Any]], output_path: Path) -> None:
	"""Plot precision-recall curve for all evaluated model strategies."""
	sns.set_theme(style="whitegrid")
	plt.figure(figsize=(9, 6))

	for record in records:
		precision, recall, _ = precision_recall_curve(record["y_true"], record["y_prob"])
		curve_auc = auc(recall, precision)
		label = f"{record['model_label']} [{record['strategy_label']}] (AUC={curve_auc:.3f})"
		plt.plot(recall, precision, linewidth=2, label=label)

	plt.title("Precision-Recall Curves - Phase 6 Evaluation")
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.legend(loc="lower left", fontsize=9)
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close()


def _plot_confusion_matrix(
	conf_matrix: list[list[int]],
	model_label: str,
	strategy_label: str,
	output_path: Path,
) -> None:
	"""Save confusion matrix heatmap for one model strategy."""
	sns.set_theme(style="whitegrid")
	plt.figure(figsize=(6, 5))
	cm_df = pd.DataFrame(conf_matrix, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
	sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
	plt.title(f"Confusion Matrix - {model_label} [{strategy_label}]")
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close()


def main() -> None:
	"""Run Phase 6 model evaluation and reporting pipeline."""
	args = parse_args()
	config = load_config(args.config)
	logger, console = setup_logging(
		level=config.get("logging", {}).get("level", "INFO"),
		fmt=config.get("logging", {}).get("format"),
		script_name="evaluate_model",
		log_dir=PROJECT_ROOT / Path(config.get("paths", {}).get("logs_dir", "reports/logs")),
	)

	best_model_path = PROJECT_ROOT / Path(config["paths"]["best_model"])
	metrics_dir = PROJECT_ROOT / Path(config["paths"]["metrics_dir"])
	figures_dir = PROJECT_ROOT / Path(config["paths"]["figures_dir"])
	ensure_directory(metrics_dir)
	eval_figures_dir = ensure_directory(figures_dir / "evaluation")

	primary_model_path = PROJECT_ROOT / Path("models/primary_model_gradient_boosting.joblib")
	svm_model_path = PROJECT_ROOT / Path("models/high_recall_model_svm.joblib")
	decision_path = metrics_dir / "model_decision_strategy.json"

	prepared = _load_evaluation_dataset(config=config, logger=logger)
	X_test: pd.DataFrame = prepared["X_test"]
	y_test: pd.Series = prepared["y_test"]

	if not best_model_path.exists():
		raise FileNotFoundError(f"Best model artifact not found: {best_model_path}")

	# Build model registry, preferring named artifacts when available.
	model_paths: dict[str, Path] = {}
	if primary_model_path.exists():
		model_paths["gradient_boosting"] = primary_model_path
	if svm_model_path.exists():
		model_paths["svm"] = svm_model_path

	if not model_paths:
		model_paths[_find_model_label_from_path(best_model_path)] = best_model_path

	strategy_payload: dict[str, Any] = {}
	if decision_path.exists():
		strategy_payload = json.loads(decision_path.read_text(encoding="utf-8"))

	evaluation_rows: list[dict[str, Any]] = []
	curve_records: list[dict[str, Any]] = []

	for model_label, artifact_path in model_paths.items():
		logger.info("Evaluating model '%s' from %s", model_label, artifact_path)
		pipeline = joblib.load(artifact_path)
		y_prob = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=y_test.index)

		# Always evaluate default decision threshold.
		default_metrics, default_pred = _evaluate_at_threshold(y_true=y_test, y_prob=y_prob, threshold=0.5)
		default_conf_matrix = get_confusion_matrix(y_true=y_test, y_pred=default_pred)
		evaluation_rows.append(
			{
				"model_label": model_label,
				"artifact_path": str(artifact_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
				"strategy_label": "default_threshold",
				"threshold": 0.5,
				"accuracy": default_metrics["accuracy"],
				"precision": default_metrics["precision"],
				"recall": default_metrics["recall"],
				"f1_score": default_metrics["f1_score"],
				"roc_auc": default_metrics["roc_auc"],
				"confusion_matrix": default_conf_matrix,
			}
		)
		curve_records.append(
			{
				"model_label": model_label,
				"strategy_label": "default_threshold",
				"y_true": y_test,
				"y_prob": y_prob,
			}
		)

		# Evaluate threshold policy if strategy file identifies this as primary.
		if (
			strategy_payload
			and model_label == strategy_payload.get("primary_model")
			and isinstance(strategy_payload.get("threshold_tuning"), dict)
		):
			policy_threshold = float(strategy_payload["threshold_tuning"].get("recommended_threshold", 0.5))
			policy_metrics, policy_pred = _evaluate_at_threshold(
				y_true=y_test,
				y_prob=y_prob,
				threshold=policy_threshold,
			)
			policy_conf_matrix = get_confusion_matrix(y_true=y_test, y_pred=policy_pred)
			evaluation_rows.append(
				{
					"model_label": model_label,
					"artifact_path": str(artifact_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
					"strategy_label": "policy_threshold",
					"threshold": policy_threshold,
					"accuracy": policy_metrics["accuracy"],
					"precision": policy_metrics["precision"],
					"recall": policy_metrics["recall"],
					"f1_score": policy_metrics["f1_score"],
					"roc_auc": policy_metrics["roc_auc"],
					"confusion_matrix": policy_conf_matrix,
				}
			)
			curve_records.append(
				{
					"model_label": model_label,
					"strategy_label": "policy_threshold",
					"y_true": y_test,
					"y_prob": y_prob,
				}
			)

	# Persist tabular and JSON reports.
	evaluation_df = pd.DataFrame(evaluation_rows)
	evaluation_df.to_csv(metrics_dir / "evaluation_summary.csv", index=False)
	(metrics_dir / "evaluation_summary.json").write_text(
		json.dumps(evaluation_rows, indent=2),
		encoding="utf-8",
	)
	logger.info("Saved evaluation summary artifacts to %s", metrics_dir)

	# Save confusion matrix figures for each evaluated strategy.
	for row in evaluation_rows:
		cm_filename = f"confusion_matrix_{row['model_label']}_{row['strategy_label']}.png"
		_plot_confusion_matrix(
			conf_matrix=row["confusion_matrix"],
			model_label=str(row["model_label"]),
			strategy_label=str(row["strategy_label"]),
			output_path=eval_figures_dir / cm_filename,
		)

	# Save curve figures across evaluated strategies.
	roc_path = eval_figures_dir / "roc_curves.png"
	pr_path = eval_figures_dir / "precision_recall_curves.png"
	_plot_roc_curve(records=curve_records, output_path=roc_path)
	_plot_pr_curve(records=curve_records, output_path=pr_path)
	logger.info("Saved evaluation figures to %s", eval_figures_dir)

	# Terminal summary table.
	rich_table = Table(title="Phase 6 Evaluation Summary")
	rich_table.add_column("Model", style="cyan")
	rich_table.add_column("Strategy", style="magenta")
	rich_table.add_column("Threshold", justify="right", style="white")
	rich_table.add_column("ROC-AUC", justify="right", style="green")
	rich_table.add_column("Recall", justify="right", style="yellow")
	rich_table.add_column("Precision", justify="right", style="blue")

	for _, row in evaluation_df.iterrows():
		rich_table.add_row(
			str(row["model_label"]),
			str(row["strategy_label"]),
			f"{float(row['threshold']):.4f}",
			f"{float(row['roc_auc']):.4f}",
			f"{float(row['recall']):.4f}",
			f"{float(row['precision']):.4f}",
		)

	console.print(rich_table)
	console.print(f"Saved metrics: [green]{(metrics_dir / 'evaluation_summary.csv').relative_to(PROJECT_ROOT)}[/green]")
	console.print(f"Saved metrics JSON: [green]{(metrics_dir / 'evaluation_summary.json').relative_to(PROJECT_ROOT)}[/green]")
	console.print(f"Saved figures: [green]{eval_figures_dir.relative_to(PROJECT_ROOT)}[/green]")


if __name__ == "__main__":
	main()
