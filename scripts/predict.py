"""Generate churn predictions for new data using trained model artifacts."""

from __future__ import annotations

import argparse
import json
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
from bank_churn_ml.data_loading import load_raw_data, save_dataframe
from bank_churn_ml.evaluation import calculate_classification_metrics
from bank_churn_ml.predict import (
	build_prediction_output,
	load_prediction_strategy,
	prepare_features_for_inference,
	resolve_model_artifact,
	resolve_prediction_threshold,
)
from bank_churn_ml.utils import ensure_directory, setup_logging


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for prediction script."""
	parser = argparse.ArgumentParser(description="Generate churn predictions from a trained model")
	parser.add_argument(
		"--config",
		type=Path,
		default=Path("config/config.yaml"),
		help="Path to YAML config file",
	)
	parser.add_argument(
		"--input",
		type=Path,
		required=True,
		help="Input CSV path containing customer rows for prediction",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("reports/metrics/predictions/predictions.csv"),
		help="Output CSV path for predictions",
	)
	parser.add_argument(
		"--model-option",
		type=str,
		choices=["auto", "primary", "best", "svm"],
		default="auto",
		help="Model artifact selection strategy",
	)
	parser.add_argument(
		"--threshold-mode",
		type=str,
		choices=["auto", "default", "policy"],
		default="auto",
		help="Decision threshold strategy",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=None,
		help="Optional manual threshold override in [0, 1]",
	)
	return parser.parse_args()


def main() -> None:
	"""Run Phase 7 prediction flow and save prediction artifacts."""
	args = parse_args()
	# Validate early so users get a clear argument error before any file I/O.
	if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
		raise ValueError("--threshold must be between 0 and 1")

	config = load_config(args.config)
	logger, console = setup_logging(
		level=config.get("logging", {}).get("level", "INFO"),
		fmt=config.get("logging", {}).get("format"),
		script_name="predict",
		log_dir=PROJECT_ROOT / Path(config.get("paths", {}).get("logs_dir", "reports/logs")),
	)

	target_column = config["project"]["target_column"]
	metrics_dir = PROJECT_ROOT / Path(config["paths"]["metrics_dir"])
	strategy_path = metrics_dir / "model_decision_strategy.json"

	# Support both absolute and project-relative paths for CLI convenience.
	input_path = PROJECT_ROOT / args.input if not args.input.is_absolute() else args.input
	output_path = PROJECT_ROOT / args.output if not args.output.is_absolute() else args.output
	metrics_output_path = output_path.with_name(output_path.stem + "_run_metrics.json")

	# Resolve model and threshold using project policy plus explicit CLI options.
	strategy_payload = load_prediction_strategy(strategy_path)
	model_label, model_path = resolve_model_artifact(
		project_root=PROJECT_ROOT,
		config=config,
		strategy=strategy_payload,
		model_option=args.model_option,
	)
	threshold_value, threshold_source = resolve_prediction_threshold(
		strategy=strategy_payload,
		model_label=model_label,
		threshold_mode=args.threshold_mode,
		threshold_override=args.threshold,
	)

	logger.info("Loading prediction input from %s", input_path)
	input_df = load_raw_data(input_path)
	pipeline = joblib.load(model_path)

	# Align inference features to training schema expected by the pipeline.
	inference_X = prepare_features_for_inference(
		dataframe=input_df,
		pipeline=pipeline,
		target_column=target_column,
	)

	# Probabilities are transformed into binary labels via resolved threshold.
	probabilities = pd.Series(
		pipeline.predict_proba(inference_X)[:, 1],
		index=input_df.index,
	)
	predictions = (probabilities >= threshold_value).astype(int)

	output_df = build_prediction_output(
		input_dataframe=input_df,
		probabilities=probabilities,
		predictions=predictions,
		model_label=model_label,
		threshold=threshold_value,
		threshold_source=threshold_source,
	)

	ensure_directory(output_path.parent)
	save_dataframe(output_df, output_path)
	logger.info("Saved predictions to %s", output_path)

	# Save optional run-level metrics when ground-truth labels are available.
	run_metrics_payload: dict[str, float | str | int] = {
		"rows_scored": int(len(output_df)),
		"model_label": model_label,
		"model_path": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
		"threshold": float(threshold_value),
		"threshold_source": threshold_source,
		"predicted_churn_count": int(predictions.sum()),
		"predicted_non_churn_count": int((1 - predictions).sum()),
	}

	if target_column in output_df.columns:
		# If labels are present, include evaluation snapshot for quick sanity checks.
		y_true = output_df[target_column].astype(int)
		eval_metrics = calculate_classification_metrics(
			y_true=y_true,
			y_pred=predictions,
			y_prob=probabilities,
		)
		run_metrics_payload.update({f"eval_{key}": value for key, value in eval_metrics.items()})

	metrics_output_path.write_text(json.dumps(run_metrics_payload, indent=2), encoding="utf-8")
	logger.info("Saved prediction run metrics to %s", metrics_output_path)

	# Terminal summary.
	summary_table = Table(title="Phase 7 Prediction Summary")
	summary_table.add_column("Field", style="cyan")
	summary_table.add_column("Value", style="green")
	summary_table.add_row("Rows scored", str(run_metrics_payload["rows_scored"]))
	summary_table.add_row("Model", model_label)
	summary_table.add_row("Threshold", f"{threshold_value:.4f}")
	summary_table.add_row("Threshold source", threshold_source)
	summary_table.add_row("Predicted churn", str(run_metrics_payload["predicted_churn_count"]))
	summary_table.add_row("Predicted non-churn", str(run_metrics_payload["predicted_non_churn_count"]))

	if "eval_recall" in run_metrics_payload:
		summary_table.add_row("Eval ROC-AUC", f"{float(run_metrics_payload['eval_roc_auc']):.4f}")
		summary_table.add_row("Eval recall", f"{float(run_metrics_payload['eval_recall']):.4f}")
		summary_table.add_row("Eval precision", f"{float(run_metrics_payload['eval_precision']):.4f}")

	console.print(summary_table)
	console.print(f"Saved predictions: [green]{output_path.relative_to(PROJECT_ROOT)}[/green]")
	console.print(f"Saved run metrics: [green]{metrics_output_path.relative_to(PROJECT_ROOT)}[/green]")


if __name__ == "__main__":
	main()
