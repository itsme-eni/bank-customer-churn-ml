"""Prediction helpers for Phase 7 inference CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from bank_churn_ml.features import add_engineered_features


def load_prediction_strategy(strategy_path: Path | str) -> dict[str, Any]:
	"""Load model decision strategy if it exists, else return empty payload."""
	path = Path(strategy_path)
	# Keep prediction CLI robust even when strategy generation has not run yet.
	if not path.exists():
		return {}
	return json.loads(path.read_text(encoding="utf-8"))


def infer_model_label_from_path(model_path: Path | str) -> str:
	"""Infer model label from artifact file name."""
	name = Path(model_path).stem.lower()
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


def resolve_model_artifact(
	project_root: Path,
	config: dict[str, Any],
	strategy: dict[str, Any],
	model_option: str,
) -> tuple[str, Path]:
	"""Resolve model artifact path and model label based on user option."""
	# Build canonical artifact paths from config and fallback defaults.
	best_path = project_root / Path(config["paths"]["best_model"])
	primary_path = project_root / Path(strategy.get("primary_model_path", "models/primary_model_gradient_boosting.joblib"))
	svm_path = project_root / Path(strategy.get("challenger_model_path", "models/high_recall_model_svm.joblib"))

	if model_option == "best":
		if not best_path.exists():
			raise FileNotFoundError(f"Best model artifact not found: {best_path}")
		return infer_model_label_from_path(best_path), best_path

	if model_option == "primary":
		# Primary mode prefers policy-selected baseline and degrades gracefully to best model.
		if primary_path.exists():
			return str(strategy.get("primary_model", "gradient_boosting")), primary_path
		if best_path.exists():
			return infer_model_label_from_path(best_path), best_path
		raise FileNotFoundError(f"Primary model artifact not found: {primary_path}")

	if model_option == "svm":
		# SVM mode is explicit for aggressive recall-focused campaigns.
		if not svm_path.exists():
			raise FileNotFoundError(f"SVM challenger artifact not found: {svm_path}")
		return "svm", svm_path

	# auto
	# Auto mirrors production policy: primary first, then best-model fallback.
	if primary_path.exists():
		return str(strategy.get("primary_model", "gradient_boosting")), primary_path
	if best_path.exists():
		return infer_model_label_from_path(best_path), best_path
	raise FileNotFoundError("No model artifact found for prediction")


def resolve_prediction_threshold(
	strategy: dict[str, Any],
	model_label: str,
	threshold_mode: str,
	threshold_override: float | None,
) -> tuple[float, str]:
	"""Resolve threshold value and source label for prediction decisions."""
	# Manual override always wins to support ad-hoc business simulations.
	if threshold_override is not None:
		return float(threshold_override), "manual"

	policy_section = strategy.get("threshold_tuning", {}) if isinstance(strategy, dict) else {}
	policy_threshold = policy_section.get("recommended_threshold")
	is_primary = model_label == strategy.get("primary_model")

	if threshold_mode == "default":
		return 0.5, "default"

	if threshold_mode == "policy":
		# In policy mode we still return a deterministic fallback when strategy is unavailable.
		if policy_threshold is None:
			return 0.5, "policy_fallback_default"
		return float(policy_threshold), "policy"

	# auto
	# Auto uses policy only for the primary model; others stay on default threshold.
	if is_primary and policy_threshold is not None:
		return float(policy_threshold), "policy"
	return 0.5, "default"


def prepare_features_for_inference(
	dataframe: pd.DataFrame,
	pipeline: Pipeline,
	target_column: str,
) -> pd.DataFrame:
	"""Apply required feature transformations and align inference columns."""
	# Copy to avoid mutating caller dataframes used elsewhere (tests/notebooks/scripts).
	df = dataframe.copy()

	# Rebuild engineered features when inference input comes from raw schema.
	if "balance_to_salary_ratio" not in df.columns and {"balance", "estimated_salary", "products_number", "tenure", "age"}.issubset(df.columns):
		df = add_engineered_features(df)

	if target_column in df.columns:
		# Inference should run on features only; label is optional metadata at this stage.
		df = df.drop(columns=[target_column])

	required_features = getattr(pipeline, "feature_names_in_", None)
	if required_features is None:
		raise ValueError("Loaded model pipeline is missing feature_names_in_; cannot align inference features")

	missing = [column for column in required_features if column not in df.columns]
	if missing:
		# Fail fast with explicit missing-column details for quick debugging.
		raise ValueError(f"Inference input is missing required feature columns: {missing}")

	# Preserve training feature ordering for predictable pipeline behavior.
	return df[list(required_features)].copy()


def build_prediction_output(
	input_dataframe: pd.DataFrame,
	probabilities: pd.Series,
	predictions: pd.Series,
	model_label: str,
	threshold: float,
	threshold_source: str,
) -> pd.DataFrame:
	"""Build output table containing original rows plus prediction fields."""
	# Keep original columns intact and append model/threshold/probability outputs.
	output_df = input_dataframe.copy()
	output_df["prediction_model"] = model_label
	output_df["prediction_threshold"] = float(threshold)
	output_df["threshold_source"] = threshold_source
	output_df["churn_probability"] = probabilities.astype(float)
	output_df["churn_prediction"] = predictions.astype(int)
	return output_df
