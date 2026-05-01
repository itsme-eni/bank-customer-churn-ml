from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bank_churn_ml.predict import (
	build_prediction_output,
	load_prediction_strategy,
	prepare_features_for_inference,
	resolve_model_artifact,
	resolve_prediction_threshold,
)


class _MockPipeline:
	"""Minimal pipeline-like object exposing feature_names_in_."""

	def __init__(self, feature_names_in_: list[str]) -> None:
		self.feature_names_in_ = feature_names_in_


def test_load_prediction_strategy_handles_missing_and_existing_file(tmp_path: Path) -> None:
	# Missing file should return empty dict, not crash.
	missing = load_prediction_strategy(tmp_path / "no_strategy.json")
	assert missing == {}

	# Existing file should load and preserve policy keys.
	strategy_path = tmp_path / "strategy.json"
	strategy_path.write_text(json.dumps({"primary_model": "gradient_boosting"}), encoding="utf-8")

	loaded = load_prediction_strategy(strategy_path)
	assert loaded["primary_model"] == "gradient_boosting"


def test_resolve_model_artifact_prefers_primary_and_can_fallback_best(tmp_path: Path) -> None:
	# Create lightweight fake artifacts so resolution can be tested without model training.
	project_root = tmp_path
	models_dir = project_root / "models"
	models_dir.mkdir(parents=True, exist_ok=True)

	best_path = models_dir / "best_model.joblib"
	best_path.write_bytes(b"fake")
	primary_path = models_dir / "primary_model_gradient_boosting.joblib"
	primary_path.write_bytes(b"fake")

	config = {"paths": {"best_model": "models/best_model.joblib"}}
	strategy = {
		"primary_model": "gradient_boosting",
		"primary_model_path": "models/primary_model_gradient_boosting.joblib",
	}

	# Auto should pick primary artifact first when available.
	label, resolved = resolve_model_artifact(project_root, config, strategy, model_option="auto")
	assert label == "gradient_boosting"
	assert resolved == primary_path

	# Primary mode should fallback to best model when primary artifact is missing.
	primary_path.unlink()
	label, resolved = resolve_model_artifact(project_root, config, strategy, model_option="primary")
	assert label == "best_model"
	assert resolved == best_path


def test_resolve_prediction_threshold_manual_and_policy_paths() -> None:
	# Strategy payload simulates output from train_model decision policy.
	strategy = {
		"primary_model": "gradient_boosting",
		"threshold_tuning": {"recommended_threshold": 0.1234},
	}

	# Auto + primary model should use policy threshold.
	threshold, source = resolve_prediction_threshold(
		strategy=strategy,
		model_label="gradient_boosting",
		threshold_mode="auto",
		threshold_override=None,
	)
	assert threshold == pytest.approx(0.1234)
	assert source == "policy"

	# Manual override should take precedence over auto/policy logic.
	threshold, source = resolve_prediction_threshold(
		strategy=strategy,
		model_label="svm",
		threshold_mode="auto",
		threshold_override=0.9,
	)
	assert threshold == pytest.approx(0.9)
	assert source == "manual"

	# Policy mode should still return deterministic 0.5 fallback if policy is absent.
	threshold, source = resolve_prediction_threshold(
		strategy={},
		model_label="gradient_boosting",
		threshold_mode="policy",
		threshold_override=None,
	)
	assert threshold == pytest.approx(0.5)
	assert source == "policy_fallback_default"


def test_prepare_features_for_inference_builds_engineered_and_orders_columns() -> None:
	# Raw-style input intentionally excludes engineered features.
	dataframe = pd.DataFrame(
		{
			"customer_id": [1, 2],
			"credit_score": [650, 700],
			"country": ["France", "Germany"],
			"gender": ["Female", "Male"],
			"age": [35, 45],
			"tenure": [5, 7],
			"balance": [50000.0, 120000.0],
			"products_number": [2, 3],
			"credit_card": [1, 1],
			"active_member": [1, 0],
			"estimated_salary": [90000.0, 110000.0],
			"churn": [0, 1],
		}
	)

	# Mock expected training feature order used by a fitted pipeline.
	pipeline = _MockPipeline(
		[
			"customer_id",
			"credit_score",
			"country",
			"gender",
			"age",
			"tenure",
			"balance",
			"products_number",
			"credit_card",
			"active_member",
			"estimated_salary",
			"balance_to_salary_ratio",
			"products_per_tenure",
			"age_group",
			"tenure_group",
			"high_value_customer",
		]
	)

	# Function should engineer features, drop target, and enforce exact column order.
	aligned = prepare_features_for_inference(dataframe, pipeline, target_column="churn")
	assert "churn" not in aligned.columns
	assert list(aligned.columns) == list(pipeline.feature_names_in_)
	assert "balance_to_salary_ratio" in aligned.columns


def test_prepare_features_for_inference_raises_for_missing_columns() -> None:
	# Intentionally missing required "country" column to verify fail-fast behavior.
	dataframe = pd.DataFrame({"credit_score": [650], "age": [35]})
	pipeline = _MockPipeline(["credit_score", "country"])

	with pytest.raises(ValueError, match="missing required feature columns"):
		prepare_features_for_inference(dataframe, pipeline, target_column="churn")


def test_build_prediction_output_appends_prediction_fields() -> None:
	# Simple two-row fixture is enough to validate output schema and values.
	input_df = pd.DataFrame({"customer_id": [1, 2], "credit_score": [700, 640]})
	prob = pd.Series([0.8, 0.2])
	pred = pd.Series([1, 0])

	output = build_prediction_output(
		input_dataframe=input_df,
		probabilities=prob,
		predictions=pred,
		model_label="gradient_boosting",
		threshold=0.1317,
		threshold_source="policy",
	)

	# Output should preserve inputs and append prediction metadata columns.
	assert "churn_probability" in output.columns
	assert "churn_prediction" in output.columns
	assert output["prediction_model"].iloc[0] == "gradient_boosting"
	assert output["threshold_source"].iloc[0] == "policy"
	assert output["churn_prediction"].tolist() == [1, 0]