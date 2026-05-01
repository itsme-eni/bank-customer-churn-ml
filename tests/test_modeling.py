from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from bank_churn_ml.features import add_engineered_features
from bank_churn_ml.modeling import train_and_compare_models
from bank_churn_ml.preprocessing import prepare_training_matrices


def test_model_pipeline_fit_and_prediction_format() -> None:
	# Build a slightly larger synthetic dataset for model training.
	base = pd.DataFrame(
		{
			"customer_id": list(range(1, 41)),
			"credit_score": [600 + (i % 120) for i in range(40)],
			"country": ["France", "Germany", "Spain", "France"] * 10,
			"gender": ["Female", "Male"] * 20,
			"age": [25 + (i % 35) for i in range(40)],
			"tenure": [i % 10 for i in range(40)],
			"balance": [float((i % 6) * 25000) for i in range(40)],
			"products_number": [1 + (i % 3) for i in range(40)],
			"credit_card": [i % 2 for i in range(40)],
			"active_member": [(i + 1) % 2 for i in range(40)],
			"estimated_salary": [35000.0 + (i * 2500.0) for i in range(40)],
			"churn": [0, 1, 0, 1] * 10,
		}
	)
	# Full prep flow: feature engineering + split + preprocessing.
	dataframe = add_engineered_features(base)
	prepared = prepare_training_matrices(
		dataframe,
		target_column="churn",
		id_column="customer_id",
		test_size=0.25,
		random_state=42,
		stratify=True,
	)

	# Train all required baseline models and get ranked results.
	results = train_and_compare_models(
		X_train=prepared["X_train"],
		y_train=prepared["y_train"],
		preprocessor=prepared["preprocessor"],
		random_state=42,
		cv_folds=3,
	)

	# Ensure model list exists and contains sklearn pipelines.
	assert len(results) >= 4
	assert isinstance(results[0].pipeline, Pipeline)

	# Validate prediction outputs from the top-ranked pipeline.
	best_pipeline = results[0].pipeline
	predictions = best_pipeline.predict(prepared["X_test"])
	probabilities = best_pipeline.predict_proba(prepared["X_test"])[:, 1]

	assert predictions.shape[0] == prepared["X_test"].shape[0]
	assert probabilities.shape[0] == prepared["X_test"].shape[0]
	assert set(pd.Series(predictions).unique()).issubset({0, 1})
