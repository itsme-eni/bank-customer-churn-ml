"""Model training utilities for churn classification."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# Modeling utilities here are deliberately lightweight so scripts can compare
# multiple estimators using the same preprocessing pipeline.


@dataclass
class TrainedModelResult:
	"""Container for a trained candidate model and its score."""

	model_name: str
	pipeline: Pipeline
	cv_roc_auc_mean: float


def build_candidate_models(random_state: int) -> dict[str, ClassifierMixin]:
	"""Create baseline model candidates required for comparison."""
	# Use class_weight='balanced' for models that support it to reduce class-imbalance bias.
	return {
		"logistic_regression": LogisticRegression(
			max_iter=1000,
			class_weight="balanced",
			random_state=random_state,
		),
		"random_forest": RandomForestClassifier(
			n_estimators=250,
			class_weight="balanced",
			random_state=random_state,
			n_jobs=1,
		),
		"gradient_boosting": GradientBoostingClassifier(random_state=random_state),
		"svm": SVC(probability=True, class_weight="balanced", random_state=random_state),
	}


def train_and_compare_models(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	preprocessor: ColumnTransformer,
	random_state: int,
	cv_folds: int,
) -> list[TrainedModelResult]:
	"""Train candidate models and compare with cross-validated ROC-AUC."""
	# Store fitted pipelines and CV score for each candidate model.
	results: list[TrainedModelResult] = []

	for model_name, estimator in build_candidate_models(random_state=random_state).items():
		# Wrap shared preprocessing + model into one reusable sklearn Pipeline.
		pipeline = Pipeline(
			steps=[
				("preprocessor", preprocessor),
				("classifier", estimator),
			]
		)

		# Evaluate with CV on train set to compare models fairly.
		cv_scores = cross_val_score(
			pipeline,
			X_train,
			y_train,
			cv=cv_folds,
			scoring="roc_auc",
			n_jobs=1,
		)
		# Fit once on all training data so the winning pipeline is ready for testing/artifacts.
		pipeline.fit(X_train, y_train)

		results.append(
			TrainedModelResult(
				model_name=model_name,
				pipeline=pipeline,
				cv_roc_auc_mean=float(cv_scores.mean()),
			)
		)

	# Best model appears first.
	results.sort(key=lambda item: item.cv_roc_auc_mean, reverse=True)
	return results
