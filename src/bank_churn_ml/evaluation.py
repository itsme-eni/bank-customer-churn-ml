"""Evaluation utilities for churn classification models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)


# Centralized metric helpers keep evaluation naming and sorting logic consistent
# across training/evaluation scripts and saved artifacts.


def calculate_classification_metrics(
	y_true: pd.Series,
	y_pred: pd.Series,
	y_prob: pd.Series,
) -> dict[str, float]:
	"""Compute key binary classification metrics for churn prediction."""
	# Return core metrics used in business-facing churn model comparison.
	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
		"roc_auc": float(roc_auc_score(y_true, y_prob)),
	}


def build_model_comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
	"""Build a sortable model comparison table from evaluation records."""
	# Convert list-of-dicts records into a tabular comparison view.
	dataframe = pd.DataFrame(results)
	if dataframe.empty:
		return dataframe

	# Prefer ranking by ROC-AUC, then recall, then F1 when available.
	sort_columns = [column for column in ["test_roc_auc", "test_recall", "test_f1_score"] if column in dataframe.columns]
	return dataframe.sort_values(by=sort_columns, ascending=False).reset_index(drop=True)


def get_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> list[list[int]]:
	"""Return confusion matrix as a JSON-serializable nested list."""
	# Nested list output is easy to store in JSON metrics artifacts.
	matrix = confusion_matrix(y_true, y_pred)
	return matrix.astype(int).tolist()
