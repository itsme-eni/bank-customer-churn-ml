"""Validation functions for raw and engineered churn data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataQualityReport:
	"""Summary of key quality checks for a dataset."""

	row_count: int
	column_count: int
	duplicate_rows: int
	missing_by_column: dict[str, int]


def validate_required_columns(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
	"""Ensure all required columns are present."""
	# Compare required vs actual columns so errors are explicit and actionable.
	missing_columns = sorted(set(required_columns) - set(dataframe.columns))
	if missing_columns:
		raise ValueError(f"Missing required columns: {missing_columns}")


def validate_binary_target(dataframe: pd.DataFrame, target_column: str) -> None:
	"""Validate that the target exists and is binary."""
	# Fail fast if target column is absent.
	if target_column not in dataframe.columns:
		raise ValueError(f"Target column '{target_column}' not found")

	# Churn is a binary classification label and should contain only 0/1 values.
	non_null_values = set(dataframe[target_column].dropna().unique().tolist())
	if not non_null_values.issubset({0, 1}):
		raise ValueError(
			f"Target column '{target_column}' must be binary 0/1. Found values: {sorted(non_null_values)}"
		)


def generate_data_quality_report(dataframe: pd.DataFrame) -> DataQualityReport:
	"""Build a compact quality report for logging and diagnostics."""
	# Count missing values per column for quick quality inspection.
	missing_by_column = dataframe.isna().sum().to_dict()
	return DataQualityReport(
		row_count=len(dataframe),
		column_count=dataframe.shape[1],
		duplicate_rows=int(dataframe.duplicated().sum()),
		missing_by_column={str(column): int(count) for column, count in missing_by_column.items()},
	)
