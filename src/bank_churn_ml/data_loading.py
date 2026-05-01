"""Data loading utilities for the bank churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Canonical schema expected from the Kaggle churn dataset.
EXPECTED_COLUMNS: list[str] = [
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
	"churn",
]


def load_raw_data(data_path: Path | str) -> pd.DataFrame:
	"""Load the raw churn dataset from disk.

	Args:
		data_path: CSV file path.

	Returns:
		DataFrame with normalized snake_case column names.
	"""
	path = Path(data_path)
	if not path.exists():
		raise FileNotFoundError(f"Raw dataset not found: {path}")

	# Read CSV and standardize names to snake_case for consistent downstream code.
	dataframe = pd.read_csv(path)
	dataframe.columns = [column.strip().lower().replace(" ", "_") for column in dataframe.columns]
	return dataframe


def save_dataframe(dataframe: pd.DataFrame, output_path: Path | str) -> Path:
	"""Persist a DataFrame to CSV.

	Args:
		dataframe: DataFrame to save.
		output_path: Destination CSV path.

	Returns:
		Path to saved file.
	"""
	path = Path(output_path)
	# Ensure destination directory exists before writing output file.
	path.parent.mkdir(parents=True, exist_ok=True)
	dataframe.to_csv(path, index=False)
	return path
