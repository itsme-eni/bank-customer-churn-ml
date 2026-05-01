from __future__ import annotations

import pandas as pd

from bank_churn_ml.data_loading import EXPECTED_COLUMNS, load_raw_data
from bank_churn_ml.validation import validate_binary_target, validate_required_columns


def test_load_raw_data_and_required_columns(tmp_path) -> None:
	# Build a minimal valid sample matching required schema.
	dataframe = pd.DataFrame(
		{
			"customer_id": [1, 2],
			"credit_score": [700, 680],
			"country": ["France", "Germany"],
			"gender": ["Female", "Male"],
			"age": [42, 35],
			"tenure": [5, 3],
			"balance": [120000.0, 0.0],
			"products_number": [2, 1],
			"credit_card": [1, 1],
			"active_member": [1, 0],
			"estimated_salary": [100000.0, 90000.0],
			"churn": [0, 1],
		}
	)
	csv_path = tmp_path / "sample.csv"
	dataframe.to_csv(csv_path, index=False)

	# Load through project loader to test column normalization/read path.
	loaded = load_raw_data(csv_path)

	# Validate expected schema and binary target assumptions.
	validate_required_columns(loaded, EXPECTED_COLUMNS)
	validate_binary_target(loaded, "churn")
	assert "churn" in loaded.columns
