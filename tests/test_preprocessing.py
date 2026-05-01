from __future__ import annotations

import pandas as pd

from bank_churn_ml.features import add_engineered_features
from bank_churn_ml.preprocessing import prepare_training_matrices


def test_prepare_training_matrices_shapes_and_id_drop() -> None:
	# Synthetic dataset with mixed numeric/categorical features.
	base = pd.DataFrame(
		{
			"customer_id": list(range(1, 11)),
			"credit_score": [650, 700, 710, 640, 690, 720, 680, 670, 705, 695],
			"country": ["France", "Germany", "Spain", "France", "Germany", "Spain", "France", "Germany", "Spain", "France"],
			"gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
			"age": [35, 42, 50, 28, 46, 39, 31, 44, 53, 37],
			"tenure": [4, 7, 3, 2, 5, 8, 1, 6, 9, 4],
			"balance": [0.0, 120000.0, 80000.0, 30000.0, 50000.0, 150000.0, 20000.0, 110000.0, 130000.0, 40000.0],
			"products_number": [1, 2, 2, 1, 3, 2, 1, 2, 3, 1],
			"credit_card": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
			"active_member": [1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
			"estimated_salary": [50000.0, 90000.0, 100000.0, 45000.0, 70000.0, 120000.0, 40000.0, 95000.0, 110000.0, 60000.0],
			"churn": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
		}
	)
	# Apply project feature engineering before preprocessing.
	dataframe = add_engineered_features(base)

	result = prepare_training_matrices(
		dataframe,
		target_column="churn",
		id_column="customer_id",
		test_size=0.2,
		random_state=42,
		stratify=True,
	)

	# Validate split sizes and leakage prevention (ID must be dropped).
	assert result["X_train"].shape[0] == 8
	assert result["X_test"].shape[0] == 2
	assert "customer_id" not in result["X_train"].columns

	# Ensure preprocessor can fit and transform without shape mismatch.
	transformed = result["preprocessor"].fit_transform(result["X_train"], result["y_train"])
	assert transformed.shape[0] == result["X_train"].shape[0]
