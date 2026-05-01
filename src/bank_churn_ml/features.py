"""Feature engineering functions for churn modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Keep feature engineering logic isolated here so notebooks and scripts can use
# the same transformations without copy-paste.


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
	"""Compute ratio and gracefully handle divide-by-zero cases."""
	# Use 0.0 whenever denominator is zero to avoid inf/NaN values.
	ratio = np.where(denominator > 0, numerator / denominator, 0.0)
	return pd.Series(ratio, index=numerator.index, dtype="float64")


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
	"""Create domain-inspired features used in EDA and modeling."""
	# Work on a copy to keep the caller's DataFrame unchanged.
	df = dataframe.copy()

	# Intensity features that capture relative financial behavior.
	df["balance_to_salary_ratio"] = _safe_ratio(df["balance"], df["estimated_salary"])
	df["products_per_tenure"] = _safe_ratio(df["products_number"], df["tenure"])

	# Bucket age into business-friendly customer cohorts.
	df["age_group"] = pd.cut(
		df["age"],
		bins=[0, 30, 45, 60, np.inf],
		labels=["young", "mid_age", "senior", "elder"],
		include_lowest=True,
	).astype("object")

	# Bucket tenure into relationship maturity cohorts.
	df["tenure_group"] = pd.cut(
		df["tenure"],
		bins=[-1, 2, 5, 10],
		labels=["new", "developing", "loyal"],
		include_lowest=True,
	).astype("object")

	# Flag customers in the upper quartile of balance or salary as high-value.
	high_balance_threshold = df["balance"].quantile(0.75)
	high_salary_threshold = df["estimated_salary"].quantile(0.75)
	df["high_value_customer"] = (
		(df["balance"] >= high_balance_threshold)
		| (df["estimated_salary"] >= high_salary_threshold)
	).astype(int)

	return df
