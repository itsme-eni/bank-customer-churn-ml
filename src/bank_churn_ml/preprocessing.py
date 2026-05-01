"""Preprocessing and train/test split helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# This module creates reusable train/test and preprocessing artifacts so model
# scripts can focus on training logic instead of data plumbing.


def split_features_target(
	dataframe: pd.DataFrame,
	target_column: str,
	id_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
	"""Split feature matrix and target vector, dropping identifier columns."""
	# Ensure training target exists before creating X/y splits.
	if target_column not in dataframe.columns:
		raise ValueError(f"Target column '{target_column}' not found")

	# Remove target and non-predictive identifiers from feature matrix.
	feature_dataframe = dataframe.drop(columns=[target_column]).copy()
	if id_column and id_column in feature_dataframe.columns:
		feature_dataframe = feature_dataframe.drop(columns=[id_column])

	# Cast to int so metrics/model expectations stay consistent.
	target_series = dataframe[target_column].astype(int)
	return feature_dataframe, target_series


def train_test_split_stratified(
	X: pd.DataFrame,
	y: pd.Series,
	test_size: float,
	random_state: int,
	stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Perform a deterministic train/test split with optional stratification."""
	# Stratification preserves churn distribution between train and test sets.
	stratify_series = y if stratify else None
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify_series,
	)
	return X_train, X_test, y_train, y_test


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
	"""Infer categorical and numerical feature names from a DataFrame."""
	# Include modern pandas string dtype to avoid pandas 3 migration issues.
	categorical_features = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
	numerical_features = [column for column in X.columns if column not in categorical_features]
	return categorical_features, numerical_features


def build_preprocessor(
	categorical_features: list[str],
	numerical_features: list[str],
) -> ColumnTransformer:
	"""Build sklearn ColumnTransformer for categorical and numeric preprocessing."""
	# Categorical path: impute missing values then one-hot encode.
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", OneHotEncoder(handle_unknown="ignore")),
		]
	)
	# Numerical path: impute with median then standardize.
	numerical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)

	# Keep only declared columns to avoid accidental leakage.
	return ColumnTransformer(
		transformers=[
			("categorical", categorical_pipeline, categorical_features),
			("numerical", numerical_pipeline, numerical_features),
		],
		remainder="drop",
	)


def prepare_training_matrices(
	dataframe: pd.DataFrame,
	target_column: str,
	id_column: str,
	test_size: float,
	random_state: int,
	stratify: bool = True,
) -> dict[str, Any]:
	"""Run end-to-end split prep and return all key training objects."""
	# Build X/y from the full prepared dataset.
	X, y = split_features_target(dataframe, target_column=target_column, id_column=id_column)
	# Create reproducible train/test partitions.
	X_train, X_test, y_train, y_test = train_test_split_stratified(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify,
	)
	# Infer feature groups from train split only.
	categorical_features, numerical_features = infer_feature_types(X_train)
	preprocessor = build_preprocessor(categorical_features, numerical_features)

	return {
		"X_train": X_train,
		"X_test": X_test,
		"y_train": y_train,
		"y_test": y_test,
		"categorical_features": categorical_features,
		"numerical_features": numerical_features,
		"preprocessor": preprocessor,
	}
