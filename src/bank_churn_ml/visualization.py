"""Visualization utilities for EDA and model reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save_current_figure(output_path: Path) -> None:
	"""Save the current matplotlib figure and close it."""
	# Ensure destination folder exists before writing image files.
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(output_path, dpi=150, bbox_inches="tight")
	plt.close()


def generate_initial_eda_figures(dataframe: pd.DataFrame, output_dir: Path | str) -> list[Path]:
	"""Generate a core set of initial EDA charts and save them to disk."""
	# Use a clean and readable style for portfolio-ready figures.
	sns.set_theme(style="whitegrid")
	out_dir = Path(output_dir)
	saved_paths: list[Path] = []

	# Plot churn class distribution.
	plt.figure(figsize=(8, 5))
	sns.countplot(data=dataframe, x="churn", hue="churn", legend=False, palette="Blues")
	plt.title("Target Distribution: Churn")
	plt.xlabel("Churn (0 = No, 1 = Yes)")
	plt.ylabel("Count")
	target_path = out_dir / "target_distribution.png"
	_save_current_figure(target_path)
	saved_paths.append(target_path)

	# Plot churn rate by country.
	plt.figure(figsize=(8, 5))
	country_rate = dataframe.groupby("country", as_index=False)["churn"].mean().sort_values("churn", ascending=False)
	sns.barplot(data=country_rate, x="country", y="churn", hue="country", legend=False, palette="viridis")
	plt.title("Churn Rate by Country")
	plt.xlabel("Country")
	plt.ylabel("Churn Rate")
	country_path = out_dir / "churn_by_country.png"
	_save_current_figure(country_path)
	saved_paths.append(country_path)

	# Plot churn rate by gender.
	plt.figure(figsize=(7, 5))
	gender_rate = dataframe.groupby("gender", as_index=False)["churn"].mean().sort_values("churn", ascending=False)
	sns.barplot(data=gender_rate, x="gender", y="churn", hue="gender", legend=False, palette="Set2")
	plt.title("Churn Rate by Gender")
	plt.xlabel("Gender")
	plt.ylabel("Churn Rate")
	gender_path = out_dir / "churn_by_gender.png"
	_save_current_figure(gender_path)
	saved_paths.append(gender_path)

	# Plot churn rate by active member flag.
	plt.figure(figsize=(7, 5))
	active_rate = dataframe.groupby("active_member", as_index=False)["churn"].mean().sort_values("active_member")
	sns.barplot(data=active_rate, x="active_member", y="churn", hue="active_member", legend=False, palette="mako")
	plt.title("Churn Rate by Active Member Status")
	plt.xlabel("Active Member (0 = No, 1 = Yes)")
	plt.ylabel("Churn Rate")
	active_path = out_dir / "churn_by_active_member.png"
	_save_current_figure(active_path)
	saved_paths.append(active_path)

	# Plot age distribution by churn class.
	plt.figure(figsize=(8, 5))
	sns.boxplot(data=dataframe, x="churn", y="age", hue="churn", legend=False, palette="coolwarm")
	plt.title("Age vs Churn")
	plt.xlabel("Churn (0 = No, 1 = Yes)")
	plt.ylabel("Age")
	age_path = out_dir / "age_vs_churn.png"
	_save_current_figure(age_path)
	saved_paths.append(age_path)

	# Plot balance distribution by churn class.
	plt.figure(figsize=(8, 5))
	sns.boxplot(data=dataframe, x="churn", y="balance", hue="churn", legend=False, palette="rocket")
	plt.title("Balance vs Churn")
	plt.xlabel("Churn (0 = No, 1 = Yes)")
	plt.ylabel("Balance")
	balance_path = out_dir / "balance_vs_churn.png"
	_save_current_figure(balance_path)
	saved_paths.append(balance_path)

	# Plot correlation heatmap for numeric features.
	numeric_df = dataframe.select_dtypes(include=["number"])
	plt.figure(figsize=(11, 8))
	sns.heatmap(numeric_df.corr(numeric_only=True), cmap="RdBu_r", center=0, annot=False)
	plt.title("Correlation Heatmap (Numeric Features)")
	corr_path = out_dir / "correlation_heatmap.png"
	_save_current_figure(corr_path)
	saved_paths.append(corr_path)

	return saved_paths
