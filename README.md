# Bank Customer Churn ML

End-to-end machine learning portfolio project for predicting customer churn using structured bank customer data.

## 1) Business Problem

Customer churn is one of the highest-impact revenue risks in retail banking. This project predicts which customers are most likely to churn so retention teams can:

- prioritize outreach to high-risk customers,
- tune campaign aggressiveness based on business goals,
- balance precision (cost control) vs recall (churn capture).

## 2) Project Goals

- Build a reproducible ML pipeline from data preparation to prediction CLI.
- Compare multiple baseline models under consistent preprocessing.
- Move from a single "best model" mindset to an operational policy:
	- keep Gradient Boosting as primary production baseline,
	- apply threshold tuning for higher recall where needed,
	- retain SVM as high-recall challenger for aggressive campaigns.

## 3) Repository Structure

```text
config/
data/
models/
notebooks/
reports/
scripts/
src/bank_churn_ml/
tests/
```

Key files:

- `scripts/prepare_data.py`: data loading, validation, feature engineering output.
- `scripts/generate_intermediate_figures.py`: EDA charts and summaries.
- `scripts/train_model.py`: training, comparison, and policy strategy export.
- `scripts/evaluate_model.py`: Phase 6 evaluation metrics and plots.
- `scripts/predict.py`: Phase 7 inference CLI with threshold options.
- `docs/PIPELINE_STATUS.md`: up-to-date implementation status and command reference.

## 4) Technical Stack

- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- PyYAML
- joblib
- pytest
- rich

## 5) ML Pipeline Overview

### Phase 1: Data Preparation

- Load raw CSV with schema normalization.
- Validate required columns and binary target.
- Save processed dataset with engineered features.

### Phase 2: Exploratory Data Analysis

- Export EDA summary tables.
- Generate business-facing plots (distribution, segment churn rates, heatmap).

### Phase 3-4: Feature Engineering + Preprocessing

Engineered features include:

- `balance_to_salary_ratio`
- `products_per_tenure`
- `age_group`
- `tenure_group`
- `high_value_customer`

Preprocessing uses:

- categorical imputation + one-hot encoding,
- numerical imputation + scaling,
- deterministic stratified train/test split.

### Phase 5: Model Development

Compared models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM

Primary ranking metric: ROC-AUC.

### Phase 6: Evaluation

- Metric summary export (CSV + JSON).
- ROC and Precision-Recall curves.
- Confusion matrix plots for key strategies.

### Phase 7: Prediction CLI

- Predict on new data using `auto`, `primary`, `best`, or `svm` model selection.
- Threshold handling with `auto`, `default`, `policy`, or manual override.
- Exports prediction CSV and run-level metrics JSON.

## 6) Model Performance Snapshot

From `reports/metrics/model_comparison.csv` (test split):

| Model | Test ROC-AUC | Test Recall | Test Precision |
|---|---:|---:|---:|
| Gradient Boosting | 0.8718 | 0.4840 | 0.8107 |
| Random Forest | 0.8485 | 0.4398 | 0.7817 |
| SVM | 0.8429 | 0.7322 | 0.4992 |
| Logistic Regression | 0.7901 | 0.6781 | 0.4077 |

Policy-driven evaluation from `reports/metrics/evaluation_summary.csv`:

- Gradient Boosting (default threshold 0.5):
	- Recall: 0.4840
	- Precision: 0.8107
- Gradient Boosting (policy threshold 0.1317):
	- Recall: 0.8600
	- Precision: 0.4018
- SVM (default threshold 0.5):
	- Recall: 0.5086
	- Precision: 0.6699

## 7) Decision Policy (Operational)

Stored in `reports/metrics/model_decision_strategy.json`.

- **Primary baseline:** Gradient Boosting
- **Threshold objective:** increase recall while maintaining precision >= 0.40
- **Recommended threshold:** 0.1317
- **High-recall challenger:** SVM

This allows deployment to shift between conservative and aggressive retention modes without retraining.

## 8) Reproducibility

### Environment Setup (PowerShell + conda)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
conda init powershell  # one-time setup
# Restart PowerShell after init, then activate your environment.
conda activate bank-churn-ml
python -m pip install -r requirements.txt
```

### End-to-End Commands

```powershell
python scripts/prepare_data.py
python scripts/generate_intermediate_figures.py
python scripts/export_intermediate_results.py
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/predict.py --input data/processed/processed_bank_churn.csv --output reports/metrics/predictions/processed_data_predictions.csv
pytest -q
```

## 9) Main Artifacts

- Processed data: `data/processed/processed_bank_churn.csv`
- Trained models:
	- `models/best_model.joblib`
	- `models/primary_model_gradient_boosting.joblib`
	- `models/high_recall_model_svm.joblib`
- Metrics:
	- `reports/metrics/model_comparison.csv`
	- `reports/metrics/model_decision_strategy.json`
	- `reports/metrics/evaluation_summary.csv`
	- `reports/metrics/evaluation_summary.json`
- Evaluation figures:
	- `reports/figures/evaluation/roc_curves.png`
	- `reports/figures/evaluation/precision_recall_curves.png`
	- `reports/figures/evaluation/confusion_matrix_gradient_boosting_default_threshold.png`
	- `reports/figures/evaluation/confusion_matrix_gradient_boosting_policy_threshold.png`
	- `reports/figures/evaluation/confusion_matrix_svm_default_threshold.png`

## 10) Testing

Current automated tests cover data loading, preprocessing, modeling, and prediction helper logic.

```powershell
pytest -q
```

## 11) Notebook Guides

- `notebooks/01_exploratory_data_analysis.ipynb`: EDA workflow and findings.
- `notebooks/02_model_development.ipynb`: modeling workflow and comparison narrative.
- `notebooks/03_model_interpretation.ipynb`: strategy interpretation and threshold trade-off analysis.

## 12) Current Status

Core pipeline phases (1-7) are implemented.

See `docs/PIPELINE_STATUS.md` for the latest execution status and next priorities.
