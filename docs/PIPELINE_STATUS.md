# Bank Customer Churn ML - Pipeline Status

Last updated: 2026-05-02

## 0) Phase to Script Mapping

- Phase 1
	Purpose: Raw data load and validation
	Script(s): `scripts/prepare_data.py`
	Status: Done

- Phase 2
	Purpose: EDA figures and EDA summaries
	Script(s): `scripts/generate_intermediate_figures.py`
	Status: Done (baseline EDA scope)

- Phase 3
	Purpose: Feature engineering application
	Script(s): `scripts/prepare_data.py`, `scripts/export_intermediate_results.py`
	Status: Done (core)

- Phase 4
	Purpose: Preprocessing and split summaries
	Script(s): `scripts/export_intermediate_results.py`
	Status: Done (core)

- Phase 5
	Purpose: Model training and selection
	Script(s): `scripts/train_model.py`
	Status: Done (baseline training scope)

- Phase 6
	Purpose: Evaluation metrics and curves
	Script(s): `scripts/evaluate_model.py`
	Status: Done

- Phase 7
	Purpose: Inference/prediction CLI
	Script(s): `scripts/predict.py`
	Status: Done

## 0.1) Script Quick Reference

- `scripts/prepare_data.py`
	Output: Processed dataset with engineered features.

- `scripts/generate_intermediate_figures.py`
	Output: Intermediate EDA plots and EDA summary CSV tables.

- `scripts/export_intermediate_results.py`
	Output: Phase-wise intermediate artifacts for completed phases.

- `scripts/train_model.py`
	Output: Trains candidate models, saves best model, and exports model comparison CSV.

- `scripts/evaluate_model.py`
	Output: Phase 6 metrics summary (CSV/JSON), confusion matrices, ROC and PR curves.

- `scripts/predict.py`
	Output: Prediction CSV and run-level metrics JSON for scored input rows.

## 1) Original Scope Tracking

### 1. Load and validate raw Kaggle dataset
Status: Done
Implemented in:
- [src/bank_churn_ml/data_loading.py](src/bank_churn_ml/data_loading.py)
- [src/bank_churn_ml/validation.py](src/bank_churn_ml/validation.py)
- [scripts/prepare_data.py](scripts/prepare_data.py)

### 2. Perform exploratory data analysis
Status: Done (baseline EDA scope)
Implemented so far:
- Initial EDA figures generator
- Intermediate figures script
- Reproducible EDA summary table exports (overview, missing values, numeric summary, categorical summary)
- Notebook narrative and business interpretation sections
Implemented in:
- [src/bank_churn_ml/visualization.py](src/bank_churn_ml/visualization.py)
- [scripts/generate_intermediate_figures.py](scripts/generate_intermediate_figures.py)
- [notebooks/01_exploratory_data_analysis.ipynb](notebooks/01_exploratory_data_analysis.ipynb)

### 3. Clean and preprocess data
Status: Done (core)
Implemented in:
- [src/bank_churn_ml/preprocessing.py](src/bank_churn_ml/preprocessing.py)

### 4. Engineer useful features
Status: Done (core)
Implemented in:
- [src/bank_churn_ml/features.py](src/bank_churn_ml/features.py)
Features currently implemented:
- balance_to_salary_ratio
- products_per_tenure
- age_group
- tenure_group
- high_value_customer

### 5. Train multiple ML models
Status: Done (baseline training scope)
Implemented in:
- [src/bank_churn_ml/modeling.py](src/bank_churn_ml/modeling.py)
- [scripts/train_model.py](scripts/train_model.py)
- [reports/metrics/model_comparison.csv](reports/metrics/model_comparison.csv)
- [models/best_model.joblib](models/best_model.joblib)

### 6. Evaluate with classification metrics
Status: Done
Implemented in:
- [src/bank_churn_ml/evaluation.py](src/bank_churn_ml/evaluation.py)
- [scripts/evaluate_model.py](scripts/evaluate_model.py)
Key outputs:
- [reports/metrics/evaluation_summary.csv](reports/metrics/evaluation_summary.csv)
- [reports/metrics/evaluation_summary.json](reports/metrics/evaluation_summary.json)
- [reports/figures/evaluation](reports/figures/evaluation)

### 7. Handle class imbalance
Status: Partially done
Current handling:
- class_weight='balanced' for Logistic Regression, Random Forest, SVM
Defined in:
- [src/bank_churn_ml/modeling.py](src/bank_churn_ml/modeling.py)

### 8. Model comparison table export
Status: Done
Output:
- [reports/metrics/model_comparison.csv](reports/metrics)

### 9. Save best model
Status: Done
Output:
- [models/best_model.joblib](models)

### 10. Prediction script for new data
Status: Done
Implemented in:
- [scripts/predict.py](scripts/predict.py)
Key outputs:
- [reports/metrics/predictions](reports/metrics/predictions)

### 11. Reproducible CLI pipeline
Status: Done (core scripts)
Ready now:
- data prep
- intermediate EDA figure generation
- train
- evaluate
- predict

### 12. Professional recruiter-ready README
Status: Done (expanded)
Current file:
- [README.md](README.md)

### 14. Notebook narrative coverage
Status: Done (core)
Completed notebooks:
- [notebooks/01_exploratory_data_analysis.ipynb](notebooks/01_exploratory_data_analysis.ipynb)
- [notebooks/02_model_development.ipynb](notebooks/02_model_development.ipynb)
- [notebooks/03_model_interpretation.ipynb](notebooks/03_model_interpretation.ipynb)

### 13. Automated test coverage
Status: Expanded (core unit coverage)
Current test files:
- [tests/test_data_loading.py](tests/test_data_loading.py)
- [tests/test_preprocessing.py](tests/test_preprocessing.py)
- [tests/test_modeling.py](tests/test_modeling.py)
- [tests/test_predict.py](tests/test_predict.py)
Latest run:
- `9 passed`

## 2) Current Artifacts

Available now:
- Processed dataset: [data/processed/processed_bank_churn.csv](data/processed/processed_bank_churn.csv)
- EDA summary tables: [reports/metrics](reports/metrics)
- Script logs (rich console + file): [reports/logs](reports/logs)

Generated when figure script is run:
- Intermediate plots folder: [reports/figures/intermediate](reports/figures)

Generated after Phase 5 training:
- Best model artifact: [models/best_model.joblib](models/best_model.joblib)
- Model comparison table: [reports/metrics/model_comparison.csv](reports/metrics/model_comparison.csv)

Generated after Phase 6 evaluation:
- Evaluation summary CSV: [reports/metrics/evaluation_summary.csv](reports/metrics/evaluation_summary.csv)
- Evaluation summary JSON: [reports/metrics/evaluation_summary.json](reports/metrics/evaluation_summary.json)
- Evaluation plots (ROC/PR/confusion): [reports/figures/evaluation](reports/figures/evaluation)

Generated after Phase 7 prediction:
- Prediction outputs folder: [reports/metrics/predictions](reports/metrics/predictions)
- Example predictions file: [reports/metrics/predictions/processed_data_predictions.csv](reports/metrics/predictions/processed_data_predictions.csv)
- Example run metrics: [reports/metrics/predictions/processed_data_predictions_run_metrics.json](reports/metrics/predictions/processed_data_predictions_run_metrics.json)

## 3) Exact Terminal Commands (PowerShell)

Run from project root:
`<repo-root>`

### A. Activate conda env in PowerShell (session-safe)
1. Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
2. conda init powershell  (one-time setup)
3. Restart PowerShell
4. conda activate bank-churn-ml

### B. Install dependencies
1. python -m pip install -r requirements.txt

### C. Run what is implemented today
1. python scripts/prepare_data.py
2. python scripts/generate_intermediate_figures.py
3. python scripts/export_intermediate_results.py
4. python scripts/train_model.py
5. python scripts/evaluate_model.py
6. python scripts/predict.py --input data/processed/processed_bank_churn.csv --output reports/metrics/predictions/processed_data_predictions.csv
7. pytest -q

### D. Quick output checks
1. dir data/processed
2. dir reports/figures
3. dir reports/figures/intermediate
4. dir reports/metrics
5. dir reports/metrics/intermediate
6. dir models

## 4) Next Implementation Priority

1. Add higher-level CLI integration tests (optional but recommended)
2. Add model monitoring and drift-check template artifacts (optional)
3. Package a lightweight demo app or report dashboard (optional)
