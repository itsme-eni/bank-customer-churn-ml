# Bank Customer Churn ML - Pipeline Status

Last updated: 2026-05-01

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
	Status: In progress

- Phase 7
	Purpose: Inference/prediction CLI
	Script(s): `scripts/predict.py`
	Status: Not done

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
	Output: Placeholder (to be implemented).

- `scripts/predict.py`
	Output: Placeholder (to be implemented).

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
Status: Partially done
Implemented helper metrics table/confusion utilities in:
- [src/bank_churn_ml/evaluation.py](src/bank_churn_ml/evaluation.py)
Remaining:
- End-to-end evaluation CLI in [scripts/evaluate_model.py](scripts/evaluate_model.py)
- ROC/PR/confusion figure exports

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
Status: Not done
Target script:
- [scripts/predict.py](scripts/predict.py)

### 11. Reproducible CLI pipeline
Status: In progress
Ready now:
- data prep
- intermediate EDA figure generation
Pending:
- train/evaluate/predict scripts

### 12. Professional recruiter-ready README
Status: Not done yet
Current file:
- [README.md](README.md)

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

## 3) Exact Terminal Commands (PowerShell)

Run from project root:
C:/Users/eniko/Documents/coding_projects/bank-customer-churn-ml

### A. Activate conda env in PowerShell (session-safe)
1. Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
2. & "C:/Users/eniko/miniconda3/shell/condabin/conda-hook.ps1"
3. conda activate bank-churn-ml

### B. Install dependencies
1. python -m pip install -r requirements.txt

### C. Run what is implemented today
1. python scripts/prepare_data.py
2. python scripts/generate_intermediate_figures.py
3. python scripts/export_intermediate_results.py
4. python scripts/train_model.py
5. pytest -q

### D. Quick output checks
1. dir data/processed
2. dir reports/figures
3. dir reports/figures/intermediate
4. dir reports/metrics
5. dir reports/metrics/intermediate
6. dir models

## 4) Next Implementation Priority

1. Implement [scripts/evaluate_model.py](scripts/evaluate_model.py)
2. Implement [scripts/predict.py](scripts/predict.py)
3. Expand tests for training/evaluation scripts
4. Expand [README.md](README.md) to full professional portfolio version
5. Populate notebooks with final narrative and interpretation
