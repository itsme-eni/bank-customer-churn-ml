# Bank Customer Churn ML - Pipeline Status

Last updated: 2026-05-01

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
Status: Partially done
Implemented core model factory/training comparison in:
- [src/bank_churn_ml/modeling.py](src/bank_churn_ml/modeling.py)
Remaining:
- End-to-end training CLI in [scripts/train_model.py](scripts/train_model.py)

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
Status: Not done
Target output:
- [reports/metrics/model_comparison.csv](reports/metrics)

### 9. Save best model
Status: Not done
Target output:
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

Generated when figure script is run:
- Intermediate plots folder: [reports/figures/intermediate](reports/figures)

Not generated yet:
- Best model artifact: [models](models)

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
3. pytest -q

### D. Quick output checks
1. dir data/processed
2. dir reports/figures
3. dir reports/figures/intermediate
4. dir reports/metrics
5. dir models

## 4) Next Implementation Priority

1. Implement [scripts/train_model.py](scripts/train_model.py)
2. Implement [scripts/evaluate_model.py](scripts/evaluate_model.py)
3. Implement [scripts/predict.py](scripts/predict.py)
4. Expand [README.md](README.md) to full professional portfolio version
5. Populate notebooks with final narrative and interpretation
