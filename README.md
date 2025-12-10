# Adult Income Classification

Capstone project: Prediction of income brackets using machine learning on Adult Census dataset

## 1. Project Overview

This project aims to build a machine learning model to predict whether an individual's income exceeds $50K per year based on census data. The Adult Census Income dataset contains demographic and employment-related features that will be used to train and evaluate classification models.

## 2.  Project Structure

```
adult-income-classification/
│
├── data/                  # Raw and processed data files
│
├── notebooks/             # Jupyter notebooks for exploration and analysis
│
├── scripts/               # Python scripts for data processing and modeling
│
├── models/                # Trained models and model artifacts
│
├── docs/                  # Project documentation
│
├── requirements.txt       # Python package dependencies
├── .gitignore            # Git ignore file
└── README.md             # Project README
```

## Project Goals

1. **Data Exploration**: Analyze the Adult Census dataset to understand feature distributions and relationships
2. **Data Preprocessing**: Clean and prepare data for machine learning models
3. **Feature Engineering**: Create meaningful features to improve model performance
4. **Model Development**: Train and evaluate multiple classification algorithms
5. **Model Optimization**: Tune hyperparameters and select the best performing model
6. **Documentation**: Maintain comprehensive documentation of methodology and results

---

## 3. Dataset & Provenance

- **Name:** Adult / Census Income dataset  
- **Source:** UCI Machine Learning Repository  
- **Records:** 48,842 rows (train + test combined)  
- **Features:** 14 attributes (6 numeric, 8 categorical) plus binary income label  
- **Task:** Predict whether `income` is `>50K` or `<=50K` annually.[file:173]

You must manually download the dataset:

1. Visit the Adult dataset page on the UCI Machine Learning Repository.[file:173]
2. Download:
   - `adult.data`  → save as `adult_train.csv`
   - `adult.test`  → save as `adult_test.csv`
3. Place both files under `data/raw/`.

---

## 4. Environment Setup

### 4.1. Clone the Repository

```

git clone https://github.com/<YOUR_USERNAME>/adult-income-classification.git
cd adult-income-classification

```

### 4.2. Create and Activate Virtual Environment (Windows)

```

python -m venv .venv
.venv\Scripts\activate

```

On macOS / Linux:

```

python3 -m venv .venv
source .venv/bin/activate

```

### 4.3. Install Dependencies

```

pip install --upgrade pip
pip install -r requirements.txt

```

Main libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `jupyter`, `joblib`.[file:173]

---

## 5. Reproducible Workflow

Below is the recommended order to reproduce all results.

### Step 1 – Data Preprocessing & Splits

Creates a fitted scikit‑learn `ColumnTransformer`, deterministic 70/15/15 splits, and transformed numpy arrays.

```

python scripts/preprocessing.py

```

Outputs:

- `models/preprocessor.joblib`
- `data/processed/X_train.npy`, `X_val.npy`, `X_test.npy`
- `data/processed/y_train.npy`, `y_val.npy`, `y_test.npy`

This script:

- Combines `adult_train.csv` and `adult_test.csv`.
- Treats `?` as missing values and imputes numeric (median) and categorical (most frequent) features.
- Applies standard scaling to numeric columns and one‑hot encoding to categorical columns.
- Uses `random_state=42` and stratified sampling for all splits.[file:173]

### Step 2 – Baseline: Logistic Regression

```

python scripts/train_baseline.py

```

Outputs:

- `models/logreg_baseline.joblib`
- `docs/baseline_metrics.txt`
- `docs/confusion_matrix_val.png`

Trains a logistic regression classifier and reports Accuracy, Precision, Recall, F1, ROC‑AUC, plus a confusion matrix.

### Step 3 – Random Forest (Phase 1 Ensemble)

```

python scripts/train_rf.py

```

Outputs:

- `models/rf_best.joblib`
- `docs/rf_metrics.txt`
- `docs/rf_feature_importance.png`

Performs `RandomizedSearchCV` over key Random Forest hyperparameters (e.g., `n_estimators`, `max_depth`, `min_samples_split`, `class_weight`) using F1 as the scoring metric, reports train/validation accuracy and OOB score, and saves a top‑20 feature importance plot.

### Step 4 – XGBoost (Phase 2 Ensemble)

```

python scripts/train_gb.py

```

Outputs:

- `models/xgb_best.joblib`
- `docs/xgb_metrics.txt`
- `docs/xgb_learning_curve.png`
- `docs/xgb_feature_importance.png`

This script tunes XGBoost with `RandomizedSearchCV`, then refits the best configuration with early stopping using the validation set, generating learning‑curve and feature‑importance plots.

### Step 5 – Error & Feature Analysis

```

python scripts/error_analysis.py

```

Outputs:

- `docs/error_confidence.png`
- `docs/feature_importance.png`
- `docs/error_analysis_summary.txt`

Analyzes false positives/negatives, prediction confidence distributions, and high‑impact features.

---

## 6. Notebooks

All key analysis steps are reproduced in Jupyter notebooks.

1. **`01_eda.ipynb`** – Exploratory data analysis: distributions, missingness, correlations, and class balance.[file:173]  
2. **`02_test_preprocessing.ipynb`** – Sanity checks on preprocessing and transformed arrays.  
3. **`03_modeling_rf_gb.ipynb`** – Random Forest and XGBoost training, tuning, and evaluation.  
4. **`04_model_comparison.ipynb`** – Loads all three models, produces a comparison table and joint ROC curves.  
5. **`05_fairness_analysis.ipynb`** – Subgroup performance by sex and race (accuracy, recall, precision, positive prediction rate) with fairness plots.

Launch notebooks with:

```

.venv\Scripts\activate   \# or `source .venv/bin/activate`
jupyter notebook

```

---

## 7. Results Summary (High Level)

- **Best overall model:** XGBoost (`models/xgb_best.joblib`)
- **Typical validation performance (approximate):**
  - Accuracy ≈ 0.86
  - Precision ≈ 0.74
  - Recall ≈ 0.79
  - F1 ≈ 0.77
  - ROC‑AUC ≈ 0.93

Random Forest tends to achieve the highest recall but lower precision, while Logistic Regression often achieves the highest precision but lowest recall, illustrating the standard precision–recall trade‑off on imbalanced datasets.

Fairness analysis indicates substantially higher recall and positive prediction rates for males compared with females, and for White individuals compared with some minority groups, highlighting important ethical considerations for real‑world deployment.

---

## 8. Reproducibility & Integrity

- All random processes use `random_state=42` for reproducibility.  
- Raw data under `data/raw/` is not committed; users must obtain it from the original UCI source.
- External libraries and the Adult dataset should be cited appropriately in reports and presentations. 
- Any use of AI tooling for code or writing assistance should be disclosed in the project appendix, following course academic‑integrity guidelines.

---

## 9. Contributing

This repository is primarily for a course capstone but supports collaboration:

1. Create a feature branch (`git checkout -b feature/my-change`).  
2. Implement and test your changes.  
3. Open a Pull Request describing your modifications and rationale.

---

## 10. License

This project is for educational and research purposes only as part of an academic capstone; no commercial use is intended.
```
