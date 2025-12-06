import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
RANDOM_STATE = 42

# CAPSTONE UPGRADE: Wider search space + max_features
PARAM_DIST = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'], # Try different feature sampling
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# More iterations for better tuning
N_ITER = 30  
CV_FOLDS = 5 # Standard academic rigor

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("Loading data...")
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_train = np.load('data/processed/y_train.npy')
y_val = np.load('data/processed/y_val.npy')

preprocessor = joblib.load('models/preprocessor.joblib')
feature_names = preprocessor.get_feature_names_out()

# ---------------------------------------------------------
# 2. HYPERPARAMETER TUNING
# ---------------------------------------------------------
print(f"Starting Randomized Search (n_iter={N_ITER}, cv={CV_FOLDS})...")
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, oob_score=True) # Enable OOB

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=PARAM_DIST,
    n_iter=N_ITER,
    cv=CV_FOLDS,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring='f1'
)

rf_random.fit(X_train, y_train)

best_model = rf_random.best_estimator_
print(f"\nBest Parameters found: {rf_random.best_params_}")

# ---------------------------------------------------------
# 3. OVERFITTING CHECK (Crucial for Analysis)
# ---------------------------------------------------------
print("\n=== Overfitting Check ===")
train_acc = best_model.score(X_train, y_train)
val_acc = best_model.score(X_val, y_val)
oob_score = best_model.oob_score_

print(f"Training Accuracy:   {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"OOB Score:           {oob_score:.4f}")
print(f"Gap (Train - Val):   {train_acc - val_acc:.4f}")

if (train_acc - val_acc) > 0.10:
    print("WARNING: High overfitting detected. Consider increasing min_samples_leaf or reducing max_depth in future runs.")

# ---------------------------------------------------------
# 4. DETAILED EVALUATION
# ---------------------------------------------------------
print("\nEvaluating on Validation Set...")
y_pred = best_model.predict(X_val)
y_proba = best_model.predict_proba(X_val)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_val, y_pred),
    "Precision": precision_score(y_val, y_pred),
    "Recall": recall_score(y_val, y_pred),
    "F1 Score": f1_score(y_val, y_pred),
    "ROC AUC": roc_auc_score(y_val, y_proba)
}

print("\n=== Validation Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Save detailed report
with open('docs/rf_metrics.txt', 'w') as f:
    f.write("Random Forest Validation Metrics\n")
    f.write("================================\n")
    f.write(f"Best Params: {rf_random.best_params_}\n\n")
    f.write(f"Training Accuracy:   {train_acc:.4f}\n")
    f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Generalization Gap:  {train_acc - val_acc:.4f}\n\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

# ---------------------------------------------------------
# 5. FEATURE IMPORTANCE (Enhanced)
# ---------------------------------------------------------
print("\nGenerating Feature Importance Plot...")
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20

plt.figure(figsize=(12, 8))
plt.title("Top 20 Feature Importances (Random Forest)", fontsize=14)
plt.barh(range(top_n), importances[indices][:top_n], align="center", color='#2ecc71')
plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]], fontsize=10)
plt.xlabel("Gini Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('docs/rf_feature_importance.png', dpi=300)

joblib.dump(best_model, 'models/rf_best.joblib')
print("\n✓ Comprehensive analysis complete. Model saved.")
