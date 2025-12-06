import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
RANDOM_STATE = 42

# Fine-grained search space for boosting
PARAM_DIST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 8],          # Trees are usually shallower in boosting
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],     # Prevent overfitting by sampling rows
    'colsample_bytree': [0.7, 0.8, 1.0],   # Prevent overfitting by sampling columns
    'gamma': [0, 0.1, 0.5],                # Minimum loss reduction for split
    'scale_pos_weight': [1, 3]             # 1 = Standard, 3 ≈ Balanced (since neg/pos ratio is ~3:1)
}

N_ITER = 30
CV_FOLDS = 5

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
print(f"Starting Randomized Search (n_iter={N_ITER})...")

# XGBoost Scikit-Learn Wrapper
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# We need a custom setup for Early Stopping within CV, but for simplicity in RandomizedSearch,
# we usually drop early_stopping or perform it during the final fit. 
# Here we will tune first, then refine with early stopping.

xgb_random = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=PARAM_DIST,
    n_iter=N_ITER,
    cv=CV_FOLDS,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring='f1' # Continue optimizing for F1
)

xgb_random.fit(X_train, y_train)

best_params = xgb_random.best_params_
print(f"\nBest Parameters found: {best_params}")

# ---------------------------------------------------------
# 3. RE-TRAIN WITH EARLY STOPPING (Refinement)
# ---------------------------------------------------------
print("\nRefining Best Model with Early Stopping...")
final_model = xgb.XGBClassifier(
    **best_params,
    objective='binary:logistic',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='logloss',
    early_stopping_rounds=20 # Stop if val score plateaus for 20 rounds
)

# Fit using the explicit validation set for early stopping
final_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

# ---------------------------------------------------------
# 4. OVERFITTING CHECK
# ---------------------------------------------------------
results = final_model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

train_loss = results['validation_0']['logloss'][-1]
val_loss = results['validation_1']['logloss'][-1]

print("\n=== Overfitting Check (Log Loss) ===")
print(f"Training Loss:   {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Gap:             {val_loss - train_loss:.4f}")

# Plot Learning Curve (Classic Boosting Analysis)
plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
plt.legend()
plt.ylabel('Log Loss')
plt.xlabel('Estimators (Iterations)')
plt.title('XGBoost Learning Curve')
plt.grid(True, alpha=0.3)
plt.savefig('docs/xgb_learning_curve.png', dpi=300)
print("✓ Learning curve saved to docs/xgb_learning_curve.png")

# ---------------------------------------------------------
# 5. DETAILED EVALUATION
# ---------------------------------------------------------
print("\nEvaluating on Validation Set...")
y_pred = final_model.predict(X_val)
y_proba = final_model.predict_proba(X_val)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_val, y_pred),
    "Precision": precision_score(y_val, y_pred),
    "Recall": recall_score(y_val, y_pred),
    "F1 Score": f1_score(y_val, y_pred),
    "ROC AUC": roc_auc_score(y_val, y_proba)
}

print("\n=== XGBoost Validation Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Save detailed report
with open('docs/xgb_metrics.txt', 'w') as f:
    f.write("XGBoost Validation Metrics\n")
    f.write("==========================\n")
    f.write(f"Best Params: {best_params}\n\n")
    f.write(f"Final Estimators: {final_model.best_iteration}\n")
    f.write(f"Validation Loss:  {val_loss:.4f}\n\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

# ---------------------------------------------------------
# 6. FEATURE IMPORTANCE (Gain vs Weight)
# ---------------------------------------------------------
print("\nGenerating Feature Importance Plot...")
# XGBoost calculates importance by "Gain" (improvement in accuracy brought by a feature)
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20

plt.figure(figsize=(12, 8))
plt.title("Top 20 Feature Importances (XGBoost - Gain)", fontsize=14)
plt.barh(range(top_n), importances[indices][:top_n], align="center", color='#e67e22') # Orange for XGB
plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]], fontsize=10)
plt.xlabel("Importance (Gain)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('docs/xgb_feature_importance.png', dpi=300)

joblib.dump(final_model, 'models/xgb_best.joblib')
print("\n✓ Phase 2 Complete. XGBoost Model saved.")
