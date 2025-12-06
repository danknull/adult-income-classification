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
# Speedrun Grid: Small enough to run fast, wide enough to find good params
PARAM_DIST = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}
N_ITER = 15  # Number of random combinations to try (Speedrun setting)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("Loading data...")
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_train = np.load('data/processed/y_train.npy')
y_val = np.load('data/processed/y_val.npy')

# Load preprocessor to get feature names later
preprocessor = joblib.load('models/preprocessor.joblib')
feature_names = preprocessor.get_feature_names_out()

# ---------------------------------------------------------
# 2. HYPERPARAMETER TUNING (RandomizedSearchCV)
# ---------------------------------------------------------
print(f"Starting Randomized Search (n_iter={N_ITER})...")
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1) # n_jobs=-1 uses all CPU cores

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=PARAM_DIST,
    n_iter=N_ITER,
    cv=3,            # 3-fold CV is faster than 5-fold
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring='f1'     # Optimize for F1 to balance precision/recall
)

rf_random.fit(X_train, y_train)

best_model = rf_random.best_estimator_
print(f"\nBest Parameters found: {rf_random.best_params_}")

# ---------------------------------------------------------
# 3. EVALUATION
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

print("\n=== Random Forest Validation Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Save Metrics
with open('docs/rf_metrics.txt', 'w') as f:
    f.write("Random Forest Validation Metrics\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

# ---------------------------------------------------------
# 4. FEATURE IMPORTANCE ANALYSIS
# ---------------------------------------------------------
print("\nGenerating Feature Importance Plot...")
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20

plt.figure(figsize=(10, 8))
plt.title("Top 20 Feature Importances (Random Forest)")
plt.barh(range(top_n), importances[indices][:top_n], align="center")
plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
plt.gca().invert_yaxis() # Highest importance at top
plt.tight_layout()
plt.savefig('docs/rf_feature_importance.png', dpi=300)
print("✓ Plot saved to docs/rf_feature_importance.png")

# ---------------------------------------------------------
# 5. SAVE MODEL
# ---------------------------------------------------------
joblib.dump(best_model, 'models/rf_best.joblib')
print("\n✓ Best Random Forest model saved to models/rf_best.joblib")
