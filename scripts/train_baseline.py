import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42

# Load preprocessed data
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_val = np.load('data/processed/y_val.npy')
y_test = np.load('data/processed/y_test.npy')

# Load preprocessor
preprocessor = joblib.load('models/preprocessor.joblib')

# Build pipeline: preprocessor (already fitted) + Logistic Regression
# Since X_train is already transformed, we skip preprocessor in pipeline for baseline
clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)

# Train model
clf.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = clf.predict(X_val)
y_val_proba = clf.predict_proba(X_val)[:, 1]

print("=== Validation Metrics ===")
print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall:    {recall_score(y_val, y_val_pred):.4f}")
print(f"F1 Score:  {f1_score(y_val, y_val_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_val, y_val_proba):.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['≤50K', '>50K'], yticklabels=['≤50K', '>50K'])
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('docs/confusion_matrix_val.png', dpi=300)
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=['≤50K', '>50K']))

# Save model
joblib.dump(clf, 'models/logreg_baseline.joblib')
print("\n✓ Baseline Logistic Regression model saved to models/logreg_baseline.joblib")

# Evaluate on test set (for final report)
y_test_pred = clf.predict(X_test)
y_test_proba = clf.predict_proba(X_test)[:, 1]
print("\n=== Test Metrics ===")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_test_proba):.4f}")

# Save metrics for report
with open('docs/baseline_metrics.txt', 'w') as f:
    f.write("Validation Metrics\n")
    f.write(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}\n")
    f.write(f"Precision: {precision_score(y_val, y_val_pred):.4f}\n")
    f.write(f"Recall:    {recall_score(y_val, y_val_pred):.4f}\n")
    f.write(f"F1 Score:  {f1_score(y_val, y_val_pred):.4f}\n")
    f.write(f"ROC AUC:   {roc_auc_score(y_val, y_val_proba):.4f}\n")
    f.write("\nTest Metrics\n")
    f.write(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}\n")
    f.write(f"Precision: {precision_score(y_test, y_test_pred):.4f}\n")
    f.write(f"Recall:    {recall_score(y_test, y_test_pred):.4f}\n")
    f.write(f"F1 Score:  {f1_score(y_test, y_test_pred):.4f}\n")
    f.write(f"ROC AUC:   {roc_auc_score(y_test, y_test_proba):.4f}\n")
print("✓ Metrics saved to docs/baseline_metrics.txt")
