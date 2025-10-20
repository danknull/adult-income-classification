"""
Error analysis and feature importance for Logistic Regression baseline
Identifies misclassification patterns and top predictive features
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

RANDOM_STATE = 42

# Column names for reference
COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'
]

NUMERIC_FEATURES = ['age', 'hours-per-week', 'capital-gain', 'capital-loss', 'education-num']
CATEGORICAL_NOMINAL = ['workclass', 'marital-status', 'occupation', 'relationship', 
                       'race', 'sex', 'native-country']


def load_original_data():
    """Load original training data for error inspection"""
    df = pd.read_csv('data/raw/adult_train.csv', 
                     header=None, 
                     names=COLUMN_NAMES,
                     skipinitialspace=True,
                     na_values='?')
    return df


def analyze_misclassifications():
    """Analyze patterns in false positives and false negatives"""
    
    print("="*70)
    print(" ERROR ANALYSIS: LOGISTIC REGRESSION BASELINE")
    print("="*70)
    
    # Load data and model
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    
    model = joblib.load('models/logreg_baseline.joblib')
    
    # Get predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Identify error types
    false_positives = (y_pred == 1) & (y_val == 0)  # Predicted >50K, actually ≤50K
    false_negatives = (y_pred == 0) & (y_val == 1)  # Predicted ≤50K, actually >50K
    true_positives = (y_pred == 1) & (y_val == 1)
    true_negatives = (y_pred == 0) & (y_val == 0)
    
    print("\n1. CONFUSION MATRIX BREAKDOWN")
    print("-" * 70)
    cm = confusion_matrix(y_val, y_pred)
    print(f"True Negatives:  {cm[0,0]:>6} (correctly predicted ≤50K)")
    print(f"False Positives: {cm[0,1]:>6} (predicted >50K, actually ≤50K)")
    print(f"False Negatives: {cm[1,0]:>6} (predicted ≤50K, actually >50K)")
    print(f"True Positives:  {cm[1,1]:>6} (correctly predicted >50K)")
    
    fp_rate = cm[0,1] / (cm[0,0] + cm[0,1]) * 100
    fn_rate = cm[1,0] / (cm[1,0] + cm[1,1]) * 100
    print(f"\nFalse Positive Rate: {fp_rate:.2f}% (of actual ≤50K)")
    print(f"False Negative Rate: {fn_rate:.2f}% (of actual >50K)")
    
    # Visualize prediction confidence for errors
    print("\n2. PREDICTION CONFIDENCE ANALYSIS")
    print("-" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # False Positives
    fp_proba = y_proba[false_positives]
    axes[0].hist(fp_proba, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold')
    axes[0].set_title('False Positives: Prediction Confidence', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Probability (>50K)', fontsize=10)
    axes[0].set_ylabel('Count', fontsize=10)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # False Negatives
    fn_proba = y_proba[false_negatives]
    axes[1].hist(fn_proba, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold')
    axes[1].set_title('False Negatives: Prediction Confidence', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Probability (>50K)', fontsize=10)
    axes[1].set_ylabel('Count', fontsize=10)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/error_confidence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"False Positives mean confidence: {fp_proba.mean():.3f}")
    print(f"False Negatives mean confidence: {fn_proba.mean():.3f}")
    print("✓ Confidence plots saved to docs/error_confidence.png")
    
    return false_positives, false_negatives


def analyze_feature_importance():
    """Extract and visualize Logistic Regression coefficients"""
    
    print("\n3. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 70)
    
    # Load model and preprocessor
    model = joblib.load('models/logreg_baseline.joblib')
    preprocessor = joblib.load('models/preprocessor.joblib')
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Create dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Top 20 features
    top_20 = feature_importance.head(20)
    
    print("\nTop 20 Most Important Features:")
    print(top_20[['Feature', 'Coefficient']].to_string(index=False))
    
    # Visualize top 15
    top_15 = feature_importance.head(15)
    
    plt.figure(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in top_15['Coefficient']]
    plt.barh(range(len(top_15)), top_15['Coefficient'], color=colors, edgecolor='black')
    plt.yticks(range(len(top_15)), top_15['Feature'], fontsize=9)
    plt.xlabel('Coefficient Value', fontsize=11, fontweight='bold')
    plt.title('Top 15 Features by Absolute Coefficient\n(Green = Increases >50K probability, Red = Decreases)', 
              fontsize=12, fontweight='bold')
    plt.axvline(0, color='black', linewidth=1.5)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Feature importance plot saved to docs/feature_importance.png")
    
    return feature_importance


def generate_insights(feature_importance):
    """Generate concrete next steps based on analysis"""
    
    print("\n4. KEY INSIGHTS & NEXT STEPS")
    print("="*70)
    
    print("\n📊 ERROR PATTERNS:")
    print("   • False Positive Rate indicates model over-predicts high income")
    print("   • False Negative Rate shows missed high earners")
    print("   • Confidence distributions reveal model uncertainty near decision boundary")
    
    print("\n🔍 TOP PREDICTIVE FACTORS:")
    top_5 = feature_importance.head(5)
    for idx, row in top_5.iterrows():
        direction = "increases" if row['Coefficient'] > 0 else "decreases"
        print(f"   • {row['Feature']}: {direction} >50K likelihood (coef: {row['Coefficient']:.3f})")
    
    print("\n💡 CONCRETE NEXT STEPS FOR MILESTONE REPORT:")
    print("-" * 70)
    
    print("\n1. ADDRESS CLASS IMBALANCE:")
    print("   → Try class_weight='balanced' in Logistic Regression")
    print("   → Compare precision-recall tradeoff with adjusted thresholds")
    print("   → Evaluate impact on False Positive vs False Negative rates")
    
    print("\n2. FEATURE ENGINEERING:")
    print("   → Create interaction features (e.g., education × hours-per-week)")
    print("   → Bin capital-gain/loss (most values are 0, creating skewness)")
    print("   → Group rare categories in occupation and native-country")
    
    print("\n3. ENSEMBLE METHODS:")
    print("   → Random Forest: Can capture non-linear relationships better")
    print("   → Gradient Boosting: May improve on borderline cases (prob ≈ 0.5)")
    print("   → Compare feature importance across models for robustness")
    
    print("\n4. ERROR ANALYSIS DEEP DIVE:")
    print("   → Inspect specific examples of high-confidence errors")
    print("   → Check if false positives share demographic patterns")
    print("   → Validate if rare occupations/countries correlate with errors")
    
    print("\n" + "="*70)
    
    # Save insights to file
    with open('docs/error_analysis_summary.txt', 'w') as f:
        f.write("ERROR ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write("TOP 5 PREDICTIVE FEATURES:\n")
        for idx, row in top_5.iterrows():
            f.write(f"  {row['Feature']}: {row['Coefficient']:.4f}\n")
        f.write("\nRECOMMENDED NEXT STEPS:\n")
        f.write("1. Adjust class weights or decision threshold\n")
        f.write("2. Engineer interaction and binned features\n")
        f.write("3. Try Random Forest and Gradient Boosting\n")
        f.write("4. Deep dive into high-confidence misclassifications\n")
    
    print("✓ Summary saved to docs/error_analysis_summary.txt")


def main():
    """Run complete error analysis pipeline"""
    
    # Analyze misclassifications
    fp_mask, fn_mask = analyze_misclassifications()
    
    # Feature importance
    feature_importance = analyze_feature_importance()
    
    # Generate actionable insights
    generate_insights(feature_importance)
    
    print("\n✅ ERROR ANALYSIS COMPLETE")
    print("   Generated files:")
    print("   - docs/error_confidence.png")
    print("   - docs/feature_importance.png")
    print("   - docs/error_analysis_summary.txt")


if __name__ == "__main__":
    main()
