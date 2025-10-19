"""
Preprocessing pipeline for Adult Income Classification
Implements ColumnTransformer with numeric and categorical handling
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Define column names
COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'
]

# Feature groups
NUMERIC_FEATURES = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']

CATEGORICAL_NOMINAL = ['workclass', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']

# Note: 'education' is ordinal but we'll use one-hot for simplicity
# 'education-num' already encodes the order, so we include it in numeric
NUMERIC_FEATURES.append('education-num')

# Exclude 'fnlwgt' (census weight, not useful for prediction)
# Exclude 'education' (redundant with education-num)

def load_data(train_path='data/raw/adult_train.csv', 
              test_path='data/raw/adult_test.csv'):
    """Load training and test data"""
    
    # Load train
    df_train = pd.read_csv(train_path, 
                          header=None, 
                          names=COLUMN_NAMES,
                          skipinitialspace=True,
                          na_values='?')
    
    # Load test (UCI format has a period after labels in test set)
    df_test = pd.read_csv(test_path, 
                         header=None, 
                         names=COLUMN_NAMES,
                         skipinitialspace=True,
                         na_values='?',
                         skiprows=1)  # Skip first line (metadata)
    
    # Clean income labels (remove periods from test set)
    df_test['income'] = df_test['income'].str.replace('.', '', regex=False)
    
    print(f"Training data: {df_train.shape}")
    print(f"Test data: {df_test.shape}")
    
    return df_train, df_test


def create_preprocessing_pipeline():
    """
    Create scikit-learn preprocessing pipeline
    - Numeric: impute median + standard scaling
    - Categorical: impute mode + one-hot encoding
    """
    
    # Numeric pipeline: impute -> scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: impute -> one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_NOMINAL)
        ],
        remainder='drop'  # Drop fnlwgt and education
    )
    
    return preprocessor


def prepare_data(df, target_col='income'):
    """Separate features and target"""
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode target as binary (0: <=50K, 1: >50K)
    y = (y == '>50K').astype(int)
    
    return X, y


def create_stratified_splits(X, y, test_size_val=0.15, test_size_test=0.15):
    """
    Create 70/15/15 stratified train/validation/test splits
    """
    
    # First split: separate test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size_test, stratify=y, random_state=RANDOM_STATE
    )
    
    # Second split: separate validation from train
    # We want 15% of original data for validation
    # temp is now 85% of data, so val should be 15/85 ≈ 0.176 of temp
    val_size_adjusted = test_size_val / (1 - test_size_test)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    print("\n=== Split Sizes ===")
    print(f"Training:   {len(X_train):>6} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation: {len(X_val):>6} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test:       {len(X_test):>6} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Total:      {len(X):>6}")
    
    print("\n=== Class Distribution ===")
    print(f"Train:      {y_train.mean()*100:.1f}% >50K")
    print(f"Validation: {y_val.mean()*100:.1f}% >50K")
    print(f"Test:       {y_test.mean()*100:.1f}% >50K")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Save processed splits as CSV for inspection"""
    
    os.makedirs('data/processed', exist_ok=True)
    
    # Save splits
    pd.DataFrame(X_train).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_val).to_csv('data/processed/X_val.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/processed/X_test.csv', index=False)
    
    pd.DataFrame(y_train, columns=['income']).to_csv('data/processed/y_train.csv', index=False)
    pd.DataFrame(y_val, columns=['income']).to_csv('data/processed/y_val.csv', index=False)
    pd.DataFrame(y_test, columns=['income']).to_csv('data/processed/y_test.csv', index=False)
    
    print("\n✓ Splits saved to data/processed/")


def main():
    """Main preprocessing workflow"""
    
    print("="*60)
    print(" ADULT INCOME PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df_train, df_test = load_data()
    
    # Combine for consistent preprocessing
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    print(f"\nCombined dataset: {df_full.shape}")
    
    # Prepare features and target
    print("\n2. Preparing features and target...")
    X, y = prepare_data(df_full)
    
    # Create splits
    print("\n3. Creating stratified splits (70/15/15)...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_stratified_splits(X, y)
    
    # Create and fit preprocessor
    print("\n4. Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline()
    
    print("\n5. Fitting preprocessor on training data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    print(f"\nTransformed training data shape: {X_train_transformed.shape}")
    print(f"  - Original features: {X_train.shape[1]}")
    print(f"  - After preprocessing: {X_train_transformed.shape[1]}")
    
    # Transform validation and test
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Save preprocessor
    print("\n6. Saving preprocessor...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("✓ Preprocessor saved to models/preprocessor.joblib")
    
    # Save splits (transformed versions as numpy arrays)
    print("\n7. Saving processed data...")
    np.save('data/processed/X_train.npy', X_train_transformed)
    np.save('data/processed/X_val.npy', X_val_transformed)
    np.save('data/processed/X_test.npy', X_test_transformed)
    np.save('data/processed/y_train.npy', y_train.values)
    np.save('data/processed/y_val.npy', y_val.values)
    np.save('data/processed/y_test.npy', y_test.values)
    
    print("✓ Transformed splits saved to data/processed/")
    
    print("\n" + "="*60)
    print(" PREPROCESSING COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run baseline model training: python scripts/train_baseline.py")
    print("  2. Check preprocessor: import joblib; preprocessor = joblib.load('models/preprocessor.joblib')")
    

if __name__ == "__main__":
    main()
