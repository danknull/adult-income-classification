import pandas as pd
import os

EXPECTED_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'
]

def validate_dataset():
    train_path = 'data\\raw\\adult_train.csv'
    
    df_train = pd.read_csv(train_path, header=None, names=EXPECTED_COLUMNS, 
                           skipinitialspace=True)
    
    print("=== Data Validation Report ===")
    print(f"\nSource: UCI Machine Learning Repository")
    print(f"Download Date: 2025-10-19")
    print(f"\nTraining Set:")
    print(f"  Rows: {len(df_train)}")
    print(f"  Columns: {len(df_train.columns)}")
    print(f"  Expected: 32,561 train rows")
    
    missing_count = (df_train == '?').sum().sum()
    print(f"\nMissing values ('?'): {missing_count}")
    
    print(f"\nClass distribution:")
    print(df_train['income'].value_counts())
    
    return df_train

if __name__ == "__main__":
    validate_dataset()
