import numpy as np
import pandas as pd


def check_missing_values(df):
    """
    Check for missing values in dataset.
    """
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values found:")
        print(missing[missing > 0])
    else:
        print("No missing values found.")


def check_data_types(df):
    """
    Print data types of all columns.
    """
    print("\nData Types:")
    print(df.dtypes)


def check_duplicates(df):
    """
    Check for duplicate rows.
    """
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Duplicate rows found: {duplicates}")
    else:
        print("No duplicate rows found.")


def numerical_summary(df, numerical_cols):
    """
    Print statistics for numerical columns.
    """
    for col in numerical_cols:
        print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}")
        print(f"{col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, std={df[col].std():.2f}")


def binary_summary(df, binary_cols):
    """
    Print value counts for binary columns.
    """
    for col in binary_cols:
        print(f"{col} value counts:\n{df[col].value_counts()}")


def fraud_analysis(df):
    """
    Analyze fraud distribution.
    """
    fraud_rate = df['is_fraud'].mean() * 100
    print(f"Fraud rate: {fraud_rate:.2f}%")


def outlier_analysis(df, cols):
    """
    Detect outliers using IQR method.

    Args:
        df (pd.DataFrame): Dataset
        cols (list): Columns to analyze
    """
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")


def correlation_with_target(df):
    """
    Compute correlation of numerical features with target.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('transaction_id')
    corr = df[numeric_cols].corr()['is_fraud'].drop('is_fraud').sort_values(ascending=False)

    print("Correlation with target:")
    print(corr)