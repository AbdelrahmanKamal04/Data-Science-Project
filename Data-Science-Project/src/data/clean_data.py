import pandas as pd


def handle_missing_values(df):
    """
    Fill missing values:
        - numerical → median
        - categorical → mode
    """
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def remove_duplicates(df):
    """
    Remove exact and partial duplicates.
    """
    df = df.copy()

    df = df.drop_duplicates()

    df = df.drop_duplicates(
        subset=['amount', 'cardholder_age', 'transaction_hour', 'merchant_category'],
        keep='first'
    )

    return df


def fix_data_types(df):
    """
    Convert columns to correct data types.
    """
    df = df.copy()

    df['transaction_id'] = df['transaction_id'].astype(int)
    df['transaction_hour'] = df['transaction_hour'].astype(int)
    df['device_trust_score'] = df['device_trust_score'].astype(float)
    df['velocity_last_24h'] = df['velocity_last_24h'].astype(float)
    df['cardholder_age'] = df['cardholder_age'].astype(int)
    df['merchant_category'] = df['merchant_category'].astype('category')
    df['amount'] = df['amount'].astype(float)
    df['foreign_transaction'] = df['foreign_transaction'].astype(int)
    df['location_mismatch'] = df['location_mismatch'].astype(int)
    df['is_fraud'] = df['is_fraud'].astype(int)

    return df


def clean_data(df):
    """
    Full data cleaning pipeline.

    Steps:
        - handle missing values
        - remove duplicates
        - fix data types

    Returns:
        pd.DataFrame: cleaned dataset
    """
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = fix_data_types(df)

    return df


def save_clean_data(df, path):
    """
    Save cleaned dataset to disk.

    Args:
        df (pd.DataFrame): Cleaned dataset
        path (str): Output file path
    """
    df.to_csv(path, index=False)