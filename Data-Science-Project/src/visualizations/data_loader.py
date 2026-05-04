import pandas as pd


def load_datasets():
    """
    Load datasets required for EDA.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - credit_data_df: Original cleaned dataset
            - smote_data_df: SMOTE-generated dataset (used ONLY for visualization)

    Description:
        - Loads preprocessed datasets from disk.
        - Ensures column names are clean (no trailing spaces).
        - Converts target column 'is_fraud' to integer type.

    Notes:
        - This function does NOT generate SMOTE data.
        - SMOTE dataset is used ONLY for distribution comparison.
    """
    credit_data_df = pd.read_csv('data/interim/cleaned/credit_card_fraud_10k_cleaned.csv')
    credit_data_df.columns = credit_data_df.columns.str.strip()

    smote_data_df = pd.read_csv('data/interim/smote/X_train_smote_eda.csv')
    smote_data_df.columns = smote_data_df.columns.str.strip()

    if 'is_fraud' in credit_data_df.columns:
        credit_data_df['is_fraud'] = credit_data_df['is_fraud'].astype(int)

    if 'is_fraud' in smote_data_df.columns:
        smote_data_df['is_fraud'] = smote_data_df['is_fraud'].astype(int)

    return credit_data_df, smote_data_df