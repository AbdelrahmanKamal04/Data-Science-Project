import pandas as pd


def load_datasets():
    """
    Load original cleaned dataset and SMOTE dataset used for EDA.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - credit_data_df: Original cleaned dataset
            - smote_data_df: SMOTE-generated dataset (EDA only)

    Notes:
        - This function does NOT generate SMOTE.
        - It only loads precomputed datasets for visualization.
    """
    credit_data_df = pd.read_csv('../data/interim/cleaned/credit_card_fraud_10k_cleaned.csv')
    credit_data_df.columns = credit_data_df.columns.str.strip()

    smote_data_df = pd.read_csv('../data/interim/smote/X_train_smote_eda.csv')
    smote_data_df.columns = smote_data_df.columns.str.strip()

    credit_data_df['is_fraud'] = credit_data_df['is_fraud'].astype(int)
    smote_data_df['is_fraud'] = smote_data_df['is_fraud'].astype(int)

    return credit_data_df, smote_data_df