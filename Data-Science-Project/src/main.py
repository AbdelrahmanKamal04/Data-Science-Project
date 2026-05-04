from data.clean import clean_data
from data.clean import save_clean_data
from data.inspect_data import basic_inspection
from data.load_data import load_data_csv
from data.validate import validate_data

from features.encoding import encode_features
from features.engineering import engineer_features
from features.save import save_data
from features.scaling import scale_features
from features.split import split_data
from features.validation import validate_splits


RANDOM_STATE = 42


def main():
    """
    # Execute the complete data preprocessing and feature engineering pipeline.
    
    Orchestrates sequential data transformation steps while maintaining reproducibility
    and preventing data leakage. All scaling and engineering parameters are derived 
    solely from the training set, then applied to validation and test sets.
    
    Steps:
        1. Load raw transaction data
        2. Run initial inspection & validation checks (missing values, types, duplicates)
        3. Clean & optimize data types
        4. Save interim cleaned dataset
        5. Perform stratified train/val/test split
        6. Compute feature engineering statistics from training data only
        7. Apply engineered features, one-hot encoding, and standard scaling
        8. Validate split integrity & save all processed artifacts
        
    Note:
        - Uses `RANDOM_STATE = 42` for deterministic splitting and sampling.
        - Processed datasets are saved to `../data/processed/` for downstream modeling.
        - The fitted `StandardScaler` is serialized for consistent inference transformation.
        
    Raises:
        FileNotFoundError: If raw data or output directories are missing.
        ValueError: If validation checks or split integrity assertions fail.
    """  
    df = load_data_csv("../data/raw/credit_card_fraud_10k.csv")

    basic_inspection(df)
    validate_data(df)

    df = clean_data(df)
    save_clean_data(df, "../data/interim/cleaned/credit_card_fraud_10k_cleaned.csv")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_col="is_fraud", random_state=RANDOM_STATE)

    validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)

    X_train, X_val, X_test = engineer_features(X_train, X_val, X_test)
    X_train, X_val, X_test = encode_features(X_train, X_val, X_test, column="merchant_category")

    num_cols = [
        'transaction_hour',
        'device_trust_score',
        'velocity_last_24h',
        'cardholder_age',
        'log_amount',
        'velocity_per_hour',
        'relevant_amount'
    ]

    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test, num_cols)

    save_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler)

    print("\nFull pipeline executed successfully.")


if __name__ == "__main__":
    main()