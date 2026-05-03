from data.load_data import load_data_csv
from data.inspect_data import basic_inspection
from data.validate_data import (
    check_missing_values,
    check_data_types,
    check_duplicates,
    fraud_analysis
)
from data.clean_data import clean_data, save_clean_data

from features.split import split_data
from features.engineering import compute_training_statistics, apply_feature_engineering
from features.encoding import one_hot_encode
from features.scaling import scale_features
from features.validation import validate_splits
from features.save_data import save_datasets, save_scaler


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
    check_missing_values(df)
    check_data_types(df)
    check_duplicates(df)
    fraud_analysis(df)

    df = clean_data(df)
    save_clean_data(df, "../data/interim/cleaned/credit_card_fraud_10k_cleaned.csv")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_col="is_fraud", random_state=RANDOM_STATE)

    validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)

    stats = compute_training_statistics(X_train)

    X_train = apply_feature_engineering(X_train, stats)
    X_val = apply_feature_engineering(X_val, stats)
    X_test = apply_feature_engineering(X_test, stats)

    X_train, X_val, X_test = one_hot_encode(X_train, X_val, X_test, column="merchant_category")

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

    save_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
    save_scaler(scaler)

    print("\nFull pipeline executed successfully.")


if __name__ == "__main__":
    main()