import numpy as np

def engineer_features(X_train, X_val, X_test):
    """
    Perform feature engineering on training and test datasets.

    Args:
        X_train (pd.DataFrame): Training feature set.
        X_val (pd.DataFrame): Validation feature set.
        X_test (pd.DataFrame): Test feature set.
    Returns:
        tuple: Transformed X_train, X_val, and X_test.
    """
    stats = compute_training_statistics(X_train)

    X_train_transformed = apply_feature_engineering(X_train, stats)
    X_val_transformed = apply_feature_engineering(X_val, stats)
    X_test_transformed = apply_feature_engineering(X_test, stats)
    return X_train_transformed, X_val_transformed, X_test_transformed

def compute_training_statistics(X_train):
    """
    Compute statistics from training data for feature engineering.

    Args:
        X_train (pd.DataFrame): Training feature set.

    Returns:
        dict: Dictionary containing computed statistics.
    """
    stats = {}

    stats["avg_merchant_amount"] = X_train.groupby('merchant_category')['amount'].mean()
    stats["global_avg_amount"] = stats["avg_merchant_amount"].mean()

    stats["high_velocity_threshold"] = X_train['velocity_last_24h'].quantile(0.9)
    stats["low_velocity_threshold"] = X_train['velocity_last_24h'].quantile(0.1)

    return stats


def apply_feature_engineering(df, stats):
    """
    Apply feature engineering transformations.

    Args:
        df (pd.DataFrame): Input dataset.
        stats (dict): Precomputed training statistics.

    Returns:
        pd.DataFrame: Transformed dataset.
    """
    df = df.copy()

    df['log_amount'] = np.log1p(df['amount'])
    df['is_night_transaction'] = df['transaction_hour'].between(0, 5).astype(int)
    df['velocity_per_hour'] = df['velocity_last_24h'] / np.maximum(df['transaction_hour'], 1)

    df['high_risk_abroad'] = (
        (df['foreign_transaction'] == 1) &
        (df['location_mismatch'] == 1)
    ).astype(int)

    df['is_high_velocity_low_trust'] = (
        (df['velocity_last_24h'] > stats["high_velocity_threshold"]) &
        (df['device_trust_score'] < stats["low_velocity_threshold"])
    ).astype(int)

    df['relevant_amount'] = df['amount'] / (
        df['merchant_category']
        .map(stats["avg_merchant_amount"])
        .fillna(stats["global_avg_amount"]) + 1e-6
    )

    return df