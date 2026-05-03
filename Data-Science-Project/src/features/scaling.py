from sklearn.preprocessing import StandardScaler


def scale_features(X_train, X_val, X_test, num_cols):
    """
    Scale numerical features using StandardScaler.

    Args:
        X_train, X_val, X_test (pd.DataFrame): Dataset splits.
        num_cols (list): Numerical columns.

    Returns:
        tuple: Scaled datasets and fitted scaler.
    """
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val_scaled[num_cols] = scaler.transform(X_val[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler