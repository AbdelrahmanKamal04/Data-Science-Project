import joblib


def save_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Save processed datasets to disk.

    Args:
        All datasets (pd.DataFrame): Train, validation, and test splits.
    """
    X_train.to_csv("../data/interim/scaled/X_train.csv", index=False)
    X_val.to_csv("../data/interim/scaled/X_val.csv", index=False)
    X_test.to_csv("../data/interim/scaled/X_test.csv", index=False)

    y_train.to_csv("../data/interim/label/Y_train.csv", index=False)
    y_val.to_csv("../data/interim/label/Y_val.csv", index=False)
    y_test.to_csv("../data/interim/label/Y_test.csv", index=False)


def save_scaler(scaler, path="scaler.pkl"):
    """
    Save trained scaler to disk.

    Args:
        scaler: Fitted scaler object.
        path (str): File path.
    """
    joblib.dump(scaler, path)