def validate_splits(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Validate dataset splits for consistency and correctness.

    Raises:
        AssertionError if any validation fails.
    """
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    assert len(X_train) + len(X_val) + len(X_test) == \
           len(y_train) + len(y_val) + len(y_test)


def check_scaled_data(df, num_cols):
    """
    Validate scaled features.

    Args:
        df (pd.DataFrame): Scaled dataset.
        num_cols (list): Numerical columns.
    """
    assert df.isnull().sum().sum() == 0, "Missing values detected"
    print(df[num_cols].describe().loc[['mean', 'std']])