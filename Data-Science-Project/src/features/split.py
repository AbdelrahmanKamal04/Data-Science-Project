from sklearn.model_selection import train_test_split


def split_data(df, target_col, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split dataset into train, validation, and test sets using stratification.

    Args:
        df (pd.DataFrame): Input dataset.
        target_col (str): Target column name.
        train_size (float): Training set ratio.
        val_size (float): Validation set ratio.
        test_size (float): Test set ratio.
        random_state (int): Random seed.

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - train_size),
        stratify=y,
        random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(test_size / (test_size + val_size)),
        stratify=y_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test