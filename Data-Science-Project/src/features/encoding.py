import pandas as pd


def one_hot_encode(train, val, test, column):
    """
    Apply one-hot encoding and align columns across splits.

    Args:
        train, val, test (pd.DataFrame): Dataset splits.
        column (str): Column to encode.

    Returns:
        tuple: Encoded (train, val, test)
    """
    train = pd.get_dummies(train, columns=[column], drop_first=True, dtype=int)
    val = pd.get_dummies(val, columns=[column], drop_first=True, dtype=int)
    test = pd.get_dummies(test, columns=[column], drop_first=True, dtype=int)

    all_cols = sorted(set(train.columns) | set(val.columns) | set(test.columns))

    train = train.reindex(columns=all_cols, fill_value=0)
    val = val.reindex(columns=all_cols, fill_value=0)
    test = test.reindex(columns=all_cols, fill_value=0)

    return train, val, test