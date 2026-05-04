import pandas as pd


def get_numerical_columns(df, exclude_cols=None, threshold=10):
    """
    Automatically detect numerical (continuous) columns.

    Args:
        df (pd.DataFrame)
        exclude_cols (list): Columns to exclude (e.g., target, IDs)
        threshold (int): Min unique values to consider continuous

    Returns:
        list[str]
    """
    exclude_cols = exclude_cols or []
    numerical_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() > threshold:
                numerical_cols.append(col)

    return numerical_cols