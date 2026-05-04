import pandas as pd


def load_data_csv(path: str) -> pd.DataFrame:
    """
    Load csv dataset from disk.

    Args:
        path (str): Path to cleaned dataset CSV file.

    Returns:
        pd.DataFrame: Loaded dataset with stripped column names.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df