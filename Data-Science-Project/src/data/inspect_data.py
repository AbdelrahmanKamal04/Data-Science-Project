def basic_inspection(df):
    """
    Perform initial inspection of dataset.

    Args:
        df (pd.DataFrame): Input dataset.
    """
    print("Head:")
    print(df.head())

    print("\nShape:", df.shape)

    print("\nDescribe:")
    print(df.describe())