import pandas as pd


def load_post_fe_data(mode="all"):
    """
     Load post-feature engineering datasets for visualization.

     Args:
         mode (str): Which dataset to load. Options:
             - "train": Load only training set
             - "val": Load only validation set
             - "test": Load only test set
             - "all": Load and concatenate all sets (default)

     Returns:
         pd.DataFrame: Dataset with features + target, ready for visualization.

     Description:
         - Loads unscaled feature datasets and corresponding labels from disk.
         - Merges features and target into a single DataFrame.
         - Allows loading specific splits or the entire dataset for comprehensive analysis.
    """
    X_train = pd.read_csv('data/interim/unscaled/X_train.csv')
    X_val = pd.read_csv('data/interim/unscaled/X_val.csv')
    X_test = pd.read_csv('data/interim/unscaled/X_test.csv')

    y_train = pd.read_csv('data/interim/label/Y_train.csv')
    y_val = pd.read_csv('data/interim/label/Y_val.csv')
    y_test = pd.read_csv('data/interim/label/Y_test.csv')

    def attach_target(X, y):
        df = X.copy()
        df['is_fraud'] = y.values
        return df

    df_train = attach_target(X_train, y_train)
    df_val = attach_target(X_val, y_val)
    df_test = attach_target(X_test, y_test)

    if mode == "train":
        return df_train
    elif mode == "val":
        return df_val
    elif mode == "test":
        return df_test
    else:
        df_all = pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)
        return df_all