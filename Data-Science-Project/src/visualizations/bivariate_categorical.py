import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def compare_categorical(df_orig, df_smote, cols):
    """
    Compare categorical feature distributions between original and SMOTE datasets.

    Args:
        df_orig (pd.DataFrame): Original dataset.
        df_smote (pd.DataFrame): SMOTE dataset.
        cols (list[str]): List of categorical columns.
    """
    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.countplot(x=col, data=df_orig, ax=axes[0])
        axes[0].set_title(f'Original: {col}')
        axes[0].tick_params(axis='x', rotation=45)

        sns.countplot(x=col, data=df_smote, ax=axes[1])
        axes[1].set_title(f'SMOTE: {col}')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


def categorical_heatmap(df, col1, col2):
    """
    Plot heatmap for relationship between two categorical variables.

    Args:
        df (pd.DataFrame): Input dataset.
        col1 (str): First categorical variable.
        col2 (str): Second categorical variable.
    """
    table = pd.crosstab(df[col1], df[col2])
    sns.heatmap(table, annot=True, fmt='g')
    plt.title(f'{col1} vs {col2}')
    plt.show()