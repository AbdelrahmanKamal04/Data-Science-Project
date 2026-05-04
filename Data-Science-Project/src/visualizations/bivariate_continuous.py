import seaborn as sns
import matplotlib.pyplot as plt


def compare_continuous_distributions(df_orig, df_smote, cols):
    """
    Compare distributions of continuous features between original and SMOTE datasets.

    Args:
        df_orig (pd.DataFrame): Original dataset.
        df_smote (pd.DataFrame): SMOTE dataset.
        cols (list[str]): List of continuous feature names.
    """
    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(df_orig[col], ax=axes[0])
        axes[0].set_title(f'Original: {col}')

        sns.histplot(df_smote[col], ax=axes[1])
        axes[1].set_title(f'SMOTE: {col}')

        plt.tight_layout()
        plt.show()


def plot_continuous_relationships(df_orig, df_smote, pairs):
    """
    Plot relationships between pairs of continuous features.

    Args:
        df_orig (pd.DataFrame): Original dataset.
        df_smote (pd.DataFrame): SMOTE dataset.
        pairs (list[tuple]): List of feature pairs (col1, col2).
    """
    for col1, col2 in pairs:
        sns.regplot(x=col1, y=col2, data=df_orig)
        plt.title(f'Original: {col1} vs {col2}')
        plt.show()