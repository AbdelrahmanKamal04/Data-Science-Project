import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def bivariate_continuous_analysis(df_orig, df_smote):
    """
    Perform bivariate analysis on continuous features.

    Includes:
        - Distribution comparison (Original vs SMOTE)
        - Feature relationships (scatter + regression)
    """
    print("Running Bivariate Continuous Analysis...")

    cols = get_continuous_columns(df_orig)
    print(f"Detected continuous columns: {cols}")

    compare_continuous_distributions(df_orig, df_smote, cols)

    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    pairs = pairs[:5]

    plot_continuous_relationships(df_orig, pairs)


def compare_continuous_distributions(df_orig, df_smote, cols):
    """
    Compare feature distributions between original and SMOTE datasets.
    """
    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.kdeplot(df_orig[col], ax=axes[0])
        axes[0].set_title(f'Original: {col}')

        sns.kdeplot(df_smote[col], ax=axes[1])
        axes[1].set_title(f'SMOTE: {col}')

        plt.tight_layout()
        plt.show()


def plot_continuous_relationships(df, pairs):
    """
    Plot relationships between continuous feature pairs.
    """
    for col1, col2 in pairs:
        sns.scatterplot(x=col1, y=col2, hue='is_fraud', data=df)
        plt.title(f'{col1} vs {col2}')
        plt.show()

        sns.regplot(x=col1, y=col2, data=df, scatter_kws={'alpha': 0.3})
        plt.title(f'Regression: {col1} vs {col2}')
        plt.show()


def get_continuous_columns(df, threshold=10):
    """
    Detect continuous numerical columns automatically.
    """
    return [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > threshold
    ]