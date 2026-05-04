import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def bivariate_continuous_analysis(df_orig, df_smote=None, cols=None):
    """
    Perform bivariate analysis on continuous features.

    Includes:
        - Distribution comparison (Original vs SMOTE if provided)
        - Feature relationships (scatter + regression)
        - Boxplots vs target
    """
    print("Running Bivariate Continuous Analysis")

    if cols is None:
        cols = get_continuous_columns(df_orig)

    print(f"Detected continuous columns: {cols}")

    if df_smote is not None:
        compare_continuous_distributions(df_orig, df_smote, cols)

    plot_continuous_relationships(df_orig, cols)
    plot_boxplots_vs_target(df_orig, cols)


def compare_continuous_distributions(df_orig, df_smote, cols):
    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.kdeplot(df_orig[col], ax=axes[0])
        axes[0].set_title(f'Original: {col}')

        sns.kdeplot(df_smote[col], ax=axes[1])
        axes[1].set_title(f'SMOTE: {col}')

        plt.tight_layout()
        plt.show()


def plot_continuous_relationships(df, cols):
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    pairs = pairs[:5]

    for col1, col2 in pairs:
        sns.scatterplot(x=col1, y=col2, hue='is_fraud', data=df)
        plt.title(f'{col1} vs {col2}')
        plt.show()

        sns.regplot(x=col1, y=col2, data=df, scatter_kws={'alpha': 0.3})
        plt.title(f'Regression: {col1} vs {col2}')
        plt.show()


def plot_boxplots_vs_target(df, cols):
    if 'is_fraud' not in df.columns:
        return

    for col in cols:
        sns.boxplot(x='is_fraud', y=col, data=df)
        plt.title(f'{col} vs Fraud')
        plt.show()


def get_continuous_columns(df, threshold=10):
    return [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > threshold
    ]