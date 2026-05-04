import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def bivariate_categorical_analysis(df_orig):
    print("Running Bivariate Categorical Analysis")

    cols = get_categorical_columns(df_orig)
    print(f"Detected categorical columns: {cols}")

    if not cols:
        print("No categorical columns found.")
        return

    plot_categorical_distributions(df_orig, cols)
    plot_binary_flags(df_orig, cols)

    if 'is_fraud' in df_orig.columns:
        categorical_heatmap(df_orig, cols[0], 'is_fraud')


def plot_categorical_distributions(df, cols):
    for col in cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()


def plot_binary_flags(df, cols):
    if 'is_fraud' not in df.columns:
        return

    for col in cols:
        if df[col].nunique() == 2:
            fraud_rate = df.groupby(col)['is_fraud'].mean()
            fraud_rate.plot(kind='bar')
            plt.title(f'Fraud Rate by {col}')
            plt.show()


def categorical_heatmap(df, col1, col2):
    table = pd.crosstab(df[col1], df[col2])

    sns.heatmap(table, annot=True, fmt='g', cmap='Blues')
    plt.title(f'{col1} vs {col2}')
    plt.show()


def get_categorical_columns(df, threshold=5):
    categorical_cols = []

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() <= threshold:
            categorical_cols.append(col)

    return categorical_cols