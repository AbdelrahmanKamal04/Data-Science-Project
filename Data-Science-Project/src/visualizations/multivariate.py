import matplotlib.pyplot as plt
import seaborn as sns


def multivariate_analysis(df):
    """
    Perform multivariate analysis.

    Includes:
        - Correlation matrix
        - Pair plots (sampled for performance)
    """
    print("Running Multivariate Analysis...")

    cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    correlation_matrix(df, cols)

    sample_df = df.sample(min(1000, len(df)))
    pair_plot(sample_df, cols[:5])


def correlation_matrix(df, cols):
    """
    Plot correlation matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap='coolwarm' , annot=True , fmt = '.4f' , ax = ax)
    plt.title('Correlation Matrix')
    plt.show()

    print("Top correlations:")
    print(corr.unstack().sort_values(ascending=False).head(10))


def pair_plot(df, cols):
    """
    Generate pairplot for selected features.
    """
    if 'is_fraud' in df.columns:
        sns.pairplot(df[cols + ['is_fraud']], hue='is_fraud')
        plt.show()