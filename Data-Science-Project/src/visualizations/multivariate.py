import seaborn as sns
import matplotlib.pyplot as plt


def correlation_matrix(df, cols):
    """
    Plot correlation matrix for selected continuous features.

    Args:
        df (pd.DataFrame): Input dataset.
        cols (list[str]): Continuous feature names.
    """
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


def pair_plot(df, cols):
    """
    Generate pair plots colored by fraud class.

    Args:
        df (pd.DataFrame): Input dataset.
        cols (list[str]): Continuous feature names.
    """
    sns.pairplot(df[cols + ['is_fraud']], hue='is_fraud')
    plt.show()