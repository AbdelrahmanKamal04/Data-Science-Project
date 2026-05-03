import seaborn as sns
import matplotlib.pyplot as plt


def interaction_continuous(df, x, y):
    """
    Plot regression interaction between two continuous variables.

    Args:
        df (pd.DataFrame): Input dataset.
        x (str): X-axis feature.
        y (str): Y-axis feature.
    """
    sns.lmplot(x=x, y=y, hue='is_fraud', data=df)
    plt.title(f'{x} vs {y} interaction')
    plt.show()


def interaction_categorical(df, col1, col2):
    """
    Plot interaction between two categorical variables.

    Args:
        df (pd.DataFrame): Input dataset.
        col1 (str): First categorical feature.
        col2 (str): Second categorical feature.
    """
    sns.countplot(x=col1, hue=col2, data=df)
    plt.title(f'{col1} vs {col2}')
    plt.show()