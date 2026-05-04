import matplotlib.pyplot as plt
import seaborn as sns


def interaction_analysis(df):
    """
    Perform interaction analysis between features.

    Includes:
        - Continuous vs continuous (with fraud separation)
        - Categorical vs fraud relationships
    """
    print("Running Interaction Analysis...")

    if {'amount', 'velocity_last_24h', 'is_fraud'}.issubset(df.columns):
        interaction_continuous(df, 'amount', 'velocity_last_24h')

    if {'merchant_category', 'is_fraud'}.issubset(df.columns):
        interaction_categorical(df, 'merchant_category', 'is_fraud')


def interaction_continuous(df, x, y):
    """
    Plot interaction between two continuous features.
    """
    sns.lmplot(x=x, y=y, hue='is_fraud', data=df)
    plt.title(f'{x} vs {y}')
    plt.show()


def interaction_categorical(df, col1, col2):
    """
    Plot categorical interaction and impact on amount.
    """
    sns.countplot(x=col1, hue=col2, data=df)
    plt.title(f'{col1} vs {col2}')
    plt.show()

    if 'amount' in df.columns:
        sns.boxplot(x=col1, y='amount', hue=col2, data=df)
        plt.title(f'{col1} vs amount vs {col2}')
        plt.show()