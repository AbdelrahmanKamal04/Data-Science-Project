import matplotlib.pyplot as plt
import seaborn as sns


def interaction_analysis(df, pairs=None):
    print("Running Interaction Analysis...")

    if pairs is None:
        pairs = [
            ('amount', 'velocity_last_24h'),
            ('log_amount', 'velocity_last_24h')
        ]

    for x, y in pairs:
        if x in df.columns and y in df.columns:
            interaction_continuous(df, x, y)


def interaction_continuous(df, x, y):
    sns.scatterplot(x=x, y=y, hue='is_fraud', data=df)
    plt.title(f'{x} vs {y}')
    plt.show()

    sns.regplot(x=x, y=y, data=df, scatter_kws={'alpha': 0.3})
    plt.title(f'Regression: {x} vs {y}')
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