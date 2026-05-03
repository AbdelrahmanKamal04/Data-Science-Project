import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_data_quality(df):
    """
    Visualize key univariate distributions and data quality aspects.

    Args:
        df (pd.DataFrame): Input dataset.

    Includes:
        - Log-transformed transaction amount distribution
        - Fraud rate by merchant category
        - Device trust score distribution
        - Fraud rate by transaction hour
        - Boxplot comparison of fraud vs legitimate transactions
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))

    axes[0, 0].hist(np.log1p(df['amount']), bins=50)
    axes[0, 0].set_title('Transaction Amount (log scale)')

    fraud_by_category = df.groupby('merchant_category')['is_fraud'].mean()
    axes[0, 1].barh(fraud_by_category.index, fraud_by_category.values)

    axes[1, 0].hist(df['device_trust_score'], bins=30)

    hourly_fraud = df.groupby('transaction_hour')['is_fraud'].mean()
    axes[1, 1].plot(hourly_fraud.index, hourly_fraud.values)

    axes[2, 0].boxplot([
        df[df['is_fraud'] == 0]['amount'],
        df[df['is_fraud'] == 1]['amount']
    ])

    plt.tight_layout()
    plt.show()


def plot_class_distribution(df):
    """
    # Plot class distribution of fraud vs legitimate transactions.

    Args:
        df (pd.DataFrame): Input dataset containing 'is_fraud'.
    """
    sns.countplot(x='is_fraud', data=df)
    plt.title('Class Distribution')
    plt.show()