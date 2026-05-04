import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def univariate_analysis(df):
    """
    Perform comprehensive univariate analysis.

    Includes:
        - Class distribution
        - Data quality checks
        - Feature distributions
        - Missing values visualization
    """
    print("Running Univariate Analysis...")

    plot_class_distribution(df)
    plot_data_quality(df)
    plot_missing_values(df)


def plot_class_distribution(df):
    """
    Plot fraud vs non-fraud class distribution.
    """
    if 'is_fraud' not in df.columns:
        return

    sns.countplot(x='is_fraud', data=df)
    plt.title('Class Distribution')
    plt.show()


def plot_missing_values(df):
    """
    Visualize missing values across columns.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values.")
        return

    missing.sort_values().plot(kind='barh')
    plt.title("Missing Values per Column")
    plt.show()


def plot_data_quality(df):
    """
    Generate key univariate plots for important features.

    Includes:
        - Log-transformed amount distribution
        - Fraud rate by category
        - Device trust score distribution
        - Fraud rate over time
        - Fraud vs non-fraud comparisons
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 24))

    if 'amount' in df.columns:
        axes[0, 0].hist(np.log1p(df['amount']), bins=50)
        axes[0, 0].set_title('Transaction Amount (log scale)')

    if 'merchant_category' in df.columns and 'is_fraud' in df.columns:
        fraud_by_category = df.groupby('merchant_category')['is_fraud'].mean()
        axes[0, 1].barh(fraud_by_category.index, fraud_by_category.values)
        axes[0, 1].set_title('Fraud Rate by Category')

    if 'device_trust_score' in df.columns:
        sns.kdeplot(df['device_trust_score'], ax=axes[1, 0] , clip=(25,99))
        axes[1, 0].set_title('Device Trust Score Distribution')

    if 'transaction_hour' in df.columns and 'is_fraud' in df.columns:
        hourly_fraud = df.groupby('transaction_hour')['is_fraud'].mean()
        axes[1, 1].plot(hourly_fraud.index, hourly_fraud.values)
        axes[1, 1].set_title('Fraud Rate by Hour')

    if 'amount' in df.columns and 'is_fraud' in df.columns:
        axes[2, 0].boxplot([
            df[df['is_fraud'] == 0]['amount'],
            df[df['is_fraud'] == 1]['amount']
        ])
        axes[2, 0].set_title('Amount vs Fraud')

    if 'device_trust_score' in df.columns and 'is_fraud' in df.columns:
        sns.boxplot(x='is_fraud', y='device_trust_score', data=df, ax=axes[2, 1])

    if 'amount' in df.columns and 'is_fraud' in df.columns:
        sns.histplot(df[df['is_fraud'] == 0]['amount'], label='Legit', ax=axes[3, 0])
        sns.histplot(df[df['is_fraud'] == 1]['amount'], label='Fraud', ax=axes[3, 0])
        axes[3, 0].legend()
        axes[3, 0].set_title('Amount Distribution by Class')

    if 'transaction_hour' in df.columns:
        sns.countplot(x='transaction_hour', data=df, ax=axes[3, 1])

    plt.tight_layout()
    plt.show()