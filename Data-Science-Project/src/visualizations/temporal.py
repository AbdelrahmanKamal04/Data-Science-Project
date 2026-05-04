import matplotlib.pyplot as plt
import seaborn as sns


def temporal_analysis(df, feature, time_col='transaction_hour'):
    """
    Analyze temporal trends in dataset.

    Includes:
        - Feature trend over time
        - Fraud count over time
    """
    print("Running Temporal Analysis...")

    if time_col not in df.columns:
        return

    grouped_mean = df.groupby(time_col)[feature].mean()
    grouped_count = df.groupby(time_col)['is_fraud'].sum()

    sns.lineplot(x=grouped_mean.index, y=grouped_mean.values)
    plt.title(f'{feature} over {time_col}')
    plt.show()

    sns.lineplot(x=grouped_count.index, y=grouped_count.values)
    plt.title(f'Fraud Count over {time_col}')
    plt.show()