import seaborn as sns
import matplotlib.pyplot as plt


def temporal_analysis(df, feature, time_col='transaction_hour'):
    """
    Analyze temporal trends of a feature over time.

    Args:
        df (pd.DataFrame): Input dataset.
        feature (str): Feature to analyze.
        time_col (str): Time column (default: transaction_hour).
    """
    grouped = df.groupby(time_col)[feature].mean()

    sns.lineplot(x=grouped.index, y=grouped.values)
    plt.title(f'{feature} over {time_col}')
    plt.show()