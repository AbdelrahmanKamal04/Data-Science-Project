import matplotlib.pyplot as plt
import seaborn as sns


def risk_hotspot_analysis(df):
    """
    Analyze fraud hotspots using pivot tables.
    """
    print("Running Risk Hotspot Analysis...")

    required = {'foreign_transaction', 'location_mismatch', 'is_fraud'}

    if not required.issubset(df.columns):
        print("Required columns missing.")
        return

    pivot = df.pivot_table(
        values='is_fraud',
        index='foreign_transaction',
        columns='location_mismatch',
        aggfunc='mean'
    )

    sns.heatmap(pivot, annot=True, cmap='coolwarm')
    plt.title("Risk Hotspots")
    plt.show()