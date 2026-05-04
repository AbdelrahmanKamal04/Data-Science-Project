import matplotlib.pyplot as plt


def ohe_category_analysis(df, ohe_cols):
    """
    Analyze one-hot encoded categorical columns.
    """
    print("Running OHE Category Analysis")

    fraud_rates = []

    for col in ohe_cols:
        rate = df[df[col] == 1]['is_fraud'].mean()
        fraud_rates.append(rate)

    plt.figure(figsize=(8, 5))
    plt.barh(ohe_cols, fraud_rates)
    plt.title("Fraud Rate by OHE Category")
    plt.xlabel("Fraud Rate")
    plt.show()