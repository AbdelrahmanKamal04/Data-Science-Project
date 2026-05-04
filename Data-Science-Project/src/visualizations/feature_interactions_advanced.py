def conditional_fraud_analysis(df):
    """
    Analyze advanced feature interactions.
    """
    print("Running Conditional Fraud Analysis...")

    if 'is_night_transaction' not in df.columns or 'velocity_last_24h' not in df.columns:
        print("Required columns missing.")
        return

    df['night_high_vel'] = (
        (df['is_night_transaction'] == 1) &
        (df['velocity_last_24h'] >= df['velocity_last_24h'].quantile(0.75))
    ).astype(int)

    result = df.groupby('night_high_vel')['is_fraud'].mean()
    print(result)