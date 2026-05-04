import streamlit as st


def filter_data(df):
    st.sidebar.subheader("🔍 Filters")

    df_filtered = df.copy()

    if 'is_fraud' in df.columns:
        option = st.sidebar.selectbox(
            "Fraud Filter",
            ["All", "Fraud Only", "Legit Only"]
        )

        if option == "Fraud Only":
            df_filtered = df[df['is_fraud'] == 1]
        elif option == "Legit Only":
            df_filtered = df[df['is_fraud'] == 0]

    if 'transaction_hour' in df.columns:
        hour = st.sidebar.slider("Hour", 0, 23, (0, 23))
        df_filtered = df_filtered[
            (df_filtered['transaction_hour'] >= hour[0]) &
            (df_filtered['transaction_hour'] <= hour[1])
        ]

    return df_filtered