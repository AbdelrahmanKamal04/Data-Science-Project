import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from utils.plotting import show_plot


def show_temporal(df):
    st.header("Temporal Analysis")

    if 'transaction_hour' not in df.columns:
        st.warning("No time feature available")
        return

    feature = st.selectbox("Feature", df.columns)

    grouped = df.groupby('transaction_hour')[feature].mean()

    fig, ax = plt.subplots()
    sns.lineplot(x=grouped.index, y=grouped.values, ax=ax)
    show_plot(fig)

    fraud = df.groupby('transaction_hour')['is_fraud'].sum()

    fig, ax = plt.subplots()
    sns.lineplot(x=fraud.index, y=fraud.values, ax=ax)
    show_plot(fig)