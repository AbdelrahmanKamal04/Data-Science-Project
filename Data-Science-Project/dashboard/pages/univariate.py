import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils.plotting import show_plot


def show_univariate(df):
    st.header("Univariate Analysis")

    col = st.selectbox("Feature", df.columns)
    fig, ax = plt.subplots()

    if pd.api.types.is_numeric_dtype(df[col]):
        sns.histplot(df[col], kde=True, ax=ax)
    else:
        sns.countplot(x=col, data=df, ax=ax)
        plt.xticks(rotation=45)

    show_plot(fig)

    if 'is_fraud' in df.columns and col != 'is_fraud':
        fig, ax = plt.subplots()

        if pd.api.types.is_numeric_dtype(df[col]):
            sns.boxplot(x='is_fraud', y=col, data=df, ax=ax)
        else:
            sns.countplot(x=col, hue='is_fraud', data=df, ax=ax)

        show_plot(fig)