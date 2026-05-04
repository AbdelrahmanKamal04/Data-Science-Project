import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from utils.plotting import show_plot


def show_custom(df):
    st.header("Custom Plot Builder")

    x = st.selectbox("X", df.columns)
    y = st.selectbox("Y", [None] + list(df.columns))
    plot_type = st.selectbox("Plot", ["Histogram", "Scatter", "Box", "Count"])

    fig, ax = plt.subplots()

    if plot_type == "Histogram":
        sns.histplot(df[x], kde=True, ax=ax)
    elif plot_type == "Scatter" and y:
        sns.scatterplot(x=x, y=y, hue='is_fraud', data=df, ax=ax)
    elif plot_type == "Box" and y:
        sns.boxplot(x=x, y=y, data=df, ax=ax)
    elif plot_type == "Count":
        sns.countplot(x=x, data=df, ax=ax)

    show_plot(fig)