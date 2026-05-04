import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from src.visualizations.bivariate_continuous import get_continuous_columns
from src.visualizations.bivariate_categorical import get_categorical_columns
from utils.plotting import show_plot


def show_bivariate(df):
    st.header("Bivariate Analysis")

    cont_cols = get_continuous_columns(df)
    cat_cols = get_categorical_columns(df)

    mode = st.radio("Type", ["Continuous", "Categorical"])

    if mode == "Continuous":
        x = st.selectbox("X", cont_cols)
        y = st.selectbox("Y", cont_cols, index=1)

        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, hue='is_fraud', data=df, ax=ax)
        show_plot(fig)

    else:
        col = st.selectbox("Categorical", cat_cols)

        fig, ax = plt.subplots()
        sns.countplot(x=col, hue='is_fraud', data=df, ax=ax)
        plt.xticks(rotation=45)
        show_plot(fig)