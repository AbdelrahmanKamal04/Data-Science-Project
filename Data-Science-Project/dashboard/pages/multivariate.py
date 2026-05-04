import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from src.visualizations.bivariate_continuous import get_continuous_columns
from utils.plotting import show_plot


def show_multivariate(df):
    st.header("Multivariate Analysis")

    cols = get_continuous_columns(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    show_plot(fig)

    sample = df.sample(min(500, len(df)))
    fig = sns.pairplot(sample[cols[:4] + ['is_fraud']], hue='is_fraud')
    st.pyplot(fig)