import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from utils.plotting import show_plot


def show_model_performance(df, final_df, pred_df, fi_df):
    st.header("Model Performance")

    if df is None:
        st.warning("No model results")
        return

    st.dataframe(df)

    metric = st.selectbox("Metric", df.select_dtypes(include=np.number).columns)

    fig, ax = plt.subplots()
    sns.barplot(data=df, x='model_name', y=metric, ax=ax)
    plt.xticks(rotation=30)
    show_plot(fig)

    if pred_df is not None:
        cm = confusion_matrix(pred_df['y_true'], pred_df['y_pred'])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        show_plot(fig)

        fpr, tpr, _ = roc_curve(pred_df['y_true'], pred_df['y_prob'])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        show_plot(fig)

    if fi_df is not None:
        top = fi_df.sort_values(by='importance', ascending=False).head(10)
        fig, ax = plt.subplots()
        sns.barplot(data=top, x='importance', y='feature', ax=ax)
        show_plot(fig)