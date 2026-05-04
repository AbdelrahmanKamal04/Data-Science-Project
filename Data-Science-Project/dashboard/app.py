import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.visualizations.data_loader import load_datasets
    from src.visualizations.post_feature_engineer_data_loader import load_post_fe_data
    from src.visualizations.bivariate_continuous import get_continuous_columns
    from src.visualizations.bivariate_categorical import get_categorical_columns
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

st.set_page_config(page_title="Fraud EDA Dashboard", layout="wide")

@st.cache_data
def get_pre_data():
    return load_datasets()

@st.cache_data
def get_post_data(split):
    try:
        if split == "Train Only":
            return load_post_fe_data("train")
        elif split == "Validation Only":
            return load_post_fe_data("val")
        elif split == "Test Only":
            return load_post_fe_data("test")
        else:
            return load_post_fe_data("all")
    except:
        return None

@st.cache_data
def load_model_results():
    try:
        all_models = pd.read_csv("outputs/model_selection/csv/all_models_comparison_final.csv")
        final_model = pd.read_csv("outputs/model_selection/csv/final_model_recommendation.csv")
        return all_models, final_model
    except:
        return None, None

@st.cache_data
def load_predictions():
    try:
        return pd.read_csv("outputs/model_selection/csv/lightgbm_predictions.csv")
    except:
        return None

@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv("outputs/model_selection/csv/lightgbm_feature_importance.csv")
    except:
        return None

df_orig, df_smote = get_pre_data()

split_mode = st.sidebar.radio(
    "Data Split",
    ["All", "Train Only", "Validation Only", "Test Only"]
)

df_post = get_post_data(split_mode)

all_models_df, final_model_df = load_model_results()
pred_df = load_predictions()
fi_df = load_feature_importance()

def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)

def filter_data(df):
    st.sidebar.subheader("Filters")
    df_filtered = df.copy()

    if 'is_fraud' in df.columns:
        fraud_filter = st.sidebar.selectbox("Fraud Filter", ["All", "Fraud Only", "Legit Only"])
        if fraud_filter == "Fraud Only":
            df_filtered = df_filtered[df_filtered['is_fraud'] == 1]
        elif fraud_filter == "Legit Only":
            df_filtered = df_filtered[df_filtered['is_fraud'] == 0]

    if 'transaction_hour' in df.columns:
        hour = st.sidebar.slider("Transaction Hour", 0, 23, (0, 23))
        df_filtered = df_filtered[
            (df_filtered['transaction_hour'] >= hour[0]) &
            (df_filtered['transaction_hour'] <= hour[1])
        ]

    return df_filtered

st.title("Fraud Detection Dashboard")
st.markdown("EDA + Model Performance (LightGBM Focus)")

data_mode = st.sidebar.radio(
    "Dataset Mode",
    ["Pre-Feature Engineering", "Post-Feature Engineering"]
)

if data_mode == "Post-Feature Engineering" and df_post is not None:
    df_current = df_post
else:
    df_current = df_orig

df_current = filter_data(df_current)

menu = st.sidebar.radio(
    "Analysis Type",
    [
        "Overview",
        "Univariate",
        "Bivariate",
        "Multivariate",
        "Temporal",
        "Feature Engineering",
        "Custom Explorer",
        "Model Performance"
    ]
)

if menu == "Overview":
    st.header("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Data")
        st.write(df_current.shape)
        st.dataframe(df_current.head())

    with col2:
        st.subheader("SMOTE Data")
        st.write(df_smote.shape)
        st.dataframe(df_smote.head())

    st.subheader("Statistics")
    st.dataframe(df_current.describe())

elif menu == "Univariate":
    st.header("Univariate Analysis")

    col = st.selectbox("Feature", df_current.columns)

    fig, ax = plt.subplots()

    if pd.api.types.is_numeric_dtype(df_current[col]):
        sns.histplot(df_current[col], kde=True, ax=ax)
    else:
        sns.countplot(x=col, data=df_current, ax=ax)
        plt.xticks(rotation=45)

    ax.set_title(f"{col} Distribution")
    show_plot(fig)

    if 'is_fraud' in df_current.columns and col != 'is_fraud':
        fig, ax = plt.subplots()

        if pd.api.types.is_numeric_dtype(df_current[col]):
            sns.boxplot(x='is_fraud', y=col, data=df_current, ax=ax)
        else:
            sns.countplot(x=col, hue='is_fraud', data=df_current, ax=ax)

        show_plot(fig)

elif menu == "Bivariate":
    st.header("Bivariate Analysis")

    cont_cols = get_continuous_columns(df_current)
    cat_cols = get_categorical_columns(df_current)

    mode = st.radio("Type", ["Continuous", "Categorical"])

    if mode == "Continuous":
        x = st.selectbox("X", cont_cols)
        y = st.selectbox("Y", cont_cols, index=1 if len(cont_cols) > 1 else 0)

        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, hue='is_fraud', data=df_current, ax=ax)
        show_plot(fig)

        fig, ax = plt.subplots()
        sns.regplot(x=x, y=y, data=df_current, ax=ax)
        show_plot(fig)

    else:
        col = st.selectbox("Categorical", cat_cols)

        fig, ax = plt.subplots()
        sns.countplot(x=col, hue='is_fraud', data=df_current, ax=ax)
        plt.xticks(rotation=45)
        show_plot(fig)

        if 'is_fraud' in df_current.columns:
            fraud_rate = df_current.groupby(col)['is_fraud'].mean()

            fig, ax = plt.subplots()
            fraud_rate.plot(kind='bar', ax=ax)
            show_plot(fig)

elif menu == "Multivariate":
    st.header("Multivariate Analysis")

    cont_cols = get_continuous_columns(df_current)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_current[cont_cols].corr(), cmap='coolwarm', ax=ax, annot=True, fmt='.4f')
    show_plot(fig)

    sample_df = df_current.sample(min(500, len(df_current)))
    fig = sns.pairplot(sample_df[cont_cols[:4] + ['is_fraud']], hue='is_fraud')
    st.pyplot(fig)

elif menu == "Temporal":
    st.header("Temporal Analysis")

    if 'transaction_hour' in df_current.columns:
        feature = st.selectbox("Feature", df_current.columns)

        grouped = df_current.groupby('transaction_hour')[feature].mean()

        fig, ax = plt.subplots()
        sns.lineplot(x=grouped.index, y=grouped.values, ax=ax)
        show_plot(fig)

        if 'is_fraud' in df_current.columns:
            fraud = df_current.groupby('transaction_hour')['is_fraud'].sum()

            fig, ax = plt.subplots()
            sns.lineplot(x=fraud.index, y=fraud.values, ax=ax)
            show_plot(fig)

elif menu == "Feature Engineering":
    st.header("Feature Engineering Analysis")

    if df_post is None:
        st.warning("Post-feature-engineered data not found.")
        st.stop()

    df = df_post.copy()

    engineered_cols = [
        col for col in df.columns
        if any(k in col for k in ['log', 'velocity', 'ratio', 'amount'])
    ]

    if engineered_cols:
        col = st.selectbox("Select Engineered Feature", engineered_cols)

        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        show_plot(fig)

        if 'is_fraud' in df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x='is_fraud', y=col, data=df, ax=ax)
            show_plot(fig)

    ohe_cols = [c for c in df.columns if c.startswith("merchant_category_")]

    if ohe_cols:
        fraud_rates = []

        for col in ohe_cols:
            rate = df[df[col] == 1]['is_fraud'].mean()
            fraud_rates.append(rate)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=fraud_rates, y=ohe_cols, ax=ax)
        show_plot(fig)

    if 'log_amount' in df.columns and 'velocity_last_24h' in df.columns:
        fig, ax = plt.subplots()
        sns.scatterplot(
            x='log_amount',
            y='velocity_last_24h',
            hue='is_fraud',
            data=df,
            ax=ax
        )
        show_plot(fig)

    required = {'foreign_transaction', 'location_mismatch'}

    if required.issubset(df.columns):
        pivot = df.pivot_table(
            values='is_fraud',
            index='foreign_transaction',
            columns='location_mismatch',
            aggfunc='mean'
        )

        fig, ax = plt.subplots()
        sns.heatmap(pivot, annot=True, cmap='coolwarm', ax=ax)
        show_plot(fig)

elif menu == "Custom Explorer":
    st.header("Custom Plot Builder")

    x = st.selectbox("X", df_current.columns)
    y = st.selectbox("Y", [None] + list(df_current.columns))
    plot_type = st.selectbox("Plot", ["Histogram", "Scatter", "Box", "Count"])

    fig, ax = plt.subplots()

    if plot_type == "Histogram":
        sns.histplot(df_current[x], kde=True, ax=ax)
    elif plot_type == "Scatter" and y:
        sns.scatterplot(x=x, y=y, hue='is_fraud', data=df_current, ax=ax)
    elif plot_type == "Box" and y:
        sns.boxplot(x=x, y=y, data=df_current, ax=ax)
    elif plot_type == "Count":
        sns.countplot(x=x, data=df_current, ax=ax)

    show_plot(fig)

elif menu == "Model Performance":
    st.header("Model Performance")

    if all_models_df is None:
        st.warning("No model results found.")
        st.stop()

    df = all_models_df.copy()
    df = df.drop(columns=['model_object'], errors='ignore')

    NUMERIC_COLS = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'roc_auc', 'fraud_detection_rate', 'false_alarm_rate',
        'total_business_cost', 'potential_savings',
        'roi_percentage', 'composite_score'
    ]

    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(pd.to_numeric, errors='coerce')
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("All Models Overview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Model Comparison")

    metric = st.selectbox("Select Metric", numeric_cols)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x='model_name', y=metric, palette='viridis', ax=ax)
    ax.set_title(f"{metric} by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    plt.xticks(rotation=30)
    show_plot(fig)

    if final_model_df is not None:
        st.subheader("Recommended Model")
        st.dataframe(final_model_df, use_container_width=True)

        st.metric(
            "Best Model",
            final_model_df['recommended_model'].iloc[0],
            f"ROI: {final_model_df['roi_percentage'].iloc[0]:.2f}%"
        )

    st.subheader("Model Performance Heatmap")

    heatmap_df = df.set_index("model_name")[numeric_cols]

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title("Model Performance Heatmap", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    show_plot(fig)

    st.subheader("LightGBM Deep Dive")

    lgbm = df[df['model_name'].str.contains("lightgbm", case=False, na=False)]

    if not lgbm.empty:
        row = lgbm.iloc[0]

        metrics = row.drop(['model_name', 'run_name'], errors='ignore')
        metrics = pd.to_numeric(metrics, errors='coerce').dropna()

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(metrics))
        ax.bar(x, metrics.values)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.index, rotation=45, ha='right')
        ax.set_title("LightGBM Metrics Breakdown")
        ax.set_ylabel("Value")
        plt.tight_layout()
        show_plot(fig)

    if pred_df is not None and {'y_true', 'y_pred'}.issubset(pred_df.columns):
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(pred_df['y_true'], pred_df['y_pred'])

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        show_plot(fig)

    if pred_df is not None and {'y_true', 'y_prob'}.issubset(pred_df.columns):
        st.subheader("ROC Curve")

        fpr, tpr, _ = roc_curve(pred_df['y_true'], pred_df['y_prob'])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], '--')
        ax.legend()
        show_plot(fig)

        st.subheader("Precision-Recall Curve")

        precision, recall, _ = precision_recall_curve(
            pred_df['y_true'],
            pred_df['y_prob']
        )

        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR Curve")
        show_plot(fig)

    if fi_df is not None:
        st.subheader("Top Feature Importance")

        top_features = fi_df.sort_values(
            by='importance',
            ascending=False
        ).head(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=top_features,
            x='importance',
            y='feature',
            ax=ax
        )
        show_plot(fig)

    with st.expander("Metric Explanation"):
        st.markdown("""
        - Precision: correctness of fraud predictions  
        - Recall: fraud detection coverage  
        - F1-score: balance between precision & recall  
        - ROC-AUC: ranking ability of model  
        - PR Curve: best for imbalanced datasets  
        """)

st.sidebar.divider()
st.sidebar.caption("Fraud Detection Dashboard")