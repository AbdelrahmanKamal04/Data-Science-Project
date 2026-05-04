import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.visualizations.data_loader import load_datasets
    from src.visualizations.bivariate_continuous import get_continuous_columns
    from src.visualizations.bivariate_categorical import get_categorical_columns
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

st.set_page_config(page_title="Fraud EDA Dashboard", layout="wide")

@st.cache_data
def get_data():
    return load_datasets()

df_orig, df_smote = get_data()

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
        df_pred = pd.read_csv("outputs/model_selection/csv/lightgbm_predictions.csv")
        return df_pred
    except:
        return None

@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv("outputs/model_selection/csv/lightgbm_feature_importance.csv")
    except:
        return None

all_models_df, final_model_df = load_model_results()
pred_df = load_predictions()
fi_df = load_feature_importance()

# ---------------- HELPERS ----------------
def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)

def filter_data(df):
    st.sidebar.subheader("🔍 Filters")

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

df_orig = filter_data(df_orig)

menu = st.sidebar.radio(
    "Analysis Type",
    [
        "Overview",
        "Univariate",
        "Bivariate",
        "Multivariate",
        "Temporal",
        "Custom Explorer",
        "Model Performance"
    ]
)

if menu == "Overview":
    st.header("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Data")
        st.write(df_orig.shape)
        st.dataframe(df_orig.head())

    with col2:
        st.subheader("SMOTE Data")
        st.write(df_smote.shape)
        st.dataframe(df_smote.head())

    st.subheader("Statistics")
    st.dataframe(df_orig.describe())

elif menu == "Univariate":
    st.header("Univariate Analysis")

    col = st.selectbox("Feature", df_orig.columns)

    fig, ax = plt.subplots()

    if pd.api.types.is_numeric_dtype(df_orig[col]):
        sns.histplot(df_orig[col], kde=True, ax=ax)
    else:
        sns.countplot(x=col, data=df_orig, ax=ax)
        plt.xticks(rotation=45)

    ax.set_title(f"{col} Distribution")
    show_plot(fig)

    if 'is_fraud' in df_orig.columns and col != 'is_fraud':
        fig, ax = plt.subplots()

        if pd.api.types.is_numeric_dtype(df_orig[col]):
            sns.boxplot(x='is_fraud', y=col, data=df_orig, ax=ax)
        else:
            sns.countplot(x=col, hue='is_fraud', data=df_orig, ax=ax)

        show_plot(fig)

elif menu == "Bivariate":
    st.header("Bivariate Analysis")

    cont_cols = get_continuous_columns(df_orig)
    cat_cols = get_categorical_columns(df_orig)

    mode = st.radio("Type", ["Continuous", "Categorical"])

    if mode == "Continuous":
        x = st.selectbox("X", cont_cols)
        y = st.selectbox("Y", cont_cols, index=1)

        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, hue='is_fraud', data=df_orig, ax=ax)
        show_plot(fig)

        fig, ax = plt.subplots()
        sns.regplot(x=x, y=y, data=df_orig, ax=ax)
        show_plot(fig)

    else:
        col = st.selectbox("Categorical", cat_cols)

        fig, ax = plt.subplots()
        sns.countplot(x=col, hue='is_fraud', data=df_orig, ax=ax)
        plt.xticks(rotation=45)
        show_plot(fig)

        fraud_rate = df_orig.groupby(col)['is_fraud'].mean()

        fig, ax = plt.subplots()
        fraud_rate.plot(kind='bar', ax=ax)
        show_plot(fig)

elif menu == "Multivariate":
    st.header("Multivariate Analysis")

    cont_cols = get_continuous_columns(df_orig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_orig[cont_cols].corr(), cmap='coolwarm', ax=ax , annot=True , fmt = '.4f')
    show_plot(fig)

    sample_df = df_orig.sample(min(500, len(df_orig)))
    fig = sns.pairplot(sample_df[cont_cols[:4] + ['is_fraud']], hue='is_fraud')
    st.pyplot(fig)

elif menu == "Temporal":
    st.header("Temporal Analysis")

    if 'transaction_hour' in df_orig.columns:
        feature = st.selectbox("Feature", df_orig.columns)

        grouped = df_orig.groupby('transaction_hour')[feature].mean()

        fig, ax = plt.subplots()
        sns.lineplot(x=grouped.index, y=grouped.values, ax=ax)
        show_plot(fig)

        fraud = df_orig.groupby('transaction_hour')['is_fraud'].sum()

        fig, ax = plt.subplots()
        sns.lineplot(x=fraud.index, y=fraud.values, ax=ax)
        show_plot(fig)

elif menu == "Custom Explorer":
    st.header("Custom Plot Builder")

    x = st.selectbox("X", df_orig.columns)
    y = st.selectbox("Y", [None] + list(df_orig.columns))
    plot_type = st.selectbox("Plot", ["Histogram", "Scatter", "Box", "Count"])

    fig, ax = plt.subplots()

    if plot_type == "Histogram":
        sns.histplot(df_orig[x], kde=True, ax=ax)
    elif plot_type == "Scatter" and y:
        sns.scatterplot(x=x, y=y, hue='is_fraud', data=df_orig, ax=ax)
    elif plot_type == "Box" and y:
        sns.boxplot(x=x, y=y, data=df_orig, ax=ax)
    elif plot_type == "Count":
        sns.countplot(x=x, data=df_orig, ax=ax)

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

    df[NUMERIC_COLS] = df[NUMERIC_COLS].apply(
        pd.to_numeric, errors='coerce'
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("All Models Overview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Model Comparison")

    metric = st.selectbox("Select Metric", numeric_cols)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df,
        x='model_name',
        y=metric,
        palette='viridis',
        ax=ax
    )
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

        metrics = row.drop(
            ['model_name', 'run_name'],
            errors='ignore'
        )

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
        - **Precision:** correctness of fraud predictions  
        - **Recall:** fraud detection coverage  
        - **F1-score:** balance between precision & recall  
        - **ROC-AUC:** ranking ability of model  
        - **PR Curve:** best for imbalanced datasets  
        """)

st.sidebar.divider()
st.sidebar.caption("Fraud Detection Dashboard")