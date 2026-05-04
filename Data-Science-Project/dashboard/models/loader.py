import pandas as pd
import streamlit as st


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