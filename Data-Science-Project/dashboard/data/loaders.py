import pandas as pd
import streamlit as st
from src.visualizations.data_loader import load_datasets


@st.cache_data
def get_data():
    return load_datasets()