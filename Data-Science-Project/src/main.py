from data.clean import clean_data, save_clean_data
from data.inspect_data import basic_inspection
from data.load_data import load_data_csv
from data.validate import validate_data

from features.encoding import encode_features
from features.engineering import engineer_features
from features.save import save_data
from features.scaling import scale_features
from features.split import split_data
from features.validation import validate_splits
from features.utils import get_numerical_columns

# PRE-FE EDA
from visualizations.data_loader import load_datasets
from visualizations.univariate import univariate_analysis
from visualizations.bivariate_continuous import bivariate_continuous_analysis
from visualizations.bivariate_categorical import bivariate_categorical_analysis
from visualizations.multivariate import multivariate_analysis
from visualizations.interaction import interaction_analysis

# POST-FE EDA
from visualizations.bivariate_ohe import ohe_category_analysis
from visualizations.feature_interactions_advanced import conditional_fraud_analysis
from visualizations.risk_analysis import risk_hotspot_analysis

import pandas as pd

RANDOM_STATE = 42


def run_pre_fe_visualizations():
    print("\nRunning PRE-FE Visualization Pipeline")

    df_orig, df_smote = load_datasets()

    univariate_analysis(df_orig)

    bivariate_continuous_analysis(df_orig, df_smote)
    bivariate_categorical_analysis(df_orig)

    multivariate_analysis(df_orig)
    interaction_analysis(df_orig)

    print("PRE Feature Engineer visualization completed.")


def run_post_fe_visualizations(X_train, y_train):
    print("\nRunning POST Feature Engineer Visualization Pipeline")

    df = X_train.copy()
    df['is_fraud'] = y_train.values

    engineered_cols = [
        col for col in df.columns
        if any(k in col for k in ['amount', 'velocity', 'log', 'ratio'])
    ]

    bivariate_continuous_analysis(df, cols=engineered_cols)

    bivariate_categorical_analysis(df)

    ohe_cols = [col for col in df.columns if col.startswith("merchant_category_")]

    if ohe_cols:
        ohe_category_analysis(df, ohe_cols)

    conditional_fraud_analysis(df)

    risk_hotspot_analysis(df)

    multivariate_analysis(df)

    interaction_analysis(df, pairs=[
        ('log_amount', 'velocity_last_24h'),
        ('amount', 'velocity_last_24h')
    ])

    print("POST Feature Engineer visualization completed.")


def main(run_eda: bool = False):
    df = load_data_csv("../data/raw/credit_card_fraud_10k.csv")

    basic_inspection(df)
    validate_data(df)

    df = clean_data(df)

    save_clean_data(
        df,
        "../data/interim/cleaned/credit_card_fraud_10k_cleaned.csv"
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_col="is_fraud",
        random_state=RANDOM_STATE
    )

    validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)

    X_train, X_val, X_test = engineer_features(X_train, X_val, X_test)

    X_train.to_csv("../data/interim/unscaled/X_train.csv", index=False)
    y_train.to_csv("../data/interim/label/Y_train.csv", index=False)

    X_train, X_val, X_test = encode_features(
        X_train,
        X_val,
        X_test,
        column="merchant_category"
    )

    num_cols = get_numerical_columns(
        X_train,
        exclude_cols=['is_fraud', 'transaction_id']
    )

    print(f"Detected numerical columns: {num_cols}")

    X_train, X_val, X_test, scaler = scale_features(
        X_train,
        X_val,
        X_test,
        num_cols
    )

    save_data(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    )

    print("\nPreprocessing pipeline executed successfully.")

    if run_eda:
        run_pre_fe_visualizations()
        run_post_fe_visualizations(X_train, y_train)


if __name__ == "__main__":
    main(run_eda=True)