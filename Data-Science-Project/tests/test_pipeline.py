import pandas as pd
import numpy as np
import pytest

from src.features.engineering import engineer_features
from src.features.scaling import scale_features
from src.features.split import split_data
from src.models.utils import calculate_business_metrics


@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        "transaction_id": [1, 2, 3, 4, 5, 6],
        "amount": [10.0, 50.0, 200.0, 15.0, 80.0, 30.0],
        "transaction_hour": [2, 14, 23, 9, 18, 5],
        "merchant_category": ["Grocery", "Electronics", "Grocery", "Travel", "Food", "Grocery"],
        "foreign_transaction": [0, 1, 0, 0, 1, 0],
        "location_mismatch": [0, 1, 0, 1, 0, 0],
        "device_trust_score": [70, 40, 85, 30, 60, 75],
        "velocity_last_24h": [1, 5, 0, 2, 3, 1],
        "cardholder_age": [25, 45, 60, 35, 50, 30],
        "is_fraud": [0, 0, 1, 0, 0, 0],
    })
    return df


def test_engineer_features_adds_expected_columns(sample_data):
    result = engineer_features(sample_data, sample_data, sample_data)
    X_train_eng = result[0]
    assert "log_amount" in X_train_eng.columns
    assert "is_night_transaction" in X_train_eng.columns
    assert "velocity_per_hour" in X_train_eng.columns
    assert "high_risk_abroad" in X_train_eng.columns
    assert "is_high_velocity_low_trust" in X_train_eng.columns
    assert "relevant_amount" in X_train_eng.columns


def test_scale_features_returns_scaler_and_correct_shape(sample_data):
    X_train, _, _, _ = split_data(sample_data, target_col="is_fraud")
    num_cols = ["amount", "device_trust_score", "velocity_last_24h", "cardholder_age"]
    X_train_scaled, _, _, scaler = scale_features(X_train, X_train, X_train, num_cols)
    assert scaler is not None
    assert X_train_scaled.shape[1] == len(num_cols)


def test_split_data_returns_correct_partitions(sample_data):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_data, target_col="is_fraud")
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == sample_data.shape[0]
    assert y_train.shape[0] == X_train.shape[0]
    assert "is_fraud" not in X_train.columns


def test_calculate_business_metrics_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    result = calculate_business_metrics(y_true, y_pred)
    assert result["total_business_cost"] == 0
    assert result["fraud_detection_rate"] == 100.0
    assert result["false_alarm_rate"] == 0.0


def test_calculate_business_metrics_worst_predictions():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    result = calculate_business_metrics(y_true, y_pred)
    assert result["fraud_detection_rate"] == 0.0
    assert result["total_business_cost"] > 0


def test_calculate_business_metrics_returns_expected_keys():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    result = calculate_business_metrics(y_true, y_pred)
    expected_keys = [
        "true_positives", "false_positives", "true_negatives", "false_negatives",
        "fraud_detection_rate", "false_alarm_rate", "total_business_cost",
        "potential_savings", "roi_percentage"
    ]
    for key in expected_keys:
        assert key in result


def test_pipeline_integration_end_to_end(sample_data):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(sample_data, target_col="is_fraud")
    X_train_eng, X_val_eng, X_test_eng = engineer_features(X_train, X_val, X_test)

    num_cols = ["amount", "device_trust_score", "velocity_last_24h", "cardholder_age"]
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train_eng, X_val_eng, X_test_eng, num_cols
    )

    assert X_train_scaled.shape[0] > 0
    assert X_val_scaled.shape[0] > 0
    assert X_test_scaled.shape[0] > 0
    assert X_train_scaled.shape[1] == X_val_scaled.shape[1] == X_test_scaled.shape[1]