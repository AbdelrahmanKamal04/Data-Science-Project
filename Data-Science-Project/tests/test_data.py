import pandas as pd
import pytest

from Data_Science_Module.validate import (
    check_accuracy,
    check_completeness,
    check_outliers,
    check_uniqueness,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionDT": [100, 200, 300, 400, 500],
        "TransactionAmt": [10.5, 50.0, 200.0, 15.0, 80.0],
        "isFraud": [0, 0, 1, 0, 0],
    })


def test_accuracy_passes_on_valid_data(sample_df):
    result = check_accuracy(sample_df)
    assert result["score"] == 1.0
    assert result["issues"] == []


def test_accuracy_fails_on_negative_amount(sample_df):
    sample_df.loc[0, "TransactionAmt"] = -5
    result = check_accuracy(sample_df)
    assert result["issues"]


def test_completeness_score_is_one_when_no_nulls(sample_df):
    result = check_completeness(sample_df)
    assert result["score"] == 1.0
    assert result["required_cols_with_nulls"] == []


def test_completeness_detects_nulls(sample_df):
    sample_df.loc[0, "TransactionAmt"] = None
    result = check_completeness(sample_df)
    assert result["score"] < 1.0


def test_uniqueness_detects_duplicate_ids(sample_df):
    dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    result = check_uniqueness(dup)
    assert result["exact_duplicates"] > 0


def test_uniqueness_passes_on_clean_data(sample_df):
    result = check_uniqueness(sample_df)
    assert result["exact_duplicates"] == 0
    assert result["transactionid_duplicates"] == 0


def test_outliers_returns_score(sample_df):
    result = check_outliers(sample_df)
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
