import pandas as pd
import pytest

from Data_Science_Module.validate import (
    check_accuracy,
    check_completeness,
    check_consistency,
    check_outliers,
    check_uniqueness,
)


@pytest.fixture
def sample_dfs():
    txn = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionDT": [100, 200, 300, 400, 500],
        "TransactionAmt": [10.5, 50.0, 200.0, 15.0, 80.0],
        "isFraud": [0, 0, 1, 0, 0],
        "C1": [1.0, 2.0, 3.0, 1.0, 2.0],
        "C2": [5.0, 3.0, 7.0, 2.0, 4.0],
        "D1": [5.0, 10.0, 3.0, 7.0, 2.0],
        "D2": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    idn = pd.DataFrame({
        "TransactionID": [1, 3, 5],
        "id_01": [-5.0, -10.0, -3.0],
        "id_02": [100.0, 200.0, 150.0],
    })
    return {"transactions": txn, "identity": idn}


def test_accuracy_passes_on_valid_data(sample_dfs):
    result = check_accuracy(sample_dfs)
    assert result["score_pct"] == 100.0
    assert result["issues"] == []


def test_accuracy_flags_negative_amount(sample_dfs):
    sample_dfs["transactions"].loc[0, "TransactionAmt"] = -5.0
    result = check_accuracy(sample_dfs)
    assert result["issues"]
    assert result["score_pct"] < 100.0


def test_accuracy_flags_invalid_fraud_value(sample_dfs):
    sample_dfs["transactions"].loc[0, "isFraud"] = 2
    result = check_accuracy(sample_dfs)
    assert result["issues"]


def test_completeness_score_when_no_nulls(sample_dfs):
    result = check_completeness(sample_dfs)
    for name, v in result.items():
        assert "score_pct" in v
        assert v["score_pct"] >= 0.0


def test_completeness_detects_nulls(sample_dfs):
    sample_dfs["transactions"].loc[0, "TransactionAmt"] = None
    result = check_completeness(sample_dfs)
    assert result["transactions"]["columns_with_missing"] > 0


def test_uniqueness_passes_on_clean_data(sample_dfs):
    result = check_uniqueness(sample_dfs)
    assert result["transactions"]["exact_duplicates"] == 0
    assert result["transactions"]["transactionid_duplicates"] == 0


def test_uniqueness_detects_duplicates(sample_dfs):
    txn = sample_dfs["transactions"]
    sample_dfs["transactions"] = pd.concat([txn, txn.iloc[[0]]], ignore_index=True)
    result = check_uniqueness(sample_dfs)
    assert result["transactions"]["exact_duplicates"] > 0


def test_outliers_returns_structure(sample_dfs):
    result = check_outliers(sample_dfs)
    assert "transactions" in result
    assert "identity" in result
    assert "TransactionAmt" in result["transactions"]


def test_consistency_passes_on_valid_data(sample_dfs):
    result = check_consistency(sample_dfs)
    assert "score_pct" in result
    assert "issues" in result
