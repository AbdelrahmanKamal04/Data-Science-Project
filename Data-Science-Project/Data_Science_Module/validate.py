"""Data validation across 8 quality dimensions for the fraud detection dataset."""

import json

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kstest, zscore
from sklearn.ensemble import IsolationForest

from Data_Science_Module.config import (
    CORRELATION_THRESHOLD,
    IQR_MULTIPLIER,
    MISSING_THRESHOLD_PCT,
    NUMERIC_BOUNDS,
    REQUIRED_COLS,
    VALIDATION_REPORT,
)


# ── Dimension 1: Accuracy ────────────────────────────────────────────────────

def check_accuracy(df: pd.DataFrame) -> dict:
    issues = []
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        violations = ((df[col] < lo) | (df[col] > hi)).sum()
        if violations:
            issues.append(f"{col}: {violations} values outside [{lo}, {hi}]")
    score = 1.0 if not issues else round(1 - len(issues) / len(NUMERIC_BOUNDS), 2)
    return {"score": score, "issues": issues}


# ── Dimension 2: Consistency ─────────────────────────────────────────────────

def check_consistency(df: pd.DataFrame) -> dict:
    issues = []
    if not df["TransactionDT"].is_monotonic_increasing:
        issues.append("TransactionDT is not monotonically increasing")
    if "isFraud" in df.columns:
        invalid = ~df["isFraud"].isin([0, 1])
        if invalid.any():
            issues.append(f"isFraud has {invalid.sum()} values outside {{0,1}}")
    score = 1.0 if not issues else 0.0
    return {"score": score, "issues": issues}


# ── Dimension 3: Completeness ────────────────────────────────────────────────

def check_completeness(df: pd.DataFrame) -> dict:
    missing = (df.isnull().sum() / len(df) * 100).round(2)
    critical = missing[missing > MISSING_THRESHOLD_PCT].to_dict()
    required_missing = [c for c in REQUIRED_COLS if c in df.columns and df[c].isnull().any()]
    overall_score = round(1 - df.isnull().values.mean(), 4)
    return {
        "score": overall_score,
        "critical_columns": critical,
        "required_cols_with_nulls": required_missing,
        "missing_summary": missing.to_dict(),
    }


# ── Dimension 4: Uniqueness ──────────────────────────────────────────────────

def check_uniqueness(df: pd.DataFrame) -> dict:
    exact_dupes = int(df.duplicated().sum())
    id_dupes = int(df.duplicated(subset=["TransactionID"], keep=False).sum())
    total = len(df)
    score = round((total - exact_dupes) / total, 4)
    return {
        "score": score,
        "exact_duplicates": exact_dupes,
        "transactionid_duplicates": id_dupes,
    }


# ── Dimension 5: Outliers ────────────────────────────────────────────────────

def check_outliers(df: pd.DataFrame, method: str = "iqr") -> dict:
    col = "TransactionAmt"
    if col not in df.columns:
        return {"score": None, "issues": [f"{col} not found"]}

    series = df[col].dropna()

    if method == "iqr":
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        mask = (series < q1 - IQR_MULTIPLIER * iqr) | (series > q3 + IQR_MULTIPLIER * iqr)
        n_outliers = int(mask.sum())
        bounds = (round(float(q1 - IQR_MULTIPLIER * iqr), 2), round(float(q3 + IQR_MULTIPLIER * iqr), 2))
    else:
        z = np.abs(zscore(series))
        n_outliers = int((z > 3).sum())
        bounds = None

    score = round(1 - n_outliers / len(series), 4)
    return {"score": score, "method": method, "n_outliers": n_outliers, "bounds": bounds}


# ── Dimension 6: Timeliness ──────────────────────────────────────────────────

def check_timeliness(df: pd.DataFrame) -> dict:
    col = "TransactionDT"
    issues = []
    if col not in df.columns:
        return {"score": None, "issues": [f"{col} not found"]}

    if not df[col].is_monotonic_increasing:
        issues.append("TransactionDT is not ordered — potential data pipeline issue")

    gaps = df[col].diff().dropna()
    max_gap = int(gaps.max())
    avg_gap = round(float(gaps.mean()), 2)

    score = 1.0 if not issues else 0.5
    return {"score": score, "issues": issues, "max_gap_seconds": max_gap, "avg_gap_seconds": avg_gap}


# ── Dimension 7: Distribution ─────────────────────────────────────────────────

def check_distribution(df: pd.DataFrame) -> dict:
    col = "TransactionAmt"
    series = df[col].dropna()

    ks_stat, p_value = kstest(series, "norm", args=(float(series.mean()), float(series.std())))
    skewness = round(float(series.skew()), 4)
    kurt = round(float(series.kurtosis()), 4)

    fraud_rate = None
    if "isFraud" in df.columns:
        fraud_rate = round(float(df["isFraud"].mean() * 100), 2)

    return {
        "TransactionAmt": {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "skewness": skewness,
            "kurtosis": kurt,
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value": round(p_value, 4),
        },
        "fraud_rate_pct": fraud_rate,
    }


# ── Dimension 8: Relationships ────────────────────────────────────────────────

def check_relationships(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return {"score": 1.0, "highly_correlated_pairs": []}

    sample = df[numeric_cols].sample(min(5000, len(df)), random_state=42)
    corr = sample.corr(method="spearman").abs()

    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            val = corr.iloc[i, j]
            if val >= CORRELATION_THRESHOLD:
                pairs.append({
                    "col_a": corr.columns[i],
                    "col_b": corr.columns[j],
                    "spearman_r": round(float(val), 4),
                })

    score = round(1 - len(pairs) / max(1, len(corr.columns)), 4)
    return {"score": score, "highly_correlated_pairs": pairs[:20]}


# ── Master Validator ──────────────────────────────────────────────────────────

def validate(df: pd.DataFrame, save_report: bool = True) -> dict:
    logger.info(f"Validating dataset with shape {df.shape}")

    report = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "accuracy": check_accuracy(df),
        "consistency": check_consistency(df),
        "completeness": check_completeness(df),
        "uniqueness": check_uniqueness(df),
        "outliers": check_outliers(df),
        "timeliness": check_timeliness(df),
        "distribution": check_distribution(df),
        "relationships": check_relationships(df),
    }

    scores = {k: v["score"] for k, v in report.items() if isinstance(v, dict) and "score" in v}
    valid_scores = [s for s in scores.values() if s is not None]
    report["overall_score"] = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

    logger.info(f"Overall data quality score: {report['overall_score']}")

    if save_report:
        VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(VALIDATION_REPORT, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Validation report saved to {VALIDATION_REPORT}")

    return report
