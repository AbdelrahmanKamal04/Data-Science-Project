"""
Raw data validation across 8 quality dimensions:
Accuracy, Consistency, Completeness, Uniqueness,
Outliers, Timeliness, Distribution, Relationships.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kurtosis, skew


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_all() -> dict[str, pd.DataFrame]:
    return {
        "transactions": pd.read_csv(RAW_DIR / "train_transaction.csv"),
        "identity": pd.read_csv(RAW_DIR / "train_identity.csv"),
    }


# ---------------------------------------------------------------------------
# Dimension 1 – Accuracy
# ---------------------------------------------------------------------------

def check_accuracy(dfs: dict) -> dict:
    issues = []

    txn = dfs["transactions"]

    # TransactionAmt must be positive
    neg_amt = (txn["TransactionAmt"] <= 0).sum()
    if neg_amt:
        issues.append(f"transactions: {neg_amt} rows with TransactionAmt <= 0")

    # TransactionAmt upper bound (99th percentile threshold from domain knowledge)
    high_amt = (txn["TransactionAmt"] > 20_000).sum()
    if high_amt:
        issues.append(f"transactions: {high_amt} rows with TransactionAmt > 20,000")

    # isFraud must be binary
    bad_fraud = (~txn["isFraud"].isin([0, 1])).sum()
    if bad_fraud:
        issues.append(f"transactions: {bad_fraud} non-binary isFraud values")

    # C columns (count-type features) must be >= 0
    c_cols = [c for c in txn.columns if c.startswith("C") and c[1:].isdigit()]
    for col in c_cols:
        bad = (txn[col].dropna() < 0).sum()
        if bad:
            issues.append(f"transactions: {bad} negative values in '{col}'")

    # D columns (delta/days features) must be >= 0
    d_cols = [c for c in txn.columns if c.startswith("D") and c[1:].isdigit()]
    for col in d_cols:
        bad = (txn[col].dropna() < 0).sum()
        if bad:
            issues.append(f"transactions: {bad} negative values in '{col}'")

    total = len(txn)
    issue_count = sum(
        int(s.split(":")[1].strip().split(" ")[0]) for s in issues
    ) if issues else 0
    score = round((1 - issue_count / total) * 100, 2) if total else 100.0

    return {"score_pct": score, "issues": issues}


# ---------------------------------------------------------------------------
# Dimension 2 – Consistency
# ---------------------------------------------------------------------------

def check_consistency(dfs: dict) -> dict:
    issues = []

    txn = dfs["transactions"]
    identity = dfs["identity"]

    # isFraud must be binary
    bad_fraud = (~txn["isFraud"].isin([0, 1])).sum()
    if bad_fraud:
        issues.append(f"transactions: {bad_fraud} isFraud values outside {{0,1}}")

    # TransactionDT must be monotonically increasing
    if not txn["TransactionDT"].is_monotonic_increasing:
        issues.append("transactions: TransactionDT is not in chronological order")

    # All identity TransactionIDs must exist in transactions
    txn_ids = set(txn["TransactionID"])
    orphan_ids = (~identity["TransactionID"].isin(txn_ids)).sum()
    if orphan_ids:
        issues.append(f"identity: {orphan_ids} TransactionIDs not found in transactions")

    # Each TransactionID should appear at most once in identity
    id_dupes = identity.duplicated(subset=["TransactionID"]).sum()
    if id_dupes:
        issues.append(f"identity: {id_dupes} duplicate TransactionIDs")

    total = len(txn) + len(identity)
    issue_count = sum(
        int(s.split(":")[1].strip().split(" ")[0]) for s in issues
    ) if issues else 0
    score = round((1 - issue_count / total) * 100, 2) if total else 100.0

    return {"score_pct": score, "issues": issues}


# ---------------------------------------------------------------------------
# Dimension 3 – Completeness
# ---------------------------------------------------------------------------

def check_completeness(dfs: dict) -> dict:
    # V columns are intentionally sparse (Vesta masked features)
    intentional_nulls = {
        "transactions": [f"V{i}" for i in range(1, 340)],
    }

    result = {}
    for name, df in dfs.items():
        design_cols = intentional_nulls.get(name, [])
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
        report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)

        scorable_missing = missing.drop(labels=[c for c in design_cols if c in missing.index])
        critical = report[
            (report["missing_pct"] > 20) & (~report.index.isin(design_cols))
        ]
        result[name] = {
            "total_rows": len(df),
            "columns_with_missing": len(report),
            "intentional_null_columns_count": len([c for c in design_cols if c in missing.index]),
            "critical_columns_gt20pct": len(critical),
            "score_pct": round(
                (1 - scorable_missing.sum() / (len(df) * len(df.columns))) * 100, 2
            ),
            "detail": report.to_dict(),
        }
    return result


# ---------------------------------------------------------------------------
# Dimension 4 – Uniqueness
# ---------------------------------------------------------------------------

def check_uniqueness(dfs: dict) -> dict:
    result = {}

    # transactions: TransactionID must be unique
    txn = dfs["transactions"]
    exact_dupes = txn.duplicated().sum()
    id_dupes = txn.duplicated(subset=["TransactionID"]).sum()
    result["transactions"] = {
        "total_rows": len(txn),
        "exact_duplicates": int(exact_dupes),
        "transactionid_duplicates": int(id_dupes),
        "score_pct": round((1 - exact_dupes / len(txn)) * 100, 2),
    }

    # identity: TransactionID must be unique (one identity record per transaction)
    idn = dfs["identity"]
    idn_exact = idn.duplicated().sum()
    idn_id_dupes = idn.duplicated(subset=["TransactionID"]).sum()
    result["identity"] = {
        "total_rows": len(idn),
        "exact_duplicates": int(idn_exact),
        "transactionid_duplicates": int(idn_id_dupes),
        "score_pct": round((1 - idn_exact / len(idn)) * 100, 2),
    }

    return result


# ---------------------------------------------------------------------------
# Dimension 5 – Outlier Detection  (IQR + Z-score on key numeric columns)
# ---------------------------------------------------------------------------

def _iqr_outliers(series: pd.Series) -> int:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


def _zscore_outliers(series: pd.Series, threshold: float = 3.0) -> int:
    z = np.abs(stats.zscore(series.dropna()))
    return int((z > threshold).sum())


def check_outliers(dfs: dict) -> dict:
    result = {}

    txn = dfs["transactions"]
    txn_cols = (
        ["TransactionAmt"]
        + [c for c in txn.columns if c.startswith("C") and c[1:].isdigit()]
        + [c for c in txn.columns if c.startswith("D") and c[1:].isdigit()]
    )
    txn_detail = {}
    for col in txn_cols:
        if col in txn.columns:
            s = txn[col].dropna()
            iqr_n = _iqr_outliers(s)
            z_n = _zscore_outliers(s)
            txn_detail[col] = {
                "iqr_outliers": iqr_n,
                "zscore_outliers": z_n,
                "iqr_pct": round(iqr_n / len(txn) * 100, 2),
            }
    result["transactions"] = txn_detail

    idn = dfs["identity"]
    idn_num_cols = [c for c in idn.columns if c.startswith("id_") and
                    pd.api.types.is_numeric_dtype(idn[c])]
    idn_detail = {}
    for col in idn_num_cols:
        s = idn[col].dropna()
        if len(s) < 4:
            continue
        iqr_n = _iqr_outliers(s)
        z_n = _zscore_outliers(s)
        idn_detail[col] = {
            "iqr_outliers": iqr_n,
            "zscore_outliers": z_n,
            "iqr_pct": round(iqr_n / len(idn) * 100, 2),
        }
    result["identity"] = idn_detail

    return result


# ---------------------------------------------------------------------------
# Dimension 6 – Timeliness
# ---------------------------------------------------------------------------

def check_timeliness(dfs: dict) -> dict:
    issues = []
    result = {}

    txn = dfs["transactions"]
    dt = txn["TransactionDT"]
    earliest = int(dt.min())
    latest = int(dt.max())
    result["transactions"] = {
        "earliest_dt": earliest,
        "latest_dt": latest,
        "range_seconds": latest - earliest,
        "range_days": round((latest - earliest) / 86400, 1),
    }

    if not dt.is_monotonic_increasing:
        issues.append("transactions: TransactionDT is not in chronological order")

    gaps = dt.diff().dropna()
    max_gap = int(gaps.max())
    result["transactions"]["max_gap_seconds"] = max_gap
    result["transactions"]["avg_gap_seconds"] = round(float(gaps.mean()), 2)

    if max_gap > 86400:
        issues.append(f"transactions: max gap between events is {max_gap}s (>{86400}s / 1 day)")

    result["issues"] = issues
    return result


# ---------------------------------------------------------------------------
# Dimension 7 – Distribution Profile
# ---------------------------------------------------------------------------

def check_distribution(dfs: dict) -> dict:
    result = {}

    for name, df in dfs.items():
        num_df = df.select_dtypes(include="number")
        if num_df.empty:
            continue
        profile = {}
        for col in num_df.columns:
            s = num_df[col].dropna()
            if len(s) < 4:
                continue
            ks_stat, ks_p = stats.kstest(s, "norm", args=(s.mean(), s.std() + 1e-9))
            profile[col] = {
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "skewness": round(float(skew(s)), 4),
                "kurtosis": round(float(kurtosis(s)), 4),
                "unique_values": int(s.nunique()),
                "ks_stat": round(float(ks_stat), 4),
                "ks_p_value": round(float(ks_p), 6),
                "is_normal_dist": bool(ks_p > 0.05),
            }
        result[name] = profile

    return result


# ---------------------------------------------------------------------------
# Dimension 8 – Relationships / Correlation
# ---------------------------------------------------------------------------

def check_relationships(dfs: dict) -> dict:
    result = {}

    txn = dfs["transactions"]
    feature_cols = (
        ["TransactionAmt", "isFraud"]
        + [c for c in txn.columns if c.startswith("C") and c[1:].isdigit()]
        + [c for c in txn.columns if c.startswith("D") and c[1:].isdigit()]
    )
    txn_num = txn[[c for c in feature_cols if c in txn.columns]].dropna(
        subset=["TransactionAmt", "isFraud"]
    )

    pearson = txn_num.corr(method="pearson").round(4)
    spearman = txn_num.corr(method="spearman").round(4)

    # Flag highly correlated feature pairs (|r| > 0.9) — potential data leakage
    high_corr_pairs = []
    cols = pearson.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = abs(pearson.iloc[i, j])
            if r > 0.9 and cols[i] != "isFraud" and cols[j] != "isFraud":
                high_corr_pairs.append({
                    "col_a": cols[i], "col_b": cols[j],
                    "pearson_r": round(float(pearson.iloc[i, j]), 4),
                })

    result["transactions"] = {
        "pearson_with_isFraud": pearson["isFraud"].drop("isFraud").to_dict(),
        "spearman_with_isFraud": spearman["isFraud"].drop("isFraud").to_dict(),
        "high_correlation_pairs_gt09": high_corr_pairs,
    }

    idn = dfs["identity"]
    idn_num_cols = [c for c in idn.columns if pd.api.types.is_numeric_dtype(idn[c])
                    and c != "TransactionID"]
    if idn_num_cols:
        idn_sample = idn[idn_num_cols].dropna(thresh=len(idn_num_cols) // 2)
        if len(idn_sample) >= 2:
            idn_pearson = idn_sample.corr(method="pearson").round(4)
            # Flag highly correlated id pairs
            id_high_pairs = []
            id_cols = idn_pearson.columns.tolist()
            for i in range(len(id_cols)):
                for j in range(i + 1, len(id_cols)):
                    r = abs(idn_pearson.iloc[i, j])
                    if r > 0.9:
                        id_high_pairs.append({
                            "col_a": id_cols[i], "col_b": id_cols[j],
                            "pearson_r": round(float(idn_pearson.iloc[i, j]), 4),
                        })
            result["identity"] = {
                "high_correlation_pairs_gt09": id_high_pairs,
            }

    return result


# ---------------------------------------------------------------------------
# Overall Quality Score
# ---------------------------------------------------------------------------

def overall_score(completeness: dict, uniqueness: dict, accuracy: dict, consistency: dict) -> float:
    scores = []
    for v in completeness.values():
        scores.append(v["score_pct"])
    for v in uniqueness.values():
        scores.append(v["score_pct"])
    scores.append(accuracy["score_pct"])
    scores.append(consistency["score_pct"])
    return round(sum(scores) / len(scores), 2) if scores else 0.0


# ---------------------------------------------------------------------------
# Run all checks and save report
# ---------------------------------------------------------------------------

def run_validation() -> dict:
    print("Loading raw data...")
    dfs = load_all()
    for name, df in dfs.items():
        print(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    print("\nRunning validation checks...")

    report = {}

    print("  [1/8] Accuracy...")
    report["accuracy"] = check_accuracy(dfs)

    print("  [2/8] Consistency...")
    report["consistency"] = check_consistency(dfs)

    print("  [3/8] Completeness...")
    report["completeness"] = check_completeness(dfs)

    print("  [4/8] Uniqueness...")
    report["uniqueness"] = check_uniqueness(dfs)

    print("  [5/8] Outliers...")
    report["outliers"] = check_outliers(dfs)

    print("  [6/8] Timeliness...")
    report["timeliness"] = check_timeliness(dfs)

    print("  [7/8] Distribution...")
    report["distribution"] = check_distribution(dfs)

    print("  [8/8] Relationships...")
    report["relationships"] = check_relationships(dfs)

    report["overall_quality_score_pct"] = overall_score(
        report["completeness"], report["uniqueness"],
        report["accuracy"], report["consistency"],
    )

    out_path = REPORTS_DIR / "data_validation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {out_path}")
    return report


def print_summary(report: dict) -> None:
    print("\n" + "=" * 60)
    print("DATA VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nOverall Quality Score: {report['overall_quality_score_pct']}%")

    print("\n--- ACCURACY ---")
    acc = report["accuracy"]
    print(f"  Score: {acc['score_pct']}%")
    for issue in acc["issues"]:
        print(f"  [!] {issue}")
    if not acc["issues"]:
        print("  All business rules passed.")

    print("\n--- CONSISTENCY ---")
    con = report["consistency"]
    print(f"  Score: {con['score_pct']}%")
    for issue in con["issues"]:
        print(f"  [!] {issue}")
    if not con["issues"]:
        print("  All consistency checks passed.")

    print("\n--- COMPLETENESS ---")
    for name, v in report["completeness"].items():
        print(f"  {name}: score={v['score_pct']}%  "
              f"cols_with_missing={v['columns_with_missing']}  "
              f"critical(>20%)={v['critical_columns_gt20pct']}")

    print("\n--- UNIQUENESS ---")
    for name, v in report["uniqueness"].items():
        print(f"  {name}: score={v['score_pct']}%  exact_dupes={v.get('exact_duplicates', 0)}")

    print("\n--- OUTLIERS (IQR, % of rows) ---")
    for name, cols in report["outliers"].items():
        flagged = {c: d["iqr_pct"] for c, d in cols.items() if d["iqr_pct"] > 0}
        if flagged:
            print(f"  {name}:")
            for col, pct in sorted(flagged.items(), key=lambda x: -x[1])[:10]:
                print(f"    {col}: {pct}% outliers")

    print("\n--- TIMELINESS ---")
    ti = report["timeliness"]
    for k, v in ti.items():
        if k != "issues":
            print(f"  {k}: {v}")
    for issue in ti.get("issues", []):
        print(f"  [!] {issue}")

    print("\n--- DISTRIBUTION (skewed columns) ---")
    for name, cols in report["distribution"].items():
        skewed = {c: d["skewness"] for c, d in cols.items() if abs(d["skewness"]) > 1}
        if skewed:
            print(f"  {name}:")
            for col, sk in sorted(skewed.items(), key=lambda x: -abs(x[1]))[:10]:
                print(f"    {col}: skewness={sk}")

    print("\n--- RELATIONSHIPS (Pearson with isFraud) ---")
    txn_r = report["relationships"].get("transactions", {})
    if txn_r.get("pearson_with_isFraud"):
        print("  transactions — top correlations with isFraud:")
        sorted_corr = sorted(
            txn_r["pearson_with_isFraud"].items(), key=lambda x: -abs(x[1])
        )[:10]
        for col, r in sorted_corr:
            print(f"    {col}: {r}")
    if txn_r.get("high_correlation_pairs_gt09"):
        print("  [!] Highly correlated feature pairs (|r|>0.9):")
        for pair in txn_r["high_correlation_pairs_gt09"]:
            print(f"    {pair['col_a']} ~ {pair['col_b']}: r={pair['pearson_r']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    report = run_validation()
    print_summary(report)
