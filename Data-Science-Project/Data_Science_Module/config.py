from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = ROOT / "data" / "raw"
INTERIM_DATA_DIR = ROOT / "data" / "interim"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Raw file paths
TRAIN_TRANSACTION = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY = RAW_DATA_DIR / "train_identity.csv"
TEST_TRANSACTION = RAW_DATA_DIR / "test_transaction.csv"
TEST_IDENTITY = RAW_DATA_DIR / "test_identity.csv"

# Interim merged files
TRAIN_MERGED = INTERIM_DATA_DIR / "train_merged.csv"
TEST_MERGED = INTERIM_DATA_DIR / "test_merged.csv"

# Validation report output
VALIDATION_REPORT = REPORTS_DIR / "validation_report.json"

# Column definitions
TARGET_COL = "isFraud"
JOIN_KEY = "TransactionID"
AMOUNT_COL = "TransactionAmt"
TIME_COL = "TransactionDT"

# Business rules: columns that must never be null
REQUIRED_COLS = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]

# Accuracy rules: (column, min, max)
NUMERIC_BOUNDS = {
    "TransactionAmt": (0, 20_000),
    "isFraud": (0, 1),
}

# Outlier threshold for IQR method
IQR_MULTIPLIER = 1.5

# Missing data threshold to flag as critical
MISSING_THRESHOLD_PCT = 50.0

# High correlation threshold
CORRELATION_THRESHOLD = 0.95
