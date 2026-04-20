import pandas as pd
from loguru import logger

from Data_Science_Module.config import (
    INTERIM_DATA_DIR,
    TEST_IDENTITY,
    TEST_TRANSACTION,
    TRAIN_IDENTITY,
    TRAIN_MERGED,
    TRAIN_TRANSACTION,
)
from Data_Science_Module.validate import validate


def load_and_merge(transaction_path, identity_path) -> pd.DataFrame:
    logger.info(f"Loading {transaction_path.name} ...")
    transactions = pd.read_csv(transaction_path)

    logger.info(f"Loading {identity_path.name} ...")
    identity = pd.read_csv(identity_path)

    logger.info("Merging on TransactionID ...")
    merged = transactions.merge(identity, on="TransactionID", how="left")
    logger.info(f"Merged shape: {merged.shape}")
    return merged


def make_dataset():
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train = load_and_merge(TRAIN_TRANSACTION, TRAIN_IDENTITY)
    test = load_and_merge(TEST_TRANSACTION, TEST_IDENTITY)

    logger.info("Validating training data ...")
    report = validate(train, save_report=True)
    logger.info(f"Quality score: {report['overall_score']}")

    train.to_csv(TRAIN_MERGED, index=False)
    test.to_csv(INTERIM_DATA_DIR / "test_merged.csv", index=False)
    logger.info(f"Saved merged files to {INTERIM_DATA_DIR}")


if __name__ == "__main__":
    make_dataset()
