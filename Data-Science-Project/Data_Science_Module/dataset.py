"""Merge raw transaction and identity files, save to data/interim/."""

from pathlib import Path

import pandas as pd


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
INTERIM_DIR = Path(__file__).parent.parent / "data" / "interim"


def load_and_merge(transaction_path: Path, identity_path: Path) -> pd.DataFrame:
    print(f"  Loading {transaction_path.name} ...")
    transactions = pd.read_csv(transaction_path)
    print(f"  Loading {identity_path.name} ...")
    identity = pd.read_csv(identity_path)
    print("  Merging on TransactionID ...")
    merged = transactions.merge(identity, on="TransactionID", how="left")
    print(f"  Merged shape: {merged.shape}")
    return merged


def make_dataset() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    print("Processing training data...")
    train = load_and_merge(
        RAW_DIR / "train_transaction.csv",
        RAW_DIR / "train_identity.csv",
    )
    train.to_csv(INTERIM_DIR / "train_merged.csv", index=False)

    print("Processing test data...")
    test = load_and_merge(
        RAW_DIR / "test_transaction.csv",
        RAW_DIR / "test_identity.csv",
    )
    test.to_csv(INTERIM_DIR / "test_merged.csv", index=False)

    print(f"\nFiles saved to: {INTERIM_DIR}")


if __name__ == "__main__":
    make_dataset()
