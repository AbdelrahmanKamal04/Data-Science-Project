import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def save_data(X_train, X_val, X_test, y_train, y_val, y_test, scaler):
    """
    Save processed datasets and scaler to disk.
    """
    save_datasets(X_train, X_val, X_test, y_train, y_val, y_test)
    save_scaler(scaler)

def save_datasets(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Save processed datasets to disk.
    """
    scaled_dir = BASE_DIR / "data" / "interim" / "scaled"
    label_dir = BASE_DIR / "data" / "interim" / "label"

    scaled_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(scaled_dir / "X_train.csv", index=False)
    X_val.to_csv(scaled_dir / "X_val.csv", index=False)
    X_test.to_csv(scaled_dir / "X_test.csv", index=False)

    y_train.to_csv(label_dir / "Y_train.csv", index=False)
    y_val.to_csv(label_dir / "Y_val.csv", index=False)
    y_test.to_csv(label_dir / "Y_test.csv", index=False)


def save_scaler(scaler, path=None):
    """
    Save trained scaler to disk.
    """
    if path is None:
        path = BASE_DIR / "models" / "scaler.pkl"
    else:
        path = Path(path)
        
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, str(path))