"""Utility functions for modeling: MLflow setup, business metrics, and data loading."""

import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from .config import COST_FN, COST_FP, TRANSACTION_VALUE

def setup_mlflow(tracking_uri: str, experiment_name: str):
    """Initialize MLflow tracking and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Warning: {e}")
    return mlflow.get_experiment_by_name(experiment_name)

def calculate_business_metrics(y_true, y_pred, y_proba=None, 
                               cost_fn=COST_FN, cost_fp=COST_FP, 
                               transaction_value=TRANSACTION_VALUE):
    """Calculate business-oriented metrics for fraud detection."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    potential_savings = (tp * transaction_value) - (fp * cost_fp)
    
    return {
        'true_positives': int(tp), 'false_positives': int(fp),
        'true_negatives': int(tn), 'false_negatives': int(fn),
        'fraud_detection_rate': round(tp / (tp + fn) * 100, 2) if (tp + fn) > 0 else 0.0,
        'false_alarm_rate': round(fp / (fp + tn) * 100, 2) if (fp + tn) > 0 else 0.0,
        'total_business_cost': int(total_cost),
        'potential_savings': int(potential_savings),
        'roi_percentage': round((potential_savings / total_cost) * 100, 2) if total_cost > 0 else 0.0
    }

def load_processed_data(data_dir: str):
    """Load scaled features and label splits from interim directories."""
    def load_split(prefix):
        X = pd.read_csv(os.path.join(data_dir, f'scaled/{prefix}.csv'))
        Y = pd.read_csv(os.path.join(data_dir, f'label/{prefix}.csv'))
        X.columns = X.columns.str.strip()
        Y.columns = Y.columns.str.strip()
        return X, Y
    
    X_train, Y_train = load_split('X_train', 'Y_train')
    X_val, Y_val = load_split('X_val', 'Y_val')
    X_test, Y_test = load_split('X_test', 'Y_test')
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def save_model(model, model_name: str, output_dir: str = "models/"):
    """Save model using joblib."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}.pkl")
    import joblib
    joblib.dump(model, path)
    return path

def export_deployment_info(model_name, run_id, metrics, feature_names, output_dir: str = "models/"):
    """Export model deployment metadata as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    info = {
        'model_name': model_name,
        'run_id': run_id,
        'expected_features': list(feature_names),
        'feature_count': len(feature_names),
        'scaling': 'StandardScaler (already applied)',
        'performance': metrics
    }
    path = os.path.join(output_dir, "model_deployment_info.json")
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
    return path