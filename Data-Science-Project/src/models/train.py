"""Baseline model training and MLflow logging."""

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import pandas as pd
from .utils import calculate_business_metrics, setup_mlflow
from .config import BASELINE_MODELS
from datetime import datetime

def apply_smote(X, y, random_state: int = 42):
    """Apply SMOTE to balance classes. Returns DataFrame/Series."""
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)
    if isinstance(y_res, pd.DataFrame):
        y_res = y_res.squeeze()
    return X_res, y_res

def train_and_log_baseline(model_name: str, model, params: dict, 
                           X_train, y_train, X_val, y_val, 
                           mlflow_run_name: str):
    """Train a single baseline model and log to MLflow."""
    with mlflow.start_run(run_name=mlflow_run_name):
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_proba) if y_proba is not None else 0.0
        }
        business_metrics = calculate_business_metrics(y_val, y_pred, y_proba)
        
        mlflow.log_params(params)
        for k, v in metrics.items(): mlflow.log_metric(k, v)
        for k, v in business_metrics.items(): mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, "model")
        
        return {
            'model_name': model_name,
            'run_name': mlflow_run_name,
            **metrics, **business_metrics,
            'model_object': model
        }

def run_baseline_training(X_train, y_train, X_val, y_val, tracking_uri="mlruns", experiment_name="credit_card_fraud_detection"):
    """Train all baseline models, log to MLflow, return results DataFrame."""
    setup_mlflow(tracking_uri, experiment_name)
    results = []
    
    for name, config in BASELINE_MODELS.items():
        run_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        res = train_and_log_baseline(name, config['model'], config['params'], X_train, y_train, X_val, y_val, run_name)
        results.append(res)
        print(f"✅ Trained: {name} | F1: {res['f1_score']:.4f} | ROI: {res['roi_percentage']:.2f}%")
        
    return pd.DataFrame(results)