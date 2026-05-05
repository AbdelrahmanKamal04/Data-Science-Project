"""Test set evaluation, error analysis, and final model export."""

import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from .utils import calculate_business_metrics, export_deployment_info
from .config import SCORE_WEIGHTS

def calculate_composite_scores(results_df: pd.DataFrame, weights: dict = SCORE_WEIGHTS):
    """Calculate composite score for model ranking."""
    df = results_df.copy()
    df['composite_score'] = (
        weights['f1_score'] * df['f1_score'] +
        weights['roc_auc'] * df['roc_auc'] +
        weights['recall'] * df['recall'] +
        weights['false_alarm_penalty'] * (1 - df['false_alarm_rate']/100) +
        weights['savings_ratio'] * (df['potential_savings'] / df['potential_savings'].max())
    )
    return df.sort_values('composite_score', ascending=False)

def evaluate_on_test(models_dict: dict, X_test, y_test):
    """Evaluate multiple models on held-out test set."""
    results = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            'accuracy': (y_pred == y_test.values.ravel()).mean(),
            'precision': (y_pred[y_test.values.ravel() == 1] == 1).mean() if (y_test.values.ravel() == 1).sum() > 0 else 0,
            'recall': (y_pred[y_test.values.ravel() == 1] == 1).mean() if (y_test.values.ravel() == 1).sum() > 0 else 0,
            'f1_score': 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0,
        }
        business = calculate_business_metrics(y_test.values.ravel(), y_pred, y_proba)
        results.append({
            'model_name': name, **metrics, **business,
            'confusion_matrix': confusion_matrix(y_test.values.ravel(), y_pred),
            'y_pred': y_pred, 'y_proba': y_proba
        })
    return pd.DataFrame(results)

def analyze_errors(model, X_test, y_test):
    """Analyze false negatives and false positives."""
    y_pred = model.predict(X_test)
    fn_idx = np.where((y_test.values.ravel() == 1) & (y_pred == 0))[0]
    fp_idx = np.where((y_test.values.ravel() == 0) & (y_pred == 1))[0]
    
    analysis = {
        'total_fn': len(fn_idx), 'total_fp': len(fp_idx),
        'fn_features': X_test.iloc[fn_idx].mean().to_dict() if len(fn_idx) > 0 else {},
        'fp_features': X_test.iloc[fp_idx].mean().to_dict() if len(fp_idx) > 0 else {}
    }
    return analysis

def export_final_model(model, model_info: dict, output_dir: str = "models/"):
    """Save final model and deployment metadata."""
    import joblib
    os.makedirs(output_dir, exist_ok=True)
    
    pkl_path = os.path.join(output_dir, "final_fraud_detection_model.pkl")
    joblib.dump(model, pkl_path)
    
    json_path = export_deployment_info(
        model_name=model_info['model_name'],
        run_id=model_info.get('run_id', 'unknown'),
        metrics={k: float(v) for k, v in model_info.items() if isinstance(v, (int, float))},
        feature_names=model_info.get('feature_names', []),
        output_dir=output_dir
    )
    print(f"✅ Model saved: {pkl_path}")
    print(f"✅ Deployment info saved: {json_path}")
    return pkl_path, json_path