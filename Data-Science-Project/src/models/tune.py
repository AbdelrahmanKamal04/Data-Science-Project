"""Hyperparameter tuning pipeline with MLflow integration."""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import time
import pandas as pd
from .utils import calculate_business_metrics
from .config import PARAM_DISTRIBUTIONS

def get_param_distributions():
    """Return hyperparameter search spaces for tuning."""
    return PARAM_DISTRIBUTIONS

def tune_model(model_name: str, base_model, param_dist: dict, 
               X_train, y_train, X_val, y_val, 
               n_iter: int = 30, cv: int = 3, random_state: int = 42):
    """Tune a single model and log best run to MLflow."""
    run_name = f"{model_name}_tuning_{int(time.time())}"
    
    with mlflow.start_run(run_name=run_name):
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='f1',
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        y_pred = best_model.predict(X_val)
        y_proba = best_model.predict_proba(X_val)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_proba)
        }
        business_metrics = calculate_business_metrics(y_val, y_pred, y_proba)
        
        mlflow.log_params(search.best_params_)
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("cv", cv)
        for k, v in metrics.items(): mlflow.log_metric(k, v)
        for k, v in business_metrics.items(): mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(best_model, "tuned_model")
        
        return {
            'model_name': f"{model_name}_Tuned",
            'base_model_name': model_name,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            **metrics, **business_metrics,
            'model_object': best_model
        }

def run_tuning_pipeline(baseline_results_df: pd.DataFrame, X_train, y_train, X_val, y_val, top_n: int = 3):
    """Tune top N baseline models and return combined results."""
    top_models = baseline_results_df.nlargest(top_n, 'f1_score')['model_name'].tolist()
    tuned_results = []
    
    for base_name in top_models:
        model_key = base_name.split('_')[0]
        if model_key in PARAM_DISTRIBUTIONS:
            base_model = BASELINE_MODELS.get(base_name, {}).get('model')
            if base_model:
                # Reset random state for tuning
                base_model.set_params(random_state=42)
                res = tune_model(model_key, base_model, PARAM_DISTRIBUTIONS[model_key], X_train, y_train, X_val, y_val)
                tuned_results.append(res)
                print(f"✅ Tuned: {res['model_name']} | F1: {res['f1_score']:.4f} | ROI: {res['roi_percentage']:.2f}%")
                
    return pd.concat([baseline_results_df, pd.DataFrame(tuned_results)], ignore_index=True)