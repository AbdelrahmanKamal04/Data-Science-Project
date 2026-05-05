"""Model configurations, hyperparameter spaces, and business constants."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform, loguniform

# Business cost constants
COST_FN = 150
COST_FP = 10
TRANSACTION_VALUE = 200

# Composite score weights (aligned with notebook logic)
SCORE_WEIGHTS = {
    'f1_score': 0.3,
    'roc_auc': 0.3,
    'recall': 0.2,
    'false_alarm_penalty': 0.1,
    'savings_ratio': 0.1
}

# Baseline model configurations
BASELINE_MODELS = {
    'LogisticRegression_Baseline': {
        'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'params': {'max_iter': 1000, 'random_state': 42, 'class_weight': 'balanced'}
    },
    'RandomForest_v1': {
        'model': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced'}
    },
    'XGBoost_v1': {
        'model': xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, scale_pos_weight=10),
        'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'scale_pos_weight': 10}
    },
    'LightGBM_v1': {
        'model': lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, is_unbalance=True),
        'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'is_unbalance': True}
    },
    'GradientBoosting_v1': {
        'model': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42}
    }
}

# Hyperparameter search spaces
PARAM_DISTRIBUTIONS = {
    'LightGBM': {
        'n_estimators': randint(100, 500),
        'max_depth': randint(4, 15),
        'learning_rate': uniform(0.01, 0.3),
        'num_leaves': randint(20, 100),
        'min_child_samples': randint(10, 100),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'reg_alpha': loguniform(0.001, 1.0),
        'reg_lambda': loguniform(0.001, 1.0)
    },
    'XGBoost': {
        'n_estimators': randint(100, 500),
        'max_depth': randint(4, 12),
        'learning_rate': uniform(0.01, 0.3),
        'min_child_weight': randint(1, 10),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
        'gamma': uniform(0, 1),
        'reg_alpha': loguniform(0.001, 1.0),
        'reg_lambda': loguniform(0.001, 1.0)
    },
    'GradientBoosting': {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'subsample': uniform(0.7, 0.3),
        'max_features': ['sqrt', 'log2']
    }
}