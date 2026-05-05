import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from api.config import Config

logger = logging.getLogger(__name__)

TRAIN_HIGH_VELOCITY_THRESHOLD = 4.0
TRAIN_LOW_TRUST_THRESHOLD = 0.0  

MERCHANT_AVG_AMOUNTS = {
    "Electronics": 182.78,
    "Food": 176.09,
    "Grocery": 173.54,
    "Travel": 171.99,
    "Clothing": 177.05,
}
GLOBAL_AVG_AMOUNT = 176.2901


NUMERIC_COLS_TO_SCALE = [
    'transaction_hour', 'device_trust_score', 'velocity_last_24h',
    'cardholder_age', 'log_amount', 'velocity_per_hour', 'relevant_amount'
]

class FraudDetectionEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.expected_features = None
        self._load_artifacts()

    def _load_artifacts(self):
        model_path = Path(Config.MODEL_PATH)
        scaler_path = Path(Config.SCALER_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

        if hasattr(self.model, 'feature_names_in_'):
            self.expected_features = [str(c).strip() for c in self.model.feature_names_in_]
        else:
            self.expected_features = [
                'cardholder_age', 'device_trust_score', 'foreign_transaction',
                'high_risk_abroad', 'is_high_velocity_low_trust',
                'is_night_transaction', 'location_mismatch', 'log_amount',
                'merchant_category_Electronics', 'merchant_category_Food',
                'merchant_category_Grocery', 'merchant_category_Travel',
                'relevant_amount', 'transaction_hour', 'velocity_last_24h', 'velocity_per_hour'
            ]
        logger.info(f"Model expects {len(self.expected_features)} features")

        if scaler_path.exists():
            logger.info(f"Loading scaler from {scaler_path}")
            self.scaler = joblib.load(scaler_path)
        else:
            logger.warning("Scaler not found. Predictions may be inaccurate.")

        logger.info("Artifacts loaded successfully")

    def preprocess(self, raw_features: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([raw_features])

        df['log_amount'] = np.log1p(df['amount'])
        df['is_night_transaction'] = df['transaction_hour'].between(0, 5).astype(int)
        df['velocity_per_hour'] = df['velocity_last_24h'] / np.maximum(df['transaction_hour'], 1)
        df['high_risk_abroad'] = ((df['foreign_transaction'] == 1) & (df['location_mismatch'] == 1)).astype(int)
        df['is_high_velocity_low_trust'] = (
            (df['velocity_last_24h'] > TRAIN_HIGH_VELOCITY_THRESHOLD) &
            (df['device_trust_score'] < TRAIN_LOW_TRUST_THRESHOLD)
        ).astype(int)
        df['merchant_avg'] = df['merchant_category'].map(MERCHANT_AVG_AMOUNTS).fillna(GLOBAL_AVG_AMOUNT)
        df['relevant_amount'] = df['amount'] / (df['merchant_avg'] + 1e-6)
        df = df.drop(columns=['merchant_avg'])

        for cat in ["Electronics", "Food", "Grocery", "Travel"]:
            df[f"merchant_category_{cat}"] = (df['merchant_category'] == cat).astype(int)
        df = df.drop(columns=['merchant_category'])

        df.columns = df.columns.str.strip()

        if self.scaler is not None:
            if hasattr(self.scaler, 'feature_names_in_'):
                cols_to_scale = [c for c in self.scaler.feature_names_in_ if c in df.columns]
            else:
                cols_to_scale = [c for c in NUMERIC_COLS_TO_SCALE if c in df.columns]

            for col in cols_to_scale:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            if cols_to_scale:
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

        df = df.drop(columns=['amount', 'transaction_id'], errors='ignore')


        df = df.reindex(columns=self.expected_features, fill_value=0)

        return df

    def predict(self, raw_features: Dict[str, Any]) -> Dict[str, Any]:
        processed = self.preprocess(raw_features)
        
        try:
            pred = self.model.predict(processed)[0]
            prob = float(self.model.predict_proba(processed)[0, 1]) if hasattr(self.model, 'predict_proba') else None
        except Exception as e:
            logger.warning(f"DataFrame validation failed, falling back to array input: {e}")
            pred = self.model.predict(processed.values)[0]
            prob = float(self.model.predict_proba(processed.values)[0, 1]) if hasattr(self.model, 'predict_proba') else None
            
        return {
            "is_fraud": bool(pred),
            "fraud_probability": round(prob, 4) if prob is not None else None
        }