import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/final_fraud_detection_model.pkl")
    SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
    APP_NAME = "Fraud Detection API"
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"