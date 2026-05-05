from pydantic import BaseModel, Field
from typing import Optional, Literal

class PredictionRequest(BaseModel):
    amount: float = Field(..., gt=0, example=84.47, description="Transaction amount")
    transaction_hour: int = Field(..., ge=0, le=23, example=22, description="Hour of transaction (0-23)")
    merchant_category: Literal[
        "Electronics", 
        "Travel", 
        "Grocery", 
        "Food", 
        "Clothing"
    ] = Field(..., example="Electronics", description="Merchant category")
    foreign_transaction: int = Field(..., ge=0, le=1, example=0, description="1 if foreign, 0 otherwise")
    location_mismatch: int = Field(..., ge=0, le=1, example=0, description="1 if location mismatch, 0 otherwise")
    device_trust_score: float = Field(..., ge=25, le=99, example=66, description="Device trust score")
    velocity_last_24h: float = Field(..., ge=0, example=3, description="Transactions in last 24h")
    cardholder_age: int = Field(..., ge=18, le=100, example=40, description="Cardholder age")

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: Optional[float] = Field(..., ge=0, le=1)
    model_version: str = "1.0.0"
    status: str = "success"