from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    step: int = Field(..., ge=0)
    type: str = Field(...)
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_level: str
    decision: str
    decision_reason: str


from typing import List

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    step: int = Field(..., ge=0)
    type: str = Field(...)
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_level: str
    decision: str
    decision_reason: str


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]