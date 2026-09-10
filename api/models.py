from datetime import datetime

from sqlalchemy import Column, Integer, Float, String, DateTime

from api.database import Base


class FraudPrediction(Base):

    __tablename__ = "fraud_predictions"

    id = Column(Integer, primary_key=True, index=True)

    step = Column(Integer, nullable=False)

    type = Column(String(20), nullable=False)

    amount = Column(Float, nullable=False)

    oldbalance_org = Column(Float, nullable=False)

    oldbalance_dest = Column(Float, nullable=False)

    fraud_probability = Column(Float, nullable=False)

    risk_level = Column(String(20), nullable=False)

    decision = Column(String(20), nullable=False)

    decision_reason = Column(String(255), nullable=False)

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )