from api.database import SessionLocal
from api.models import FraudPrediction
from sqlalchemy import func

from fastapi import FastAPI, HTTPException
from api.schemas import (
    TransactionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

from api.predictor import (
    predict_transaction,
    predict_batch,
)




# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="FinSight Fraud Detection API",
    description=(
        "Real-time FinTech transaction fraud "
        "detection and risk decisioning API."
    ),
    version="1.0.0",
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "service": "FinSight Fraud Detection API",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost",
    }


# ============================================================
# FRAUD PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    transaction: TransactionRequest,
):
    db = SessionLocal()

    try:
        transaction_data = transaction.model_dump()

        # ML prediction
        result = predict_transaction(transaction_data)

        # Save prediction to PostgreSQL
        prediction = FraudPrediction(
            step=transaction.step,
            type=transaction.type,
            amount=transaction.amount,
            oldbalance_org=transaction.oldbalanceOrg,
            oldbalance_dest=transaction.oldbalanceDest,
            fraud_probability=result["fraud_probability"],
            risk_level=result["risk_level"],
            decision=result["decision"],
            decision_reason=result["decision_reason"],
        )

        db.add(prediction)
        db.commit()

        return result

    except Exception as e:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:
        db.close()

@app.get("/model-info")
def model_info():
    from api.predictor import (
        model_bundle,
    )

    return {
        "model": "XGBoost",
        "model_file": "xgboost_final.joblib",
        "threshold": 0.95,
        "features": model_bundle["features"],
    }



@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch_transactions(
    request: BatchPredictionRequest,
):
    db = SessionLocal()

    try:
        transactions = [
            transaction.model_dump()
            for transaction in request.transactions
        ]

        # Generate predictions
        predictions = predict_batch(transactions)

        # Save every prediction
        for transaction, result in zip(
            request.transactions,
            predictions,
        ):

            prediction = FraudPrediction(
                step=transaction.step,
                type=transaction.type,
                amount=transaction.amount,
                oldbalance_org=transaction.oldbalanceOrg,
                oldbalance_dest=transaction.oldbalanceDest,
                fraud_probability=result["fraud_probability"],
                risk_level=result["risk_level"],
                decision=result["decision"],
                decision_reason=result["decision_reason"],
            )

            db.add(prediction)

        db.commit()

        return {
            "predictions": predictions
        }

    except Exception as e:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:
        db.close()

@app.get("/predictions")
def get_predictions(
    page: int = 1,
    page_size: int = 50,
    decision: str | None = None,
    risk_level: str | None = None,
    transaction_type: str | None = None,
    min_amount: float | None = None,
):
    db = SessionLocal()

    try:
        # -----------------------------
        # Validate pagination
        # -----------------------------

        if page < 1:
            raise HTTPException(
                status_code=400,
                detail="page must be >= 1",
            )

        if page_size < 1 or page_size > 500:
            raise HTTPException(
                status_code=400,
                detail="page_size must be between 1 and 500",
            )

        # -----------------------------
        # Base query
        # -----------------------------

        query = db.query(FraudPrediction)

        # -----------------------------
        # Filters
        # -----------------------------

        if decision:
            query = query.filter(
                FraudPrediction.decision == decision.upper()
            )

        if risk_level:
            query = query.filter(
                FraudPrediction.risk_level == risk_level.upper()
            )

        if transaction_type:
            query = query.filter(
                FraudPrediction.type == transaction_type.upper()
            )

        if min_amount is not None:
            query = query.filter(
                FraudPrediction.amount >= min_amount
            )

        # -----------------------------
        # Total matching records
        # -----------------------------

        total = query.count()

        # -----------------------------
        # Pagination
        # -----------------------------

        offset = (page - 1) * page_size

        predictions = (
            query
            .order_by(
                FraudPrediction.id.desc()
            )
            .offset(offset)
            .limit(page_size)
            .all()
        )

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (
                (total + page_size - 1) // page_size
            ),
            "predictions": [
                {
                    "id": prediction.id,
                    "step": prediction.step,
                    "type": prediction.type,
                    "amount": prediction.amount,
                    "fraud_probability": prediction.fraud_probability,
                    "risk_level": prediction.risk_level,
                    "decision": prediction.decision,
                    "decision_reason": prediction.decision_reason,
                    "created_at": prediction.created_at,
                }
                for prediction in predictions
            ],
        }

    finally:
        db.close()


@app.get("/predictions/high-risk")
def get_high_risk_predictions(limit: int = 100):
    db = SessionLocal()

    try:
        predictions = (
            db.query(FraudPrediction)
            .filter(
                FraudPrediction.risk_level.in_(
                    ["HIGH", "MEDIUM"]
                )
            )
            .order_by(
                FraudPrediction.fraud_probability.desc()
            )
            .limit(limit)
            .all()
        )

        return {
            "count": len(predictions),
            "predictions": [
                {
                    "id": prediction.id,
                    "step": prediction.step,
                    "type": prediction.type,
                    "amount": prediction.amount,
                    "fraud_probability": prediction.fraud_probability,
                    "risk_level": prediction.risk_level,
                    "decision": prediction.decision,
                    "decision_reason": prediction.decision_reason,
                    "created_at": prediction.created_at,
                }
                for prediction in predictions
            ],
        }

    finally:
        db.close()

@app.get("/predictions/stats")
def get_prediction_stats():
    db = SessionLocal()

    try:

        total_predictions = (
            db.query(FraudPrediction)
            .count()
        )

        total_amount = (
            db.query(
                func.coalesce(
                    func.sum(FraudPrediction.amount),
                    0
                )
            )
            .scalar()
        )

        blocked = (
            db.query(FraudPrediction)
            .filter(
                FraudPrediction.decision == "BLOCK"
            )
            .count()
        )

        reviewed = (
            db.query(FraudPrediction)
            .filter(
                FraudPrediction.decision == "REVIEW"
            )
            .count()
        )

        allowed = (
            db.query(FraudPrediction)
            .filter(
                FraudPrediction.decision == "ALLOW"
            )
            .count()
        )

        high_risk = (
            db.query(FraudPrediction)
            .filter(
                FraudPrediction.risk_level == "HIGH"
            )
            .count()
        )

        average_probability = (
            db.query(
                func.coalesce(
                    func.avg(
                        FraudPrediction.fraud_probability
                    ),
                    0
                )
            )
            .scalar()
        )

        return {
            "total_predictions": total_predictions,
            "total_transaction_value": float(total_amount),
            "blocked_transactions": blocked,
            "review_transactions": reviewed,
            "allowed_transactions": allowed,
            "high_risk_transactions": high_risk,
            "average_fraud_probability": float(
                average_probability
            ),
        }

    finally:
        db.close()




@app.get("/predictions/analytics")
def prediction_analytics():
    db = SessionLocal()

    try:
        total = db.query(FraudPrediction).count()

        blocked = (
            db.query(FraudPrediction)
            .filter(FraudPrediction.decision == "BLOCK")
            .count()
        )

        reviewed = (
            db.query(FraudPrediction)
            .filter(FraudPrediction.decision == "REVIEW")
            .count()
        )

        allowed = (
            db.query(FraudPrediction)
            .filter(FraudPrediction.decision == "ALLOW")
            .count()
        )

        total_value = db.query(
            func.coalesce(func.sum(FraudPrediction.amount), 0)
        ).scalar()

        blocked_value = db.query(
            func.coalesce(func.sum(FraudPrediction.amount), 0)
        ).filter(
            FraudPrediction.decision == "BLOCK"
        ).scalar()

        average_probability = db.query(
            func.coalesce(func.avg(
                FraudPrediction.fraud_probability
            ), 0)
        ).scalar()

        return {
            "total_predictions": total,
            "allowed": allowed,
            "reviewed": reviewed,
            "blocked": blocked,
            "total_transaction_value": float(total_value),
            "blocked_transaction_value": float(blocked_value),
            "average_fraud_probability": float(
                average_probability
            ),
        }

    finally:
        db.close()
        

@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int):
    db = SessionLocal()

    try:

        prediction = (
            db.query(FraudPrediction)
            .filter(
                FraudPrediction.id == prediction_id
            )
            .first()
        )

        if prediction is None:
            raise HTTPException(
                status_code=404,
                detail="Prediction not found",
            )

        return {
            "id": prediction.id,
            "step": prediction.step,
            "type": prediction.type,
            "amount": prediction.amount,
            "oldbalance_org": prediction.oldbalance_org,
            "oldbalance_dest": prediction.oldbalance_dest,
            "fraud_probability": prediction.fraud_probability,
            "risk_level": prediction.risk_level,
            "decision": prediction.decision,
            "decision_reason": prediction.decision_reason,
            "created_at": prediction.created_at,
        }

    finally:
        db.close()



@app.get("/predictions/blocked")
def get_blocked_predictions(
    page: int = 1,
    page_size: int = 50,
):
    db = SessionLocal()

    try:

        if page < 1:
            raise HTTPException(
                status_code=400,
                detail="page must be >= 1",
            )

        if page_size < 1 or page_size > 500:
            raise HTTPException(
                status_code=400,
                detail="page_size must be between 1 and 500",
            )

        query = (
            db.query(FraudPrediction)
            .filter(
                FraudPrediction.decision == "BLOCK"
            )
        )

        total = query.count()

        offset = (page - 1) * page_size

        predictions = (
            query
            .order_by(
                FraudPrediction.fraud_probability.desc()
            )
            .offset(offset)
            .limit(page_size)
            .all()
        )

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "predictions": [
                {
                    "id": p.id,
                    "step": p.step,
                    "type": p.type,
                    "amount": p.amount,
                    "fraud_probability": p.fraud_probability,
                    "risk_level": p.risk_level,
                    "decision": p.decision,
                    "decision_reason": p.decision_reason,
                    "created_at": p.created_at,
                }
                for p in predictions
            ],
        }

    finally:
        db.close()
