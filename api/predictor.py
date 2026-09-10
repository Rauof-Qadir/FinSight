import joblib
import numpy as np
import pandas as pd
from pathlib import Path


MODEL_PATH = Path("models/xgboost_final.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


# Load model bundle
model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
MODEL_FEATURES = model_bundle["features"]

# Production threshold selected during validation
THRESHOLD = model_bundle["threshold"]


def prepare_features(transaction: dict) -> pd.DataFrame:
    """
    Convert raw transaction data into the exact feature
    structure expected by the XGBoost model.
    """

    df = pd.DataFrame([transaction])

    # -----------------------------
    # Derived numerical features
    # -----------------------------

    df["amount_log"] = np.log1p(df["amount"])

    df["is_zero_amount"] = (
        df["amount"] == 0
    ).astype(int)

    # Current training threshold for high-value transactions
    # This should eventually be stored explicitly as a separate
    # training artifact/config.
    HIGH_VALUE_THRESHOLD = 1615979.50

    df["is_high_value"] = (
        df["amount"] >= HIGH_VALUE_THRESHOLD
    ).astype(int)

    # -----------------------------
    # Transaction type features
    # -----------------------------

    df["is_transfer"] = (
        df["type"] == "TRANSFER"
    ).astype(int)

    df["is_cash_out"] = (
        df["type"] == "CASH_OUT"
    ).astype(int)

    df["is_payment"] = (
        df["type"] == "PAYMENT"
    ).astype(int)

    df["is_cash_in"] = (
        df["type"] == "CASH_IN"
    ).astype(int)

    df["is_debit"] = (
        df["type"] == "DEBIT"
    ).astype(int)

    # -----------------------------
    # Time features
    # -----------------------------

    df["hour_of_day"] = df["step"] % 24
    df["day"] = df["step"] // 24

    # -----------------------------
    # One-hot encode transaction type
    # -----------------------------

    df = pd.get_dummies(
        df,
        columns=["type"],
        dtype=int
    )

    # -----------------------------
    # EXACT model feature order
    # -----------------------------

    df = df.reindex(
        columns=MODEL_FEATURES,
        fill_value=0
    )

    return df


def predict_transaction(transaction: dict) -> dict:

    X = prepare_features(transaction)

    probability = model.predict_proba(X)[0, 1]
    probability = float(probability)

    # -----------------------------
    # Production decision policy
    # -----------------------------

    if probability >= THRESHOLD:

        decision = "BLOCK"
        risk_level = "HIGH"
        reason = "Very high ML fraud probability"

    elif probability >= 0.50:

        decision = "REVIEW"
        risk_level = "MEDIUM"
        reason = "Elevated ML fraud probability"

    else:

        decision = "ALLOW"
        risk_level = "LOW"
        reason = "Low fraud probability"

    return {
        "fraud_probability": probability,
        "risk_level": risk_level,
        "decision": decision,
        "decision_reason": reason,
    }


def predict_batch(transactions: list[dict]) -> list[dict]:
    """
    Predict fraud probability for multiple transactions.
    """

    predictions = []

    for transaction in transactions:
        result = predict_transaction(transaction)
        predictions.append(result)

    return predictions