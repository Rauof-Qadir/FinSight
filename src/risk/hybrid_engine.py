from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path(
    "models/xgboost_final.joblib"
)

DEFAULT_THRESHOLD = 0.95


# ============================================================
# LOAD MODEL
# ============================================================

_model_bundle = None


def load_model():
    """
    Load the trained XGBoost model once.
    """

    global _model_bundle

    if _model_bundle is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        _model_bundle = joblib.load(
            MODEL_PATH
        )

    return _model_bundle


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df):
    """
    Prepare real-time transaction features
    for the XGBoost model.
    """

    bundle = load_model()

    model_features = bundle["features"]

    features = [
        "step",
        "type",
        "amount",
        "amount_log",
        "oldbalanceOrg",
        "oldbalanceDest",
        "is_high_value",
        "is_zero_amount",
        "is_transfer",
        "is_cash_out",
        "is_payment",
        "is_cash_in",
        "is_debit",
        "hour_of_day",
        "day",
    ]

    data = df[features].copy()

    # Encode transaction type
    data = pd.get_dummies(
        data,
        columns=["type"],
        dtype=int,
    )

    # Make sure prediction data has exactly
    # the same columns used during training.
    data = data.reindex(
        columns=model_features,
        fill_value=0,
    )

    return data


# ============================================================
# ML PREDICTION
# ============================================================

def predict_fraud_probability(df):
    """
    Generate fraud probability using XGBoost.
    """

    bundle = load_model()

    model = bundle["model"]

    X = prepare_features(df)

    probabilities = model.predict_proba(
        X
    )[:, 1]

    return probabilities


# ============================================================
# BUSINESS RULES
# ============================================================

def apply_business_rules(
    row,
    fraud_probability,
):
    """
    Apply business rules on top of ML probability.

    Returns:
        decision
        risk_level
        reason
    """

    # --------------------------------------------------------
    # CRITICAL ML SIGNAL
    # --------------------------------------------------------

    if fraud_probability >= 0.95:

        return (
            "BLOCK",
            "HIGH",
            "Very high ML fraud probability",
        )


    # --------------------------------------------------------
    # HIGH-VALUE CASH-OUT
    # --------------------------------------------------------

    if (
        row["type"] == "CASH_OUT"
        and row["is_high_value"] == 1
    ):

        return (
            "BLOCK",
            "HIGH",
            "High-value CASH_OUT transaction",
        )


    # --------------------------------------------------------
    # HIGH-VALUE TRANSFER
    # --------------------------------------------------------

    if (
        row["type"] == "TRANSFER"
        and row["is_high_value"] == 1
        and fraud_probability >= 0.70
    ):

        return (
            "REVIEW",
            "HIGH",
            "High-value TRANSFER with elevated ML risk",
        )


    # --------------------------------------------------------
    # ZERO AMOUNT
    # --------------------------------------------------------

    if row["is_zero_amount"] == 1:

        return (
            "REVIEW",
            "MEDIUM",
            "Zero-value transaction anomaly",
        )


    # --------------------------------------------------------
    # MEDIUM ML RISK
    # --------------------------------------------------------

    if fraud_probability >= 0.50:

        return (
            "REVIEW",
            "MEDIUM",
            "Elevated ML fraud probability",
        )


    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    return (
        "ALLOW",
        "LOW",
        "Low fraud probability",
    )


# ============================================================
# HYBRID PREDICTION
# ============================================================

def predict_transactions(df):
    """
    Complete hybrid fraud decision pipeline.

    Input:
        DataFrame containing transaction features.

    Output:
        DataFrame containing:
        - fraud_probability
        - decision
        - risk_level
        - reason
    """

    data = df.copy()

    probabilities = (
        predict_fraud_probability(
            data
        )
    )

    decisions = []

    risk_levels = []

    reasons = []


    for index, row in data.iterrows():

        decision, risk_level, reason = (
            apply_business_rules(
                row,
                probabilities[index],
            )
        )

        decisions.append(
            decision
        )

        risk_levels.append(
            risk_level
        )

        reasons.append(
            reason
        )


    data["fraud_probability"] = (
        probabilities
    )

    data["decision"] = decisions

    data["risk_level"] = risk_levels

    data["decision_reason"] = reasons


    return data


# ============================================================
# SINGLE TRANSACTION PREDICTION
# ============================================================

def predict_single_transaction(
    transaction,
):
    """
    Predict a single transaction.

    transaction must be a dictionary containing
    the required real-time features.
    """

    df = pd.DataFrame(
        [transaction]
    )

    result = predict_transactions(
        df
    )

    return result.iloc[0].to_dict()