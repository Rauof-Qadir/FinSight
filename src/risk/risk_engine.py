import numpy as np
import pandas as pd

def calculate_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a rule-based fraud risk score for FinSight.

    The scoring framework combines transaction characteristics
    and validated risk signals discovered during exploratory analysis.

    Score range: 0 - 100.
    """

    df = df.copy()

    # -----------------------------------------------
    # Initialize risk score
    # -----------------------------------------------

    df["risk_score"] = 0

    # -----------------------------------------------
    # Rule 1: High-value transaction
    # -----------------------------------------------

    df.loc[
        df["is_high_value"] == 1,
        "risk_score"
    ] += 20

    # -----------------------------------------------
    # Rule 2: High-risk transaction type
    # -----------------------------------------------

    df.loc[
        df["type"].isin(["TRANSFER", "CASH_OUT"]),
        "risk_score"
    ] += 15

    # -----------------------------------------------
    # Rule 3: High-value CASH_OUT
    # -----------------------------------------------

    df.loc[
        (df["type"] == "CASH_OUT")
        & (df["is_high_value"] == 1),
        "risk_score"
    ] += 40

    # -----------------------------------------------
    # Rule 4: High-value TRANSFER
    # -----------------------------------------------

    df.loc[
        (df["type"] == "TRANSFER")
        & (df["is_high_value"] == 1),
        "risk_score"
    ] += 20

    # -----------------------------------------------
    # Rule 5: Origin balance pattern
    # -----------------------------------------------

    df.loc[
        df["abs_origin_balance_error"] <= 1,
        "risk_score"
    ] += 10

    # -----------------------------------------------
    # Limit score to 100
    # -----------------------------------------------

    df["risk_score"] = (
        df["risk_score"]
        .clip(upper=100)
        .astype("int8")
    )

    return df


def assign_risk_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign business-friendly risk categories.
    """

    df = df.copy()

    conditions = [
        df["risk_score"] < 20,

        (df["risk_score"] >= 20)
        & (df["risk_score"] < 50),

        df["risk_score"] >= 50
    ]

    choices = [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]

    df["risk_level"] = np.select(
        conditions,
        choices,
        default="LOW"
    )

    return df
