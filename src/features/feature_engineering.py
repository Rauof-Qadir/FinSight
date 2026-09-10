import numpy as np
import pandas as pd


def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction-level features for FinSight.

    These features are based only on information available
    at the time of the transaction.
    """

    df = df.copy()

    # --------------------------------------------------
    # 1. Transaction Amount Features
    # --------------------------------------------------

    df["amount_log"] = np.log1p(df["amount"])

    df["is_zero_amount"] = (
        df["amount"] == 0
    ).astype("int8")

    # High-value threshold based on the dataset distribution.
    # This will later be replaced with a train-derived threshold
    # when building the ML pipeline.
    high_value_threshold = df["amount"].quantile(0.99)

    df["is_high_value"] = (
        df["amount"] >= high_value_threshold
    ).astype("int8")

    # --------------------------------------------------
    # 2. Transaction Type Features
    # --------------------------------------------------

    df["is_transfer"] = (
        df["type"] == "TRANSFER"
    ).astype("int8")

    df["is_cash_out"] = (
        df["type"] == "CASH_OUT"
    ).astype("int8")

    df["is_payment"] = (
        df["type"] == "PAYMENT"
    ).astype("int8")

    df["is_cash_in"] = (
        df["type"] == "CASH_IN"
    ).astype("int8")

    df["is_debit"] = (
        df["type"] == "DEBIT"
    ).astype("int8")

    # --------------------------------------------------
    # 3. Origin Balance Features
    # --------------------------------------------------

    df["origin_balance_change"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    )

    df["origin_balance_error"] = (
        df["origin_balance_change"] - df["amount"]
    )

    # --------------------------------------------------
    # 4. Destination Balance Features
    # --------------------------------------------------

    df["destination_balance_change"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    )

    df["destination_balance_error"] = (
        df["destination_balance_change"] - df["amount"]
    )

    df["abs_origin_balance_error"] = (
        df["origin_balance_error"].abs()
    )

    df["abs_destination_balance_error"] = (
        df["destination_balance_error"].abs()
    )

    # --------------------------------------------------
    # 5. Time Features
    # --------------------------------------------------

    # PaySim step represents hourly simulation steps

    df["hour_of_day"] = df["step"] % 24

    df["day"] = df["step"] // 24

    return df