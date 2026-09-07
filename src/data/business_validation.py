import pandas as pd


def validate_transaction_types(df: pd.DataFrame) -> None:
    """Validate allowed transaction types."""

    allowed_types = {
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    }

    actual_types = set(df["type"].unique())

    invalid_types = actual_types - allowed_types

    print("\nTRANSACTION TYPE VALIDATION")
    print("-" * 60)

    if invalid_types:
        print(f"⚠ Invalid transaction types found: {invalid_types}")
    else:
        print("✓ All transaction types are valid")

    print("\nTransaction Type Distribution:")

    print(df["type"].value_counts())


def validate_zero_amount_transactions(df: pd.DataFrame) -> None:
    """Check transactions with zero amount."""

    zero_amount = (df["amount"] == 0).sum()

    print("\nZERO AMOUNT TRANSACTIONS")
    print("-" * 60)

    print(f"Zero amount transactions: {zero_amount:,}")

    print(f"Percentage: {(zero_amount / len(df)) * 100:.4f}%")


def validate_balance_consistency(df: pd.DataFrame) -> None:
    """
    Analyze balance consistency.

    Note:
    PaySim contains known balance-update limitations,
    so inconsistencies should be investigated, not automatically removed.
    """

    print("\nBALANCE CONSISTENCY ANALYSIS")
    print("-" * 60)

    outgoing_types = ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

    outgoing = df[df["type"].isin(outgoing_types)].copy()

    outgoing["expected_balance"] = (
        outgoing["oldbalanceOrg"] - outgoing["amount"]
    )

    outgoing["balance_difference"] = (
        outgoing["newbalanceOrig"]
        - outgoing["expected_balance"]
    )

    tolerance = 0.01

    inconsistent = (
        outgoing["balance_difference"].abs() > tolerance
    ).sum()

    print(f"Outgoing transactions checked: {len(outgoing):,}")

    print(
        f"Balance inconsistencies: {inconsistent:,}"
    )

    print(
        f"Inconsistency rate: "
        f"{(inconsistent / len(outgoing)) * 100:.2f}%"
    )


def validate_fraud_distribution(df: pd.DataFrame) -> None:
    """Analyze fraud class distribution."""

    print("\nFRAUD DISTRIBUTION")
    print("-" * 60)

    fraud_counts = df["isFraud"].value_counts()

    legitimate = fraud_counts.get(0, 0)
    fraud = fraud_counts.get(1, 0)

    fraud_rate = (fraud / len(df)) * 100

    print(f"Legitimate transactions: {legitimate:,}")
    print(f"Fraudulent transactions: {fraud:,}")
    print(f"Fraud rate: {fraud_rate:.4f}%")

    print("\nFraud by Transaction Type:")

    fraud_by_type = (
        df.groupby("type", observed=True)["isFraud"]
        .agg(["count", "sum", "mean"])
        .rename(
            columns={
                "count": "transactions",
                "sum": "fraud_cases",
                "mean": "fraud_rate"
            }
        )
    )

    fraud_by_type["fraud_rate"] *= 100

    print(fraud_by_type)


def validate_flagged_transactions(df: pd.DataFrame) -> None:
    """Analyze system-flagged transactions."""

    print("\nFLAGGED TRANSACTION ANALYSIS")
    print("-" * 60)

    flagged = df["isFlaggedFraud"].sum()

    print(f"System flagged transactions: {flagged:,}")

    flagged_fraud = df[
        (df["isFlaggedFraud"] == 1)
        & (df["isFraud"] == 1)
    ].shape[0]

    print(
        f"Flagged transactions that are fraud: "
        f"{flagged_fraud:,}"
    )

    if flagged > 0:

        precision = (flagged_fraud / flagged) * 100

        print(
            f"System flag precision: {precision:.2f}%"
        )


def run_business_validation(df: pd.DataFrame) -> None:
    """Run all business/data quality validations."""

    print("\n" + "=" * 60)
    print("FINANCIAL BUSINESS RULE VALIDATION")
    print("=" * 60)

    validate_transaction_types(df)

    validate_zero_amount_transactions(df)

    validate_balance_consistency(df)

    validate_fraud_distribution(df)

    validate_flagged_transactions(df)