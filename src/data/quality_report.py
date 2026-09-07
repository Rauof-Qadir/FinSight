import pandas as pd

from pathlib import Path


def generate_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a professional data quality summary report."""

    total_rows = len(df)

    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    zero_amount_transactions = int(
        (df["amount"] == 0).sum()
    )

    fraud_transactions = int(
        df["isFraud"].sum()
    )

    flagged_transactions = int(
        df["isFlaggedFraud"].sum()
    )

    fraud_rate = (
        fraud_transactions / total_rows * 100
    )

    zero_amount_rate = (
        zero_amount_transactions / total_rows * 100
    )

    flagged_rate = (
        flagged_transactions / total_rows * 100
    )

    flagged_fraud = int(
        (
            (df["isFlaggedFraud"] == 1)
            & (df["isFraud"] == 1)
        ).sum()
    )

    flag_precision = (
        flagged_fraud / flagged_transactions * 100
        if flagged_transactions > 0
        else 0
    )

    flag_recall = (
        flagged_fraud / fraud_transactions * 100
        if fraud_transactions > 0
        else 0
    )

    report = pd.DataFrame({
        "metric": [
            "Total Rows",
            "Total Columns",
            "Missing Values",
            "Duplicate Rows",
            "Zero Amount Transactions",
            "Zero Amount Rate (%)",
            "Fraud Transactions",
            "Fraud Rate (%)",
            "Flagged Transactions",
            "Flagged Transaction Rate (%)",
            "Flag Precision (%)",
            "Flag Recall (%)"
        ],
        "value": [
            total_rows,
            df.shape[1],
            total_missing,
            duplicate_rows,
            zero_amount_transactions,
            round(zero_amount_rate, 6),
            fraud_transactions,
            round(fraud_rate, 6),
            flagged_transactions,
            round(flagged_rate, 6),
            round(flag_precision, 6),
            round(flag_recall, 6)
        ]
    })

    return report


def save_data_quality_report(
    report: pd.DataFrame,
    output_path: str | Path 
) -> None:
    """Save the data quality report."""

    report.to_csv(output_path, index=False)

    print("\n✓ Data Quality Report saved successfully!")
    print(f"Location: {output_path}")