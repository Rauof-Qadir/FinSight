import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/features/transactions_with_risk.parquet")


def evaluate_risk_engine(df):
    total_fraud = df["isFraud"].sum()
    total_transactions = len(df)

    print("\n" + "=" * 60)
    print("FINSIGHT RISK ENGINE EVALUATION")
    print("=" * 60)

    print(f"\nTotal transactions: {total_transactions:,}")
    print(f"Total fraud cases: {total_fraud:,}")

    # ---------------------------------------------------------
    # HIGH RISK
    # ---------------------------------------------------------

    high = df[df["risk_level"] == "HIGH"]

    high_fraud = high["isFraud"].sum()
    high_legitimate = len(high) - high_fraud

    high_precision = high_fraud / len(high)
    high_recall = high_fraud / total_fraud

    print("\nHIGH RISK")
    print("-" * 60)
    print(f"Transactions:        {len(high):,}")
    print(f"Fraud cases:         {high_fraud:,}")
    print(f"Legitimate cases:    {high_legitimate:,}")
    print(f"Precision:            {high_precision:.4%}")
    print(f"Recall:               {high_recall:.4%}")
    print(f"False positives:      {high_legitimate:,}")
    print(f"Fraud amount:         {high.loc[high['isFraud'] == 1, 'amount'].sum():,.2f}")

    # ---------------------------------------------------------
    # MEDIUM + HIGH
    # ---------------------------------------------------------

    flagged = df[df["risk_level"].isin(["MEDIUM", "HIGH"])]

    flagged_fraud = flagged["isFraud"].sum()
    flagged_legitimate = len(flagged) - flagged_fraud

    flagged_precision = flagged_fraud / len(flagged)
    flagged_recall = flagged_fraud / total_fraud

    print("\nMEDIUM + HIGH")
    print("-" * 60)
    print(f"Transactions:        {len(flagged):,}")
    print(f"Fraud cases:         {flagged_fraud:,}")
    print(f"Legitimate cases:    {flagged_legitimate:,}")
    print(f"Precision:            {flagged_precision:.4%}")
    print(f"Recall:               {flagged_recall:.4%}")
    print(f"False positives:      {flagged_legitimate:,}")
    print(f"Fraud amount:         {flagged.loc[flagged['isFraud'] == 1, 'amount'].sum():,.2f}")

    # ---------------------------------------------------------
    # LOW RISK
    # ---------------------------------------------------------

    low = df[df["risk_level"] == "LOW"]

    low_fraud = low["isFraud"].sum()

    print("\nLOW RISK")
    print("-" * 60)
    print(f"Transactions:        {len(low):,}")
    print(f"Fraud cases:         {low_fraud:,}")
    print(f"Missed fraud:        {low_fraud:,}")
    print(f"Missed fraud rate:   {low_fraud / len(low):.6%}")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("BUSINESS SUMMARY")
    print("=" * 60)

    print(
        f"\nReview workload (MEDIUM + HIGH): "
        f"{len(flagged):,} transactions "
        f"({len(flagged) / total_transactions:.2%} of all transactions)"
    )

    print(
        f"Fraud captured: "
        f"{flagged_fraud:,} / {total_fraud:,} "
        f"({flagged_recall:.2%})"
    )

    print(
        f"Fraud missed: "
        f"{low_fraud:,} / {total_fraud:,} "
        f"({low_fraud / total_fraud:.2%})"
    )


if __name__ == "__main__":

    print("\nLoading risk dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(f"✓ Dataset loaded: {len(df):,} transactions")

    evaluate_risk_engine(df)