import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/features/transactions_with_risk.parquet")


def analyze_scores(df):

    print("\n" + "=" * 70)
    print("FINSIGHT RISK SCORE ANALYSIS")
    print("=" * 70)

    # Score-level performance
    score_analysis = (
        df.groupby("risk_score")
        .agg(
            transactions=("isFraud", "size"),
            fraud_cases=("isFraud", "sum"),
            total_amount=("amount", "sum"),
        )
        .reset_index()
    )

    score_analysis["fraud_rate"] = (
        score_analysis["fraud_cases"]
        / score_analysis["transactions"]
    )

    print("\nRISK SCORE PERFORMANCE")
    print("-" * 70)

    print(
        score_analysis.to_string(
            index=False,
            formatters={
                "fraud_rate": "{:.4%}".format,
                "total_amount": "{:,.2f}".format,
            }
        )
    )

    # ---------------------------------------------------------
    # THRESHOLD ANALYSIS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    total_fraud = df["isFraud"].sum()
    total_transactions = len(df)

    thresholds = [20, 30, 40, 50, 60, 70, 80, 90]

    results = []

    for threshold in thresholds:

        flagged = df[df["risk_score"] >= threshold]

        transactions = len(flagged)
        fraud_cases = flagged["isFraud"].sum()

        if transactions == 0:
            continue

        precision = fraud_cases / transactions
        recall = fraud_cases / total_fraud
        workload = transactions / total_transactions

        results.append(
            {
                "threshold": threshold,
                "transactions": transactions,
                "workload_pct": workload,
                "fraud_cases": fraud_cases,
                "precision": precision,
                "recall": recall,
            }
        )

    threshold_df = pd.DataFrame(results)

    print(
        threshold_df.to_string(
            index=False,
            formatters={
                "workload_pct": "{:.2%}".format,
                "precision": "{:.4%}".format,
                "recall": "{:.2%}".format,
            }
        )
    )


if __name__ == "__main__":

    print("\nLoading risk dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(f"✓ Dataset loaded: {len(df):,} transactions")

    analyze_scores(df)