import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/features/transactions_with_risk.parquet")


def analyze_rules(df):

    print("\n" + "=" * 70)
    print("FINSIGHT RISK RULE ANALYSIS")
    print("=" * 70)

    # Recreate individual rules
    df = df.copy()

    df["rule_high_value"] = df["is_high_value"] == 1

    df["rule_risky_type"] = df["type"].isin(
        ["TRANSFER", "CASH_OUT"]
    )

    df["rule_high_value_cashout"] = (
        (df["is_high_value"] == 1)
        & (df["type"] == "CASH_OUT")
    )

    df["rule_high_value_transfer"] = (
        (df["is_high_value"] == 1)
        & (df["type"] == "TRANSFER")
    )

    df["rule_origin_error"] = (
        df["abs_origin_balance_error"] <= 1
    )

    rules = {
        "High Value": "rule_high_value",
        "Risky Type": "rule_risky_type",
        "High Value CASH_OUT": "rule_high_value_cashout",
        "High Value TRANSFER": "rule_high_value_transfer",
        "Origin Balance Error <= 1": "rule_origin_error",
    }

    print("\nINDIVIDUAL RULE PERFORMANCE")
    print("-" * 70)

    results = []

    for name, column in rules.items():

        subset = df[df[column]]

        transactions = len(subset)
        fraud_cases = subset["isFraud"].sum()

        if transactions > 0:
            fraud_rate = fraud_cases / transactions
        else:
            fraud_rate = 0

        results.append(
            {
                "rule": name,
                "transactions": transactions,
                "fraud_cases": fraud_cases,
                "fraud_rate": fraud_rate,
            }
        )

    result_df = pd.DataFrame(results)

    print(
        result_df.to_string(
            index=False,
            formatters={
                "fraud_rate": "{:.4%}".format
            }
        )
    )

    # ---------------------------------------------------------
    # SCORE → TYPE ANALYSIS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RISK SCORE × TRANSACTION TYPE")
    print("=" * 70)

    score_type = (
        df.groupby(["risk_score", "type"])
        .agg(
            transactions=("isFraud", "size"),
            fraud_cases=("isFraud", "sum"),
        )
        .reset_index()
    )

    score_type["fraud_rate"] = (
        score_type["fraud_cases"]
        / score_type["transactions"]
    )

    print(
        score_type.to_string(
            index=False,
            formatters={
                "fraud_rate": "{:.4%}".format
            }
        )
    )

    # ---------------------------------------------------------
    # HIGH SCORE TRANSACTIONS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("HIGH SCORE TRANSACTIONS")
    print("=" * 70)

    high_score = df[df["risk_score"] >= 60]

    print(
        high_score[
            [
                "type",
                "amount",
                "risk_score",
                "is_high_value",
                "abs_origin_balance_error",
                "isFraud",
            ]
        ]
        .groupby(["type", "risk_score"])
        .agg(
            transactions=("isFraud", "size"),
            fraud_cases=("isFraud", "sum"),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":

    print("\nLoading risk dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(f"✓ Dataset loaded: {len(df):,} transactions")

    analyze_rules(df)