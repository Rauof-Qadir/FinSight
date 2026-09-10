from pathlib import Path

import pandas as pd

from src.risk.risk_engine import (
    calculate_risk_score,
    assign_risk_level
)


def main():

    print("=" * 60)
    print("FINSIGHT FRAUD RISK ENGINE")
    print("=" * 60)

    input_path = Path(
        "data/features/transaction_features.parquet"
    )

    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / "transactions_with_risk.parquet"
    )

    # --------------------------------------------------
    # Load feature dataset
    # --------------------------------------------------

    print("\nLoading feature dataset...")

    df = pd.read_parquet(input_path)

    print(
        f"✓ Dataset loaded: {len(df):,} transactions"
    )

    # --------------------------------------------------
    # Calculate risk score
    # --------------------------------------------------

    print("\nCalculating risk scores...")

    df = calculate_risk_score(df)

    print("✓ Risk scores calculated")

    # --------------------------------------------------
    # Assign risk levels
    # --------------------------------------------------

    print("\nAssigning risk levels...")

    df = assign_risk_level(df)

    print("✓ Risk levels assigned")

    # --------------------------------------------------
    # Risk distribution
    # --------------------------------------------------

    print("\nRISK LEVEL DISTRIBUTION")
    print("-" * 60)

    risk_distribution = (
        df["risk_level"]
        .value_counts()
        .to_frame("transactions")
    )

    risk_distribution["percentage"] = (
        risk_distribution["transactions"]
        / len(df)
        * 100
    )

    print(risk_distribution)

    # --------------------------------------------------
    # Fraud by risk level
    # --------------------------------------------------

    print("\nFRAUD PERFORMANCE BY RISK LEVEL")
    print("-" * 60)

    risk_performance = (
        df.groupby("risk_level")
        .agg(
            transactions=("isFraud", "count"),
            fraud_cases=("isFraud", "sum"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean")
        )
    )

    risk_performance["fraud_rate"] = (
        risk_performance["fraud_cases"]
        / risk_performance["transactions"]
        * 100
    )

    risk_performance = risk_performance.sort_values(
        "fraud_rate",
        ascending=False
    )

    print(risk_performance)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    print("\nSaving risk dataset...")

    df.to_parquet(
        output_path,
        index=False
    )

    print(
        f"✓ Risk dataset saved successfully!"
    )

    print(f"Location: {output_path}")

    print("\n" + "=" * 60)
    print("RISK ENGINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()