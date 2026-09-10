from pathlib import Path
import pandas as pd


INPUT_PATH = Path(
    "data/features/transactions_with_risk.parquet"
)

OUTPUT_PATH = Path(
    "data/modeling/ml_dataset.parquet"
)


def main():

    print("=" * 65)
    print("FINSIGHT ML DATA PREPARATION")
    print("=" * 65)

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    print("\nLoading feature dataset...")

    df = pd.read_parquet(INPUT_PATH)

    print(
        f"✓ Dataset loaded: {len(df):,} transactions"
    )

    # --------------------------------------------------
    # SELECT FEATURES
    # --------------------------------------------------

    feature_columns = [

        # Transaction information
        "step",
        "type",
        "amount",

        # Account balances
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",

        # Engineered amount features
        "amount_log",
        "is_high_value",
        "is_zero_amount",

        # Transaction type features
        "is_transfer",
        "is_cash_out",
        "is_payment",
        "is_cash_in",
        "is_debit",

        # Balance behavior
        "origin_balance_change",
        "origin_balance_error",
        "destination_balance_change",
        "destination_balance_error",
        "abs_origin_balance_error",
        "abs_destination_balance_error",

        # Time features
        "hour_of_day",
        "day",

        # Target
        "isFraud"
    ]

    # --------------------------------------------------
    # CREATE ML DATASET
    # --------------------------------------------------

    print("\nSelecting modeling features...")

    ml_df = df[feature_columns].copy()

    print(
        f"✓ Features selected: "
        f"{len(feature_columns) - 1}"
    )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print("\nML DATASET VALIDATION")
    print("-" * 65)

    print(f"Rows: {len(ml_df):,}")

    print(
        f"Features: "
        f"{len(ml_df.columns) - 1}"
    )

    print(
        f"Fraud cases: "
        f"{ml_df['isFraud'].sum():,}"
    )

    print(
        f"Fraud rate: "
        f"{ml_df['isFraud'].mean():.4%}"
    )

    print("\nMissing values:")

    print(
        ml_df.isnull()
        .sum()
        .loc[
            lambda x: x > 0
        ]
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSaving ML dataset...")

    ml_df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print(
        "✓ ML dataset saved successfully!"
    )

    print(f"Location: {OUTPUT_PATH}")

    print("\n" + "=" * 65)
    print("ML DATA PREPARATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()