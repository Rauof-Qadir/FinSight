from pathlib import Path

import pandas as pd

from src.features.feature_engineering import (
    create_transaction_features
)


def main():

    print("=" * 60)
    print("FINSIGHT FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    input_path = Path(
        "data/processed/transactions_cleaned.parquet"
    )

    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / "transaction_features.parquet"
    )

    print("\nLoading cleaned dataset...")

    df = pd.read_parquet(input_path)

    print(
        f"✓ Dataset loaded: {len(df):,} transactions"
    )

    print("\nCreating transaction features...")

    df_features = create_transaction_features(df)

    print(
        f"✓ Features created successfully"
    )

    print(
        f"✓ Total columns: {len(df_features.columns)}"
    )

    print("\nNew Features:")

    new_features = [
        col
        for col in df_features.columns
        if col not in df.columns
    ]

    for feature in new_features:
        print(f"  ✓ {feature}")

    print("\nSaving feature dataset...")

    df_features.to_parquet(
        output_path,
        index=False
    )

    print(
        f"✓ Feature dataset saved successfully!"
    )

    print(f"Location: {output_path}")

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()