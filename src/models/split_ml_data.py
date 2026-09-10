from pathlib import Path

import pandas as pd

# ============================================================

# PATHS

# ============================================================

INPUT_PATH = Path("data/modeling/ml_dataset.parquet")
TRAIN_PATH = Path("data/modeling/train.parquet")
TEST_PATH = Path("data/modeling/test.parquet")

# ============================================================

# CONFIGURATION

# ============================================================

TRAIN_RATIO = 0.80


def print_dataset_summary(df, dataset_name):
    total_transactions = len(df)
    fraud_cases = df["isFraud"].sum()
    fraud_rate = df["isFraud"].mean() * 100
    legitimate_cases = total_transactions - fraud_cases

    print(f"\n{dataset_name.upper()} DATASET")
    print("-" * 65)

    print(f"Transactions:      {total_transactions:,}")
    print(f"Legitimate cases:  {legitimate_cases:,}")
    print(f"Fraud cases:       {fraud_cases:,}")
    print(f"Fraud rate:        {fraud_rate:.4f}%")

    print(
        f"Step range:        "
        f"{df['step'].min()} → {df['step'].max()}"
    )


def main():
    print("=" * 65)
    print("FINSIGHT ML TRAIN / TEST SPLIT")
    print("=" * 65)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading ML dataset...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"✓ Dataset loaded: {len(df):,} transactions")

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    print("\nSorting transactions chronologically...")

    df = df.sort_values(by="step").reset_index(drop=True)

    print("✓ Transactions sorted by step")

    # ========================================================
    # CREATE SPLIT
    # ========================================================

    split_index = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "=" * 65)
    print("SPLIT VALIDATION")
    print("=" * 65)

    print_dataset_summary(train_df, "Training")
    print_dataset_summary(test_df, "Testing")

    # ========================================================
    # COMPARISON
    # ========================================================

    print("\nFRAUD DISTRIBUTION COMPARISON")
    print("-" * 65)

    comparison = pd.DataFrame(
        {
            "dataset": ["Full Dataset", "Training", "Testing"],
            "transactions": [len(df), len(train_df), len(test_df)],
            "fraud_cases": [
                df["isFraud"].sum(),
                train_df["isFraud"].sum(),
                test_df["isFraud"].sum(),
            ],
            "fraud_rate_percent": [
                df["isFraud"].mean() * 100,
                train_df["isFraud"].mean() * 100,
                test_df["isFraud"].mean() * 100,
            ],
        }
    )

    print(
        comparison.to_string(
            index=False,
            formatters={"fraud_rate_percent": "{:.4f}%".format},
        )
    )

    # ========================================================
    # SAVE DATASETS
    # ========================================================

    print("\nSaving datasets...")

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(TRAIN_PATH, index=False)
    test_df.to_parquet(TEST_PATH, index=False)

    print("✓ Training dataset saved")
    print(f"  Location: {TRAIN_PATH}")

    print("✓ Testing dataset saved")
    print(f"  Location: {TEST_PATH}")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("TRAIN / TEST SPLIT COMPLETED")
    print("=" * 65)

    print(f"\nTraining transactions: {len(train_df):,}")
    print(f"Testing transactions:  {len(test_df):,}")


if __name__ == "__main__":
    main()
