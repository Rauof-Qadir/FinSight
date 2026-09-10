from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(
    "data/modeling/realtime_ml_dataset.parquet"
)

TRAIN_PATH = Path(
    "data/modeling/realtime_train.parquet"
)

TEST_PATH = Path(
    "data/modeling/realtime_test.parquet"
)

TARGET = "isFraud"

TRAIN_RATIO = 0.80


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINSIGHT REAL-TIME ML TRAIN / TEST SPLIT")
print("=" * 70)

print("\nLoading real-time ML dataset...")

df = pd.read_parquet(INPUT_PATH)

print(f"✓ Dataset loaded: {len(df):,} transactions")


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

print("\nSorting transactions chronologically...")

df = df.sort_values(
    "step"
).reset_index(drop=True)

print("✓ Transactions sorted by step")


# ============================================================
# SPLIT DATA
# ============================================================

split_index = int(
    len(df) * TRAIN_RATIO
)

train_df = df.iloc[:split_index].copy()

test_df = df.iloc[split_index:].copy()


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def dataset_summary(data, name):

    total = len(data)

    fraud_cases = data[TARGET].sum()

    legitimate_cases = (
        total - fraud_cases
    )

    fraud_rate = (
        fraud_cases / total
    ) * 100

    print(f"\n{name.upper()} DATASET")

    print("-" * 65)

    print(f"Transactions:      {total:,}")

    print(f"Legitimate cases:  {legitimate_cases:,}")

    print(f"Fraud cases:       {fraud_cases:,}")

    print(f"Fraud rate:        {fraud_rate:.4f}%")

    print(
        f"Step range:        "
        f"{data['step'].min()} → "
        f"{data['step'].max()}"
    )


# ============================================================
# PRINT VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SPLIT VALIDATION")
print("=" * 70)

dataset_summary(
    train_df,
    "Training"
)

dataset_summary(
    test_df,
    "Testing"
)


# ============================================================
# FRAUD DISTRIBUTION COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FRAUD DISTRIBUTION COMPARISON")
print("=" * 70)


comparison = pd.DataFrame(
    {
        "dataset": [
            "Full Dataset",
            "Training",
            "Testing",
        ],
        "transactions": [
            len(df),
            len(train_df),
            len(test_df),
        ],
        "fraud_cases": [
            df[TARGET].sum(),
            train_df[TARGET].sum(),
            test_df[TARGET].sum(),
        ],
        "fraud_rate_percent": [
            df[TARGET].mean() * 100,
            train_df[TARGET].mean() * 100,
            test_df[TARGET].mean() * 100,
        ],
    }
)

print(
    comparison.to_string(
        index=False,
        formatters={
            "fraud_rate_percent":
                "{:.4f}%".format
        }
    )
)


# ============================================================
# SAVE DATASETS
# ============================================================

print("\nSaving datasets...")

TRAIN_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

train_df.to_parquet(
    TRAIN_PATH,
    index=False
)

test_df.to_parquet(
    TEST_PATH,
    index=False
)

print("✓ Training dataset saved")
print(f"  Location: {TRAIN_PATH}")

print("✓ Testing dataset saved")
print(f"  Location: {TEST_PATH}")


print("\n" + "=" * 70)
print("REAL-TIME TRAIN / TEST SPLIT COMPLETED")
print("=" * 70)

print(f"\nTraining transactions: {len(train_df):,}")
print(f"Testing transactions:  {len(test_df):,}")