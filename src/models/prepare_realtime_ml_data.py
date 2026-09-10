from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path(
    "data/features/transaction_features.parquet"
)

OUTPUT_PATH = Path(
    "data/modeling/realtime_ml_dataset.parquet"
)

TARGET = "isFraud"


# ============================================================
# REAL-TIME FEATURES
# ============================================================

REALTIME_FEATURES = [

    # Time
    "step",
    "hour_of_day",
    "day",

    # Transaction information
    "type",
    "amount",
    "amount_log",

    # Pre-transaction balances
    "oldbalanceOrg",
    "oldbalanceDest",

    # Business indicators
    "is_high_value",
    "is_zero_amount",

    # Transaction type indicators
    "is_transfer",
    "is_cash_out",
    "is_payment",
    "is_cash_in",
    "is_debit",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINSIGHT REAL-TIME ML DATA PREPARATION")
print("=" * 70)

print("\nLoading feature dataset...")

df = pd.read_parquet(INPUT_PATH)

print(
    f"✓ Dataset loaded: {len(df):,} transactions"
)


# ============================================================
# SELECT FEATURES
# ============================================================

required_columns = REALTIME_FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


realtime_df = df[
    required_columns
].copy()


print("\nSelected real-time features:")

for i, feature in enumerate(
    REALTIME_FEATURES,
    start=1
):
    print(f"{i:02d}. {feature}")


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("REAL-TIME DATASET VALIDATION")
print("=" * 70)

print(
    f"\nRows: {len(realtime_df):,}"
)

print(
    f"Features: {len(REALTIME_FEATURES)}"
)

print(
    f"Fraud cases: "
    f"{realtime_df[TARGET].sum():,}"
)

print(
    f"Fraud rate: "
    f"{realtime_df[TARGET].mean():.4%}"
)


missing = realtime_df.isnull().sum()

missing = missing[
    missing > 0
]

print("\nMissing values:")

print(missing)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


print("\nSaving real-time ML dataset...")

realtime_df.to_parquet(
    OUTPUT_PATH,
    index=False
)


print(
    "✓ Real-time ML dataset saved successfully!"
)

print(
    f"Location: {OUTPUT_PATH}"
)


print("\n" + "=" * 70)
print("REAL-TIME ML DATA PREPARATION COMPLETED")
print("=" * 70)