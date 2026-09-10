from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(
    "data/modeling/realtime_train.parquet"
)

OUTPUT_PATH = Path(
    "data/modeling/xgb_train.parquet"
)

TARGET = "isFraud"

# Number of legitimate transactions to sample
LEGIT_SAMPLE_SIZE = 500_000

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINSIGHT XGBOOST DATA PREPARATION")
print("=" * 70)

print("\nLoading training data...")

df = pd.read_parquet(INPUT_PATH)

print(
    f"✓ Training transactions: {len(df):,}"
)

print(
    f"✓ Fraud cases: {df[TARGET].sum():,}"
)


# ============================================================
# SEPARATE FRAUD / LEGITIMATE
# ============================================================

fraud_df = df[
    df[TARGET] == 1
].copy()

legitimate_df = df[
    df[TARGET] == 0
].copy()


print("\nOriginal distribution:")

print(
    f"Fraud:       {len(fraud_df):,}"
)

print(
    f"Legitimate:  {len(legitimate_df):,}"
)


# ============================================================
# SAMPLE LEGITIMATE TRANSACTIONS
# ============================================================

print(
    f"\nSampling {LEGIT_SAMPLE_SIZE:,} legitimate transactions..."
)

legitimate_sample = legitimate_df.sample(
    n=LEGIT_SAMPLE_SIZE,
    random_state=RANDOM_STATE
)


# ============================================================
# COMBINE
# ============================================================

xgb_df = pd.concat(
    [
        fraud_df,
        legitimate_sample
    ],
    ignore_index=True
)


# ============================================================
# SHUFFLE
# ============================================================

xgb_df = xgb_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST TRAINING DATA")
print("=" * 70)

total = len(xgb_df)

fraud = xgb_df[TARGET].sum()

print(
    f"\nTransactions: {total:,}"
)

print(
    f"Fraud cases:  {fraud:,}"
)

print(
    f"Legitimate:   {total - fraud:,}"
)

print(
    f"Fraud rate:   {fraud / total:.4%}"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print(
    f"\nSaving dataset to:"
    f"\n{OUTPUT_PATH}"
)

xgb_df.to_parquet(
    OUTPUT_PATH,
    index=False
)


print("\n✓ XGBoost dataset created successfully!")

print("=" * 70)