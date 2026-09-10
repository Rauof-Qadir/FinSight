from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(
    "data/modeling/realtime_train.parquet"
)

TRAIN_PATH = Path(
    "data/modeling/xgb_train_full.parquet"
)

VALIDATION_PATH = Path(
    "data/modeling/xgb_validation.parquet"
)

TRAIN_RATIO = 0.80


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("FINSIGHT XGBOOST TRAIN / VALIDATION SPLIT")
print("=" * 70)

print("\nLoading training data...")

df = pd.read_parquet(INPUT_PATH)

print(
    f"✓ Rows: {len(df):,}"
)

print(
    f"✓ Fraud: {df['isFraud'].sum():,}"
)


# ============================================================
# CHRONOLOGICAL SORT
# ============================================================

print("\nSorting chronologically by step...")

df = df.sort_values(
    "step"
).reset_index(drop=True)


# ============================================================
# SPLIT
# ============================================================

split_index = int(
    len(df) * TRAIN_RATIO
)

train_df = df.iloc[
    :split_index
].copy()

validation_df = df.iloc[
    split_index:
].copy()


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def print_summary(name, data):

    fraud = data["isFraud"].sum()

    print(f"\n{name}")

    print("-" * 50)

    print(
        f"Rows:       {len(data):,}"
    )

    print(
        f"Fraud:      {fraud:,}"
    )

    print(
        f"Legitimate: {len(data) - fraud:,}"
    )

    print(
        f"Fraud rate: {fraud / len(data):.4%}"
    )

    print(
        f"Step range: {data['step'].min()} → "
        f"{data['step'].max()}"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print_summary(
    "XGBoost Training Set",
    train_df
)

print_summary(
    "XGBoost Validation Set",
    validation_df
)


# ============================================================
# SAVE
# ============================================================

TRAIN_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

train_df.to_parquet(
    TRAIN_PATH,
    index=False
)

validation_df.to_parquet(
    VALIDATION_PATH,
    index=False
)


print("\n" + "=" * 70)

print(
    "✓ Training set saved:"
)

print(
    TRAIN_PATH
)

print(
    "\n✓ Validation set saved:"
)

print(
    VALIDATION_PATH
)

print("=" * 70)