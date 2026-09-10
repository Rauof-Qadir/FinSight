import pandas as pd

from src.risk.hybrid_engine import (
    predict_transactions,
)


# ============================================================
# LOAD REAL TEST TRANSACTIONS
# ============================================================

TEST_PATH = (
    "data/modeling/realtime_test.parquet"
)

df = pd.read_parquet(
    TEST_PATH
)


# Take a small sample first
sample = df.sample(
    n=20,
    random_state=42,
).reset_index(drop=True)


# ============================================================
# RUN HYBRID ENGINE
# ============================================================

result = predict_transactions(
    sample
)


# ============================================================
# DISPLAY
# ============================================================

columns = [
    "step",
    "type",
    "amount",
    "is_high_value",
    "is_zero_amount",
    "isFraud",
    "fraud_probability",
    "decision",
    "risk_level",
    "decision_reason",
]


print("\n" + "=" * 100)

print(
    "FINSIGHT HYBRID RISK ENGINE"
)

print("=" * 100)

print(
    result[columns].to_string(
        index=False
    )
)


# ============================================================
# DECISION DISTRIBUTION
# ============================================================

print("\n" + "=" * 100)

print(
    "DECISION DISTRIBUTION"
)

print("=" * 100)

print(
    result["decision"]
    .value_counts()
)


print("\n" + "=" * 100)

print(
    "RISK DISTRIBUTION"
)

print("=" * 100)

print(
    result["risk_level"]
    .value_counts()
)