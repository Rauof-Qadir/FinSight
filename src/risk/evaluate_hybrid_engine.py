import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.risk.hybrid_engine import predict_transactions


# ============================================================
# CONFIG
# ============================================================

TEST_PATH = "data/modeling/realtime_test.parquet"
OUTPUT_PATH = "data/modeling/hybrid_predictions.parquet"


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 100)
print("LOADING FINAL TEST DATA")
print("=" * 100)

df = pd.read_parquet(TEST_PATH)

print(f"Transactions: {len(df):,}")
print(f"Fraud cases: {df['isFraud'].sum():,}")


# ============================================================
# RUN HYBRID ENGINE
# ============================================================

print("\n" + "=" * 100)
print("RUNNING HYBRID RISK ENGINE")
print("=" * 100)

result = predict_transactions(df)


# ============================================================
# DECISION DISTRIBUTION
# ============================================================

print("\n" + "=" * 100)
print("DECISION DISTRIBUTION")
print("=" * 100)

decision_counts = (
    result["decision"]
    .value_counts()
)

print(decision_counts)

print("\nDecision percentages:")

print(
    (
        result["decision"]
        .value_counts(normalize=True)
        * 100
    ).round(4)
)


# ============================================================
# CONFUSION MATRIX FOR BLOCK
# ============================================================

result["blocked"] = (
    result["decision"] == "BLOCK"
).astype(int)

y_true = result["isFraud"]

y_pred = result["blocked"]


tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
).ravel()


# ============================================================
# BLOCK PERFORMANCE
# ============================================================

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)


print("\n" + "=" * 100)
print("BLOCK PERFORMANCE")
print("=" * 100)

print(f"True Negatives : {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives : {tp:,}")

print(f"\nPrecision: {precision:.4%}")
print(f"Recall   : {recall:.4%}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# INVESTIGATION WORKLOAD
# ============================================================

result["investigation"] = (
    result["decision"]
    .isin(["REVIEW", "BLOCK"])
)

investigation_count = (
    result["investigation"].sum()
)

investigation_workload = (
    investigation_count
    / len(result)
    * 100
)

fraud_in_investigation = (
    result.loc[
        result["investigation"],
        "isFraud"
    ].sum()
)

fraud_capture = (
    fraud_in_investigation
    / y_true.sum()
    * 100
)


print("\n" + "=" * 100)
print("INVESTIGATION WORKLOAD")
print("=" * 100)

print(
    f"Transactions investigated: "
    f"{investigation_count:,}"
)

print(
    f"Investigation workload: "
    f"{investigation_workload:.4f}%"
)

print(
    f"Fraud cases captured: "
    f"{fraud_in_investigation:,}"
)

print(
    f"Fraud capture rate: "
    f"{fraud_capture:.4f}%"
)


# ============================================================
# FRAUD MISSED
# ============================================================

fraud_missed = (
    y_true.sum()
    - fraud_in_investigation
)

miss_rate = (
    fraud_missed
    / y_true.sum()
    * 100
)


print("\n" + "=" * 100)
print("FRAUD MISSED")
print("=" * 100)

print(
    f"Fraud missed: {fraud_missed:,}"
)

print(
    f"Miss rate: {miss_rate:.4f}%"
)


# ============================================================
# TRANSACTION VALUE
# ============================================================

total_value = result["amount"].sum()

investigated_value = (
    result.loc[
        result["investigation"],
        "amount"
    ].sum()
)

fraud_value = (
    result.loc[
        result["isFraud"] == 1,
        "amount"
    ].sum()
)

captured_fraud_value = (
    result.loc[
        result["investigation"]
        & (result["isFraud"] == 1),
        "amount"
    ].sum()
)

fraud_value_capture = (
    captured_fraud_value
    / fraud_value
    * 100
)


print("\n" + "=" * 100)
print("TRANSACTION VALUE PROTECTION")
print("=" * 100)

print(
    f"Total transaction value: "
    f"{total_value:,.2f}"
)

print(
    f"Investigated transaction value: "
    f"{investigated_value:,.2f}"
)

print(
    f"Total fraud value: "
    f"{fraud_value:,.2f}"
)

print(
    f"Captured fraud value: "
    f"{captured_fraud_value:,.2f}"
)

print(
    f"Fraud value capture: "
    f"{fraud_value_capture:.4f}%"
)


# ============================================================
# DECISION × FRAUD
# ============================================================

print("\n" + "=" * 100)
print("DECISION × FRAUD")
print("=" * 100)

decision_fraud = pd.crosstab(
    result["decision"],
    result["isFraud"],
)

print(decision_fraud)


# ============================================================
# RISK LEVEL PERFORMANCE
# ============================================================

print("\n" + "=" * 100)
print("RISK LEVEL PERFORMANCE")
print("=" * 100)

risk_summary = (
    result
    .groupby("risk_level")
    .agg(
        transactions=("isFraud", "size"),
        fraud_cases=("isFraud", "sum"),
        total_amount=("amount", "sum"),
        average_amount=("amount", "mean"),
    )
)

risk_summary["fraud_rate"] = (
    risk_summary["fraud_cases"]
    / risk_summary["transactions"]
    * 100
)

print(
    risk_summary.to_string()
)


# ============================================================
# SAVE RESULTS
# ============================================================

result.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("\n" + "=" * 100)
print("RESULTS SAVED")
print("=" * 100)

print(
    f"Output: {OUTPUT_PATH}"
)

print("\nHybrid evaluation complete.")