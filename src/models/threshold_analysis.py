from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


# ============================================================
# CONFIG
# ============================================================

PREDICTIONS_PATH = Path(
    "data/modeling/logistic_regression_predictions.parquet"
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("Loading model predictions...")

df = pd.read_parquet(PREDICTIONS_PATH)

print(f"Dataset shape: {df.shape}")

y_true = df["isFraud"].astype(int).values
y_prob = df["fraud_probability"].values


# ============================================================
# BASIC INFORMATION
# ============================================================

total_transactions = len(df)
total_fraud = y_true.sum()

print("\n" + "=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Total transactions: {total_transactions:,}")
print(f"Total fraud cases:  {total_fraud:,}")
print(f"Fraud rate:         {total_fraud / total_transactions:.4%}")


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = np.arange(
    0.05,
    1.00,
    0.05
)

results = []


for threshold in thresholds:

    predicted = (
        y_prob >= threshold
    ).astype(int)

    tp = np.sum(
        (predicted == 1) &
        (y_true == 1)
    )

    fp = np.sum(
        (predicted == 1) &
        (y_true == 0)
    )

    fn = np.sum(
        (predicted == 0) &
        (y_true == 1)
    )

    tn = np.sum(
        (predicted == 0) &
        (y_true == 0)
    )

    flagged = tp + fp

    precision = (
        tp / flagged
        if flagged > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    workload = (
        flagged / total_transactions
    )

    # Amount associated with detected fraud
    fraud_amount_captured = df.loc[
        (predicted == 1) &
        (y_true == 1),
        "amount"
    ].sum()

    # Total fraud amount
    total_fraud_amount = df.loc[
        y_true == 1,
        "amount"
    ].sum()

    amount_recall = (
        fraud_amount_captured /
        total_fraud_amount
        if total_fraud_amount > 0
        else 0
    )

    results.append(
        {
            "threshold": threshold,
            "flagged_transactions": flagged,
            "workload": workload,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fraud_amount_captured": fraud_amount_captured,
            "fraud_amount_recall": amount_recall,
        }
    )


results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD BUSINESS ANALYSIS")
print("=" * 70)

display_columns = [
    "threshold",
    "flagged_transactions",
    "workload",
    "precision",
    "recall",
    "f1",
    "fraud_amount_captured",
    "fraud_amount_recall",
]

display_df = results_df[
    display_columns
].copy()


display_df["workload"] = (
    display_df["workload"] * 100
)

display_df["precision"] = (
    display_df["precision"] * 100
)

display_df["recall"] = (
    display_df["recall"] * 100
)

display_df["fraud_amount_recall"] = (
    display_df["fraud_amount_recall"] * 100
)


print(
    display_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "workload": "{:.2f}%".format,
            "precision": "{:.2f}%".format,
            "recall": "{:.2f}%".format,
            "f1": "{:.4f}".format,
            "fraud_amount_captured": "{:,.0f}".format,
            "fraud_amount_recall": "{:.2f}%".format,
        }
    )
)


# ============================================================
# BEST F1
# ============================================================

best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"Threshold: {best_f1['threshold']:.2f}"
)

print(
    f"Precision: {best_f1['precision']:.4%}"
)

print(
    f"Recall:    {best_f1['recall']:.4%}"
)

print(
    f"F1:        {best_f1['f1']:.4f}"
)

print(
    f"Workload:  {best_f1['workload']:.2%}"
)


# ============================================================
# BEST PRECISION UNDER WORKLOAD LIMITS
# ============================================================

print("\n" + "=" * 70)
print("BEST PRECISION UNDER WORKLOAD LIMITS")
print("=" * 70)


workload_limits = [
    0.001,   # 0.1%
    0.005,   # 0.5%
    0.01,    # 1%
    0.02,    # 2%
    0.05,    # 5%
]


for limit in workload_limits:

    eligible = results_df[
        results_df["workload"] <= limit
    ]

    if eligible.empty:
        print(
            f"\nWorkload <= {limit:.1%}: "
            "No threshold available"
        )
        continue

    best = eligible.loc[
        eligible["precision"].idxmax()
    ]

    print(
        f"\nWorkload <= {limit:.1%}"
    )

    print(
        f"Threshold: {best['threshold']:.2f}"
    )

    print(
        f"Actual workload: {best['workload']:.2%}"
    )

    print(
        f"Precision: {best['precision']:.2%}"
    )

    print(
        f"Recall: {best['recall']:.2%}"
    )

    print(
        f"F1: {best['f1']:.4f}"
    )

    print(
        f"Fraud amount captured: "
        f"{best['fraud_amount_captured']:,.0f}"
    )

    print(
        f"Fraud amount recall: "
        f"{best['fraud_amount_recall']:.2%}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_PATH = Path(
    "reports/logistic_threshold_analysis.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nResults saved to: {OUTPUT_PATH}"
)