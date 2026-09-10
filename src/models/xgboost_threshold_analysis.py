from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIG
# ============================================================

PREDICTION_PATH = Path(
    "data/modeling/xgboost_predictions.parquet"
)

print("=" * 70)
print("FINSIGHT XGBOOST THRESHOLD ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(
    PREDICTION_PATH
)

y_true = df["isFraud"]

probabilities = df[
    "fraud_probability"
]


total_transactions = len(df)

total_fraud = y_true.sum()

total_fraud_amount = df.loc[
    y_true == 1,
    "amount"
].sum()


print(
    f"\nTest transactions: {total_transactions:,}"
)

print(
    f"Test fraud cases:  {total_fraud:,}"
)

print(
    f"Total fraud amount: {total_fraud_amount:,.2f}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []


thresholds = np.arange(
    0.01,
    1.00,
    0.01
)


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    flagged = predictions.sum()

    workload = (
        flagged / total_transactions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    captured_fraud_amount = df.loc[
        predictions == 1,
        "amount"
    ].where(
        y_true == 1
    ).sum()

    fraud_amount_recall = (
        captured_fraud_amount
        / total_fraud_amount
        if total_fraud_amount > 0
        else 0
    )

    results.append(
        {
            "threshold": threshold,
            "flagged_transactions": flagged,
            "workload_percent": workload * 100,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "captured_fraud_amount": captured_fraud_amount,
            "fraud_amount_recall": fraud_amount_recall,
        }
    )


results_df = pd.DataFrame(
    results
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
    f"\nThreshold:             "
    f"{best_f1['threshold']:.2f}"
)

print(
    f"Flagged transactions:  "
    f"{best_f1['flagged_transactions']:,}"
)

print(
    f"Workload:              "
    f"{best_f1['workload_percent']:.2f}%"
)

print(
    f"Precision:             "
    f"{best_f1['precision']:.4f}"
)

print(
    f"Recall:                "
    f"{best_f1['recall']:.4f}"
)

print(
    f"F1:                    "
    f"{best_f1['f1']:.4f}"
)

print(
    f"Fraud amount captured: "
    f"{best_f1['captured_fraud_amount']:,.2f}"
)

print(
    f"Fraud amount recall:   "
    f"{best_f1['fraud_amount_recall']:.2%}"
)


# ============================================================
# SELECTED BUSINESS THRESHOLDS
# ============================================================

selected_thresholds = [
    0.50,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99,
]


print("\n" + "=" * 70)
print("BUSINESS THRESHOLD COMPARISON")
print("=" * 70)

display_columns = [
    "threshold",
    "flagged_transactions",
    "workload_percent",
    "precision",
    "recall",
    "f1",
    "fraud_amount_recall",
]


print(
    results_df[
        results_df["threshold"].isin(
            selected_thresholds
        )
    ][display_columns].to_string(
        index=False
    )
)


# ============================================================
# LOW-WORKLOAD STRATEGIES
# ============================================================

print("\n" + "=" * 70)
print("LOW-WORKLOAD STRATEGIES")
print("=" * 70)


for max_workload in [
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
]:

    eligible = results_df[
        results_df["workload_percent"]
        <= max_workload
    ]

    if len(eligible) == 0:
        continue

    best = eligible.loc[
        eligible["f1"].idxmax()
    ]

    print(
        f"\nWorkload <= {max_workload:.1f}%"
    )

    print(
        f"Threshold:  {best['threshold']:.2f}"
    )

    print(
        f"Flagged:    "
        f"{best['flagged_transactions']:,}"
    )

    print(
        f"Precision:  "
        f"{best['precision']:.2%}"
    )

    print(
        f"Recall:     "
        f"{best['recall']:.2%}"
    )

    print(
        f"F1:         "
        f"{best['f1']:.4f}"
    )


print("\n" + "=" * 70)
print("✓ Threshold analysis completed")
print("=" * 70)