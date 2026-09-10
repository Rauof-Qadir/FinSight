from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = Path(
    "data/modeling/xgb_train_full.parquet"
)

VALIDATION_PATH = Path(
    "data/modeling/xgb_validation.parquet"
)

TEST_PATH = Path(
    "data/modeling/realtime_test.parquet"
)

MODEL_PATH = Path(
    "models/xgboost_final.joblib"
)

PREDICTION_PATH = Path(
    "data/modeling/xgboost_final_predictions.parquet"
)

TARGET = "isFraud"

RANDOM_STATE = 42


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "step",
    "type",
    "amount",
    "amount_log",
    "oldbalanceOrg",
    "oldbalanceDest",
    "is_high_value",
    "is_zero_amount",
    "is_transfer",
    "is_cash_out",
    "is_payment",
    "is_cash_in",
    "is_debit",
    "hour_of_day",
    "day",
]


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("FINSIGHT FINAL XGBOOST TRAINING")
print("=" * 70)

print("\nLoading datasets...")

train_df = pd.read_parquet(TRAIN_PATH)

validation_df = pd.read_parquet(
    VALIDATION_PATH
)

test_df = pd.read_parquet(
    TEST_PATH
)

print(
    f"✓ Train:      {len(train_df):,}"
)

print(
    f"✓ Validation: {len(validation_df):,}"
)

print(
    f"✓ Test:       {len(test_df):,}"
)


# ============================================================
# ENCODING
# ============================================================

print("\nEncoding transaction type...")

combined = pd.concat(
    [
        train_df[FEATURES],
        validation_df[FEATURES],
        test_df[FEATURES],
    ],
    ignore_index=True,
)

combined = pd.get_dummies(
    combined,
    columns=["type"],
    dtype=int,
)

train_end = len(train_df)

validation_end = (
    train_end + len(validation_df)
)

X_train = combined.iloc[
    :train_end
].copy()

X_validation = combined.iloc[
    train_end:validation_end
].copy()

X_test = combined.iloc[
    validation_end:
].copy()


# ============================================================
# TARGET
# ============================================================

y_train = train_df[TARGET]

y_validation = validation_df[TARGET]

y_test = test_df[TARGET]


print(
    f"\nX_train:      {X_train.shape}"
)

print(
    f"X_validation: {X_validation.shape}"
)

print(
    f"X_test:       {X_test.shape}"
)

print(
    f"\nTrain fraud:      {y_train.sum():,}"
)

print(
    f"Validation fraud: {y_validation.sum():,}"
)

print(
    f"Test fraud:       {y_test.sum():,}"
)


# ============================================================
# CLASS WEIGHT
# ============================================================

negative = (
    y_train == 0
).sum()

positive = (
    y_train == 1
).sum()

scale_pos_weight = (
    negative / positive
)

print(
    f"\nscale_pos_weight: "
    f"{scale_pos_weight:.2f}"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="aucpr",

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    tree_method="hist",
)


model.fit(
    X_train,
    y_train,
)


print("\n✓ Training completed!")


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

validation_probabilities = (
    model.predict_proba(
        X_validation
    )[:, 1]
)


# ============================================================
# VALIDATION AUC
# ============================================================

validation_roc_auc = roc_auc_score(
    y_validation,
    validation_probabilities,
)

validation_pr_auc = (
    average_precision_score(
        y_validation,
        validation_probabilities,
    )
)


print("\n" + "=" * 70)
print("VALIDATION PERFORMANCE")
print("=" * 70)

print(
    f"\nROC-AUC: {validation_roc_auc:.4f}"
)

print(
    f"PR-AUC:  {validation_pr_auc:.4f}"
)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

print("\nOptimizing threshold on validation set...")

threshold_results = []

thresholds = np.arange(
    0.01,
    1.00,
    0.01,
)


for threshold in thresholds:

    predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    flagged = predictions.sum()

    workload = (
        flagged / len(y_validation)
    )

    threshold_results.append(
        {
            "threshold": threshold,
            "flagged": flagged,
            "workload": workload,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# BEST VALIDATION F1
# ============================================================

best = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(
    best["threshold"]
)


print("\n" + "=" * 70)
print("BEST VALIDATION THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Flagged:   "
    f"{int(best['flagged']):,}"
)

print(
    f"Workload:  "
    f"{best['workload']:.2%}"
)

print(
    f"Precision: "
    f"{best['precision']:.2%}"
)

print(
    f"Recall:    "
    f"{best['recall']:.2%}"
)

print(
    f"F1:        "
    f"{best['f1']:.4f}"
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL UNSEEN TEST")
print("=" * 70)

print(
    f"\nUsing validation-selected "
    f"threshold: {best_threshold:.2f}"
)

test_probabilities = (
    model.predict_proba(
        X_test
    )[:, 1]
)

test_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


# ============================================================
# FINAL METRICS
# ============================================================

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

test_pr_auc = (
    average_precision_score(
        y_test,
        test_probabilities,
    )
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)


print(
    f"\nROC-AUC:   {test_roc_auc:.4f}"
)

print(
    f"PR-AUC:    {test_pr_auc:.4f}"
)

print(
    f"Precision: {test_precision:.4f}"
)

print(
    f"Recall:    {test_recall:.4f}"
)

print(
    f"F1:        {test_f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# BUSINESS METRICS
# ============================================================

flagged = test_predictions.sum()

workload = (
    flagged / len(test_df)
)

captured_fraud_amount = test_df.loc[
    (test_predictions == 1)
    & (y_test == 1),
    "amount",
].sum()

total_fraud_amount = test_df.loc[
    y_test == 1,
    "amount",
].sum()

fraud_amount_recall = (
    captured_fraud_amount
    / total_fraud_amount
)


print("\n" + "=" * 70)
print("BUSINESS IMPACT")
print("=" * 70)

print(
    f"\nFlagged transactions: "
    f"{flagged:,}"
)

print(
    f"Investigation workload: "
    f"{workload:.2%}"
)

print(
    f"Fraud cases captured: "
    f"{((test_predictions == 1) & (y_test == 1)).sum():,}"
)

print(
    f"Fraud amount captured: "
    f"{captured_fraud_amount:,.2f}"
)

print(
    f"Fraud amount recall: "
    f"{fraud_amount_recall:.2%}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = test_df[
    [
        "step",
        "type",
        "amount",
        TARGET,
    ]
].copy()

prediction_df[
    "fraud_probability"
] = test_probabilities

prediction_df[
    "prediction"
] = test_predictions

prediction_df[
    "decision_threshold"
] = best_threshold


PREDICTION_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

prediction_df.to_parquet(
    PREDICTION_PATH,
    index=False,
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    {
        "model": model,
        "features": list(X_train.columns),
        "threshold": best_threshold,
    },
    MODEL_PATH,
)


print(
    "\n✓ Predictions saved to:"
)

print(PREDICTION_PATH)

print(
    "\n✓ Final model saved to:"
)

print(MODEL_PATH)


print("\n" + "=" * 70)
print("✓ FINAL XGBOOST PIPELINE COMPLETED")
print("=" * 70)