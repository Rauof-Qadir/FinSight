from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = Path("data/modeling/train.parquet")
TEST_PATH = Path("data/modeling/test.parquet")

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")


# ============================================================
# FEATURES / TARGET
# ============================================================

TARGET = "isFraud"

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "amount_log",
    "is_high_value",
    "is_zero_amount",
    "is_transfer",
    "is_cash_out",
    "is_payment",
    "is_cash_in",
    "is_debit",
    "origin_balance_change",
    "origin_balance_error",
    "destination_balance_change",
    "destination_balance_error",
    "abs_origin_balance_error",
    "abs_destination_balance_error",
    "hour_of_day",
    "day",
]


X_train = train_df[FEATURES]
y_train = train_df[TARGET].astype(int)

X_test = test_df[FEATURES]
y_test = test_df[TARGET].astype(int)


print("\nTarget distribution:")
print(y_train.value_counts())
print("\nFraud rate:")
print(y_train.mean())


# ============================================================
# FEATURE TYPES
# ============================================================

CATEGORICAL_FEATURES = [
    "type"
]

NUMERICAL_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_FEATURES
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            NUMERICAL_FEATURES,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
            CATEGORICAL_FEATURES,
        ),
    ]
)


print("\nFitting preprocessing...")

X_train_processed = preprocessor.fit_transform(X_train)

print("Transforming test data...")

X_test_processed = preprocessor.transform(X_test)

print(f"Processed train shape: {X_train_processed.shape}")
print(f"Processed test shape:  {X_test_processed.shape}")


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    class_weight="balanced",
    solver="saga",
    max_iter=100,
    random_state=RANDOM_STATE,
)

model.fit(
    X_train_processed,
    y_train,
)

print("Training complete.")


# ============================================================
# PREDICT PROBABILITIES
# ============================================================

print("\nGenerating fraud probabilities...")

y_prob = model.predict_proba(X_test_processed)[:, 1]


# Default threshold
THRESHOLD = 0.50

y_pred = (y_prob >= THRESHOLD).astype(int)


# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob,
)

print("\nROC-AUC:")
print(f"{roc_auc:.4f}")


# ============================================================
# PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    y_prob,
)

print("\nPR-AUC:")
print(f"{pr_auc:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)


thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


for threshold in thresholds:

    predictions = (
        y_prob >= threshold
    ).astype(int)

    tp = np.sum(
        (predictions == 1) &
        (y_test.values == 1)
    )

    fp = np.sum(
        (predictions == 1) &
        (y_test.values == 0)
    )

    fn = np.sum(
        (predictions == 0) &
        (y_test.values == 1)
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
        if (precision + recall) > 0
        else 0
    )

    workload = (
        flagged / len(y_test)
    )

    print(
        f"\nThreshold: {threshold:.2f}"
        f"\nFlagged:   {flagged:,}"
        f"\nWorkload:  {workload:.2%}"
        f"\nTP:        {tp:,}"
        f"\nFP:        {fp:,}"
        f"\nFN:        {fn:,}"
        f"\nPrecision: {precision:.4%}"
        f"\nRecall:    {recall:.4%}"
        f"\nF1:        {f1:.4f}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print("\nSaving predictions...")

results = test_df[
    [
        "step",
        "type",
        "amount",
        "isFraud",
    ]
].copy()

results["fraud_probability"] = y_prob

results["predicted_fraud"] = y_pred

results["model"] = "logistic_regression"


OUTPUT_PATH = Path(
    "data/modeling/logistic_regression_predictions.parquet"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_parquet(
    OUTPUT_PATH,
    index=False,
)


print(
    f"\nPredictions saved to: {OUTPUT_PATH}"
)

print("\nLogistic Regression baseline completed successfully.")