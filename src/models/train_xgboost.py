from pathlib import Path

import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = Path(
    "data/modeling/xgb_train.parquet"
)

TEST_PATH = Path(
    "data/modeling/realtime_test.parquet"
)

MODEL_PATH = Path(
    "models/xgboost_fraud_model.joblib"
)

PREDICTION_PATH = Path(
    "data/modeling/xgboost_predictions.parquet"
)

TARGET = "isFraud"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINSIGHT XGBOOST FRAUD DETECTION")
print("=" * 70)

print("\nLoading training data...")

train_df = pd.read_parquet(TRAIN_PATH)

print(
    f"✓ Training rows: {len(train_df):,}"
)

print("\nLoading test data...")

test_df = pd.read_parquet(TEST_PATH)

print(
    f"✓ Test rows: {len(test_df):,}"
)


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
# ENCODE TYPE
# ============================================================

print("\nEncoding transaction type...")

combined = pd.concat(
    [
        train_df[FEATURES],
        test_df[FEATURES]
    ],
    axis=0,
    ignore_index=True
)

combined = pd.get_dummies(
    combined,
    columns=["type"],
    dtype=int
)

train_encoded = combined.iloc[
    :len(train_df)
].copy()

test_encoded = combined.iloc[
    len(train_df):
].copy()


# ============================================================
# TARGET
# ============================================================

y_train = train_df[TARGET]

y_test = test_df[TARGET]

X_train = train_encoded

X_test = test_encoded


print(
    f"\nX_train shape: {X_train.shape}"
)

print(
    f"X_test shape:  {X_test.shape}"
)

print(
    f"Training fraud: {y_train.sum():,}"
)

print(
    f"Test fraud:     {y_test.sum():,}"
)


# ============================================================
# SCALE POSITIVE CLASS
# ============================================================

negative = (y_train == 0).sum()

positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(
    f"\nscale_pos_weight: {scale_pos_weight:.2f}"
)


# ============================================================
# MODEL
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

    tree_method="hist"
)


model.fit(
    X_train,
    y_train
)


print("\n✓ Model training completed!")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


print("\n" + "=" * 70)
print("XGBOOST RESULTS")
print("=" * 70)

print(
    f"\nROC-AUC:   {roc_auc:.4f}"
)

print(
    f"PR-AUC:    {pr_auc:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = test_df[
    [
        "step",
        "type",
        "amount",
        TARGET
    ]
].copy()

prediction_df["fraud_probability"] = probabilities

prediction_df["prediction"] = predictions


PREDICTION_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

prediction_df.to_parquet(
    PREDICTION_PATH,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "features": list(X_train.columns),
    },
    MODEL_PATH
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False
)


print("\n" + "=" * 70)
print("TOP 15 FEATURES")
print("=" * 70)

print(
    importance.head(15).to_string(
        index=False
    )
)


print("\n✓ Predictions saved to:")
print(PREDICTION_PATH)

print("\n✓ Model saved to:")
print(MODEL_PATH)

print("\n" + "=" * 70)