import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import joblib


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = Path("data/modeling/realtime_train.parquet")
TEST_PATH = Path("data/modeling/realtime_test.parquet")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "realtime_logistic_regression.joblib"

PREDICTIONS_PATH = Path(
    "data/modeling/realtime_logistic_predictions.parquet"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINSIGHT REAL-TIME LOGISTIC REGRESSION")
print("=" * 70)

print("\nLoading training data...")

train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

print(f"✓ Train shape: {train_df.shape}")
print(f"✓ Test shape:  {test_df.shape}")


# ============================================================
# SPLIT FEATURES AND TARGET
# ============================================================

TARGET = "isFraud"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = [
    "type"
]

numerical_features = [
    "step",
    "hour_of_day",
    "day",
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
    "is_debit"
]


# ============================================================
# PREPROCESSING
# ============================================================

print("\nBuilding preprocessing pipeline...")

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="saga",
    random_state=42
)


# ============================================================
# FULL PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Real-Time Logistic Regression...")
print("This may take some time because of the dataset size...")

pipeline.fit(X_train, y_train)

print("✓ Training complete!")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

fraud_probabilities = pipeline.predict_proba(X_test)[:, 1]

predictions = (fraud_probabilities >= 0.50).astype(int)


# ============================================================
# MODEL METRICS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

roc_auc = roc_auc_score(
    y_test,
    fraud_probabilities
)

pr_auc = average_precision_score(
    y_test,
    fraud_probabilities
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


print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")

print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        digits=4
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
    0.95
]

for threshold in thresholds:

    threshold_predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    flagged = threshold_predictions.sum()

    workload = (
        flagged / len(y_test)
    ) * 100

    precision_t = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    recall_t = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    f1_t = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    print(f"\nThreshold: {threshold:.2f}")

    print(f"Flagged:   {flagged:,}")
    print(f"Workload:  {workload:.2f}%")

    print(f"Precision: {precision_t:.4f}")
    print(f"Recall:    {recall_t:.4f}")
    print(f"F1:        {f1_t:.4f}")


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving trained model...")

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(f"✓ Model saved: {MODEL_PATH}")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

print("\nSaving predictions...")

results = pd.DataFrame({
    "isFraud": y_test.values,
    "fraud_probability": fraud_probabilities,
    "prediction": predictions
})

results.to_parquet(
    PREDICTIONS_PATH,
    index=False
)

print(
    f"✓ Predictions saved: {PREDICTIONS_PATH}"
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("REAL-TIME LOGISTIC REGRESSION COMPLETED")
print("=" * 70)