from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
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
    "data/modeling/xgb_train_full.parquet"
)

VALIDATION_PATH = Path(
    "data/modeling/xgb_validation.parquet"
)

TEST_PATH = Path(
    "data/modeling/realtime_test.parquet"
)

RANDOM_STATE = 42


# ============================================================
# FEATURE GROUPS
# ============================================================

FULL_FEATURES = [
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


NO_TYPE_FEATURES = [
    "step",
    "amount",
    "amount_log",
    "oldbalanceOrg",
    "oldbalanceDest",
    "is_high_value",
    "is_zero_amount",
    "hour_of_day",
    "day",
]


BEHAVIORAL_FEATURES = [
    "amount",
    "amount_log",
    "oldbalanceOrg",
    "oldbalanceDest",
    "is_high_value",
    "is_zero_amount",
    "step",
    "hour_of_day",
    "day",
]


EXPERIMENTS = {
    "Full Model": FULL_FEATURES,
    "No Transaction Type": NO_TYPE_FEATURES,
    "Behavioral Only": BEHAVIORAL_FEATURES,
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINSIGHT XGBOOST ABLATION STUDY")
print("=" * 70)

print("\nLoading datasets...")

train_df = pd.read_parquet(
    TRAIN_PATH
)

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
# TARGETS
# ============================================================

y_train = train_df["isFraud"]

y_validation = validation_df["isFraud"]

y_test = test_df["isFraud"]


# ============================================================
# EXPERIMENT FUNCTION
# ============================================================

def run_experiment(
    experiment_name,
    features,
):

    print("\n" + "=" * 70)

    print(
        f"EXPERIMENT: {experiment_name}"
    )

    print("=" * 70)

    print("\nFeatures:")

    for feature in features:
        print(
            f"  - {feature}"
        )


    # --------------------------------------------------------
    # ENCODE CATEGORICAL FEATURES
    # --------------------------------------------------------

    if "type" in features:

        combined = pd.concat(
            [
                train_df[features],
                validation_df[features],
                test_df[features],
            ],
            ignore_index=True,
        )

        combined = pd.get_dummies(
            combined,
            columns=["type"],
            dtype=int,
        )

    else:

        combined = pd.concat(
            [
                train_df[features],
                validation_df[features],
                test_df[features],
            ],
            ignore_index=True,
        )


    # --------------------------------------------------------
    # SPLIT FEATURES
    # --------------------------------------------------------

    train_end = len(train_df)

    validation_end = (
        train_end
        + len(validation_df)
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


    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    negative = (
        y_train == 0
    ).sum()

    positive = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative / positive
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "\nTraining XGBoost..."
    )

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


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )


    validation_pr_auc = (
        average_precision_score(
            y_validation,
            validation_probabilities,
        )
    )


    validation_roc_auc = (
        roc_auc_score(
            y_validation,
            validation_probabilities,
        )
    )


    # --------------------------------------------------------
    # FIND BEST VALIDATION THRESHOLD
    # --------------------------------------------------------

    best_threshold = 0.50

    best_f1 = 0

    for threshold in np.arange(
        0.01,
        1.00,
        0.01,
    ):

        predictions = (
            validation_probabilities
            >= threshold
        ).astype(int)

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        if f1 > best_f1:

            best_f1 = f1

            best_threshold = float(
                threshold
            )


    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)


    test_roc_auc = (
        roc_auc_score(
            y_test,
            test_probabilities,
        )
    )

    test_pr_auc = (
        average_precision_score(
            y_test,
            test_probabilities,
        )
    )

    test_precision = (
        precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    test_recall = (
        recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    test_f1 = (
        f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )


    flagged = (
        test_predictions.sum()
    )

    workload = (
        flagged / len(y_test)
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {
        "experiment": experiment_name,
        "features": len(features),
        "validation_roc_auc": validation_roc_auc,
        "validation_pr_auc": validation_pr_auc,
        "validation_best_threshold": best_threshold,
        "validation_best_f1": best_f1,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_flagged": flagged,
        "test_workload": workload,
    }


    print(
        "\nValidation:"
    )

    print(
        f"ROC-AUC: "
        f"{validation_roc_auc:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{validation_pr_auc:.4f}"
    )

    print(
        f"Best threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best F1: "
        f"{best_f1:.4f}"
    )


    print(
        "\nFinal Test:"
    )

    print(
        f"ROC-AUC: "
        f"{test_roc_auc:.4f}"
    )

    print(
        f"PR-AUC: "
        f"{test_pr_auc:.4f}"
    )

    print(
        f"Precision: "
        f"{test_precision:.4f}"
    )

    print(
        f"Recall: "
        f"{test_recall:.4f}"
    )

    print(
        f"F1: "
        f"{test_f1:.4f}"
    )

    print(
        f"Flagged: "
        f"{flagged:,}"
    )

    print(
        f"Workload: "
        f"{workload:.2%}"
    )


    return result


# ============================================================
# RUN EXPERIMENTS
# ============================================================

results = []


for name, features in EXPERIMENTS.items():

    result = run_experiment(
        name,
        features,
    )

    results.append(
        result
    )


# ============================================================
# COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("ABLATION STUDY RESULTS")
print("=" * 70)


display_columns = [
    "experiment",
    "features",
    "validation_best_threshold",
    "validation_best_f1",
    "test_roc_auc",
    "test_pr_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_flagged",
    "test_workload",
]


print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_PATH = Path(
    "reports/xgboost_ablation_results.csv"
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
    f"\n✓ Results saved to:"
)

print(
    OUTPUT_PATH
)

print("\n" + "=" * 70)
print("✓ ABLATION STUDY COMPLETED")
print("=" * 70)