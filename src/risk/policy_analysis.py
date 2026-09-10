import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = "data/modeling/hybrid_predictions.parquet"


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 100)
print("LOADING HYBRID PREDICTIONS")
print("=" * 100)

df = pd.read_parquet(INPUT_PATH)

print(f"Transactions: {len(df):,}")
print(f"Fraud cases: {df['isFraud'].sum():,}")


# ============================================================
# POLICY FUNCTION
# ============================================================

def evaluate_policy(
    name,
    block_condition,
    review_condition,
):
    """
    Evaluate a fraud decision policy.

    BLOCK has priority over REVIEW.
    """

    decisions = pd.Series(
        "ALLOW",
        index=df.index,
    )

    decisions.loc[review_condition] = "REVIEW"
    decisions.loc[block_condition] = "BLOCK"

    y_true = df["isFraud"]

    blocked = (
        decisions == "BLOCK"
    ).astype(int)

    investigated = decisions.isin(
        ["BLOCK", "REVIEW"]
    )

    # --------------------------------------------------------
    # BLOCK METRICS
    # --------------------------------------------------------

    block_tp = (
        (blocked == 1)
        & (y_true == 1)
    ).sum()

    block_fp = (
        (blocked == 1)
        & (y_true == 0)
    ).sum()

    block_fn = (
        (blocked == 0)
        & (y_true == 1)
    ).sum()

    block_precision = precision_score(
        y_true,
        blocked,
        zero_division=0,
    )

    block_recall = recall_score(
        y_true,
        blocked,
        zero_division=0,
    )

    block_f1 = f1_score(
        y_true,
        blocked,
        zero_division=0,
    )

    # --------------------------------------------------------
    # INVESTIGATION METRICS
    # --------------------------------------------------------

    investigation_count = (
        investigated.sum()
    )

    investigation_workload = (
        investigation_count
        / len(df)
        * 100
    )

    fraud_captured = (
        investigated
        & (y_true == 1)
    )

    fraud_captured_count = (
        fraud_captured.sum()
    )

    fraud_capture_rate = (
        fraud_captured_count
        / y_true.sum()
        * 100
    )

    false_positive_reviews = (
        investigated
        & (y_true == 0)
    ).sum()

    investigation_precision = (
        fraud_captured_count
        / investigation_count
        * 100
        if investigation_count > 0
        else 0
    )

    # --------------------------------------------------------
    # FRAUD VALUE
    # --------------------------------------------------------

    total_fraud_value = df.loc[
        y_true == 1,
        "amount"
    ].sum()

    captured_fraud_value = df.loc[
        fraud_captured,
        "amount"
    ].sum()

    fraud_value_capture = (
        captured_fraud_value
        / total_fraud_value
        * 100
        if total_fraud_value > 0
        else 0
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "policy": name,
        "blocked": int(
            (decisions == "BLOCK").sum()
        ),
        "reviewed": int(
            (decisions == "REVIEW").sum()
        ),
        "investigated": int(
            investigation_count
        ),
        "workload_pct": investigation_workload,
        "block_tp": int(block_tp),
        "block_fp": int(block_fp),
        "block_fn": int(block_fn),
        "block_precision_pct": block_precision * 100,
        "block_recall_pct": block_recall * 100,
        "block_f1": block_f1,
        "fraud_captured": int(
            fraud_captured_count
        ),
        "fraud_capture_pct": fraud_capture_rate,
        "false_positive_investigations": int(
            false_positive_reviews
        ),
        "investigation_precision_pct": (
            investigation_precision
        ),
        "fraud_value_captured": (
            captured_fraud_value
        ),
        "fraud_value_capture_pct": (
            fraud_value_capture
        ),
    }


# ============================================================
# COMMON CONDITIONS
# ============================================================

prob = df["fraud_probability"]

high_value_cashout = (
    (df["type"] == "CASH_OUT")
    & (df["is_high_value"] == 1)
)

high_value_transfer = (
    (df["type"] == "TRANSFER")
    & (df["is_high_value"] == 1)
)

zero_amount = (
    df["is_zero_amount"] == 1
)


# ============================================================
# POLICIES
# ============================================================

policies = []


# ------------------------------------------------------------
# POLICY 1 — PURE ML
# ------------------------------------------------------------

policies.append(
    evaluate_policy(
        "Pure ML",
        block_condition=prob >= 0.95,
        review_condition=prob >= 0.50,
    )
)


# ------------------------------------------------------------
# POLICY 2 — CONSERVATIVE REVIEW
# ------------------------------------------------------------

policies.append(
    evaluate_policy(
        "Conservative",
        block_condition=prob >= 0.95,
        review_condition=prob >= 0.70,
    )
)


# ------------------------------------------------------------
# POLICY 3 — HIGH VALUE TRANSFER
# ------------------------------------------------------------

policies.append(
    evaluate_policy(
        "High Value Transfer",
        block_condition=prob >= 0.95,
        review_condition=(
            (prob >= 0.70)
            | high_value_transfer
        ),
    )
)


# ------------------------------------------------------------
# POLICY 4 — CRITICAL CASH OUT
# ------------------------------------------------------------

policies.append(
    evaluate_policy(
        "Critical Cash Out",
        block_condition=(
            (prob >= 0.95)
            | (
                high_value_cashout
                & (prob >= 0.50)
            )
        ),
        review_condition=prob >= 0.50,
    )
)


# ------------------------------------------------------------
# POLICY 5 — FULL HYBRID
# ------------------------------------------------------------

policies.append(
    evaluate_policy(
        "Full Hybrid",
        block_condition=(
            (prob >= 0.95)
            | (
                high_value_cashout
                & (prob >= 0.50)
            )
        ),
        review_condition=(
            (prob >= 0.50)
            | high_value_transfer
            | zero_amount
        ),
    )
)


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    policies
)


print("\n" + "=" * 100)
print("POLICY COMPARISON")
print("=" * 100)

display_columns = [
    "policy",
    "blocked",
    "reviewed",
    "investigated",
    "workload_pct",
    "block_precision_pct",
    "block_recall_pct",
    "block_f1",
    "fraud_captured",
    "fraud_capture_pct",
    "investigation_precision_pct",
    "fraud_value_capture_pct",
]

print(
    results[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_PATH = (
    "reports/policy_comparison.csv"
)

results.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n" + "=" * 100)
print("RESULTS SAVED")
print("=" * 100)

print(
    f"Output: {OUTPUT_PATH}"
)