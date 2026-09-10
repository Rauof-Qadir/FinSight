import pandas as pd
from pathlib import Path


DATA_PATH = Path(
    "data/features/transactions_with_risk.parquet"
)


def calculate_metrics(y_true, predictions):

    tp = ((predictions == 1) & (y_true == 1)).sum()

    fp = ((predictions == 1) & (y_true == 0)).sum()

    fn = ((predictions == 0) & (y_true == 1)).sum()

    tn = ((predictions == 0) & (y_true == 0)).sum()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def main():

    print("=" * 75)
    print("FINSIGHT FRAUD ALERT STRATEGY COMPARISON")
    print("=" * 75)

    print("\nLoading dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(
        f"✓ Dataset loaded: {len(df):,} transactions"
    )

    y_true = df["isFraud"]

    # =========================================================
    # DEFINE ALERT STRATEGIES
    # =========================================================

    strategies = {

        # Conservative strategy
        "Conservative (Score >= 80)": (
            df["risk_score"] >= 80
        ),

        # High-confidence strategy
        "High Confidence (Score >= 60)": (
            df["risk_score"] >= 60
        ),

        # Balanced strategy
        "Balanced (Score >= 25)": (
            df["risk_score"] >= 25
        ),

        # Broad strategy
        "Broad (Risky Type)": (
            df["type"].isin(
                ["TRANSFER", "CASH_OUT"]
            )
        ),

        # Precision-focused business rule
        "Critical CASH_OUT Pattern": (
            (df["type"] == "CASH_OUT")
            & (df["risk_score"] >= 80)
        ),

    }

    # =========================================================
    # EVALUATE STRATEGIES
    # =========================================================

    results = []

    total_transactions = len(df)

    for name, prediction in strategies.items():

        prediction = prediction.astype(int)

        metrics = calculate_metrics(
            y_true,
            prediction
        )

        flagged = prediction.sum()

        workload_pct = (
            flagged / total_transactions
        )

        fraud_amount_captured = df.loc[
            (prediction == 1)
            & (df["isFraud"] == 1),
            "amount"
        ].sum()

        results.append({

            "strategy": name,

            "flagged_transactions": flagged,

            "workload_pct": workload_pct,

            "true_positives": metrics["TP"],

            "false_positives": metrics["FP"],

            "false_negatives": metrics["FN"],

            "precision": metrics["precision"],

            "recall": metrics["recall"],

            "f1_score": metrics["f1_score"],

            "fraud_amount_captured": (
                fraud_amount_captured
            ),

        })

    results_df = pd.DataFrame(results)

    # Sort by F1 score
    results_df = results_df.sort_values(
        "f1_score",
        ascending=False
    )

    print("\n")

    print(
        results_df.to_string(
            index=False,
            formatters={

                "workload_pct":
                    "{:.3%}".format,

                "precision":
                    "{:.4%}".format,

                "recall":
                    "{:.4%}".format,

                "f1_score":
                    "{:.4f}".format,

                "fraud_amount_captured":
                    "{:,.2f}".format,

            }
        )
    )

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    output_path = Path(
        "reports/risk_strategy_comparison.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 75)

    print(
        "✓ Strategy comparison saved successfully!"
    )

    print(f"Location: {output_path}")

    print("=" * 75)


if __name__ == "__main__":
    main()