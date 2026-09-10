from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/modeling/ml_dataset.parquet")

TARGET = "isFraud"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINSIGHT FEATURE AUDIT")
print("=" * 70)

print("\nLoading ML dataset...")

df = pd.read_parquet(DATA_PATH)

print(f"✓ Dataset loaded: {len(df):,} transactions")
print(f"✓ Total columns: {len(df.columns)}")


# ============================================================
# DATASET STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE LIST")
print("=" * 70)

features = [
    column
    for column in df.columns
    if column != TARGET
]

for i, feature in enumerate(features, start=1):
    print(f"{i:02d}. {feature}")


# ============================================================
# CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL FEATURE CORRELATION WITH FRAUD")
print("=" * 70)


numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()


if TARGET in numeric_columns:
    numeric_columns.remove(TARGET)


correlations = (
    df[numeric_columns + [TARGET]]
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .abs()
    .sort_values(ascending=False)
)


print("\nAbsolute correlation with isFraud:\n")

for feature, correlation in correlations.items():
    print(
        f"{feature:<35} {correlation:.6f}"
    )


# ============================================================
# FRAUD VS NON-FRAUD FEATURE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FRAUD VS LEGITIMATE FEATURE COMPARISON")
print("=" * 70)


comparison_rows = []


for feature in numeric_columns:

    legitimate = df.loc[
        df[TARGET] == 0,
        feature
    ]

    fraud = df.loc[
        df[TARGET] == 1,
        feature
    ]

    comparison_rows.append(
        {
            "feature": feature,
            "legitimate_mean": legitimate.mean(),
            "fraud_mean": fraud.mean(),
            "difference": fraud.mean() - legitimate.mean(),
            "fraud_median": fraud.median(),
            "legitimate_median": legitimate.median(),
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


print(
    comparison_df
    .sort_values(
        "difference",
        key=lambda x: x.abs(),
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# ZERO / UNIQUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE DISTRIBUTION AUDIT")
print("=" * 70)


audit_rows = []


for feature in features:

    total = len(df)

    missing = df[feature].isna().sum()

    unique = df[feature].nunique()

    zero_count = 0

    if pd.api.types.is_numeric_dtype(
        df[feature]
    ):
        zero_count = (
            df[feature] == 0
        ).sum()

    audit_rows.append(
        {
            "feature": feature,
            "dtype": str(df[feature].dtype),
            "missing": missing,
            "unique_values": unique,
            "zero_values": zero_count,
            "zero_percent": (
                zero_count / total
            ) * 100,
        }
    )


audit_df = pd.DataFrame(
    audit_rows
)


print(
    audit_df
    .sort_values(
        "unique_values",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# SAVE REPORTS
# ============================================================

REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


correlations.reset_index().rename(
    columns={
        "index": "feature",
        TARGET: "absolute_correlation",
    }
).to_csv(
    REPORT_DIR / "feature_correlations.csv",
    index=False
)


comparison_df.to_csv(
    REPORT_DIR / "feature_fraud_comparison.csv",
    index=False
)


audit_df.to_csv(
    REPORT_DIR / "feature_audit.csv",
    index=False
)


print("\n" + "=" * 70)
print("FEATURE AUDIT COMPLETED")
print("=" * 70)

print("\nReports saved:")

print("✓ reports/feature_correlations.csv")

print("✓ reports/feature_fraud_comparison.csv")

print("✓ reports/feature_audit.csv")