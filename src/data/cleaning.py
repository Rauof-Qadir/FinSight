# src/data/cleaning.py

import pandas as pd


EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud"
]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that all expected columns exist."""

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("✓ Schema validation passed")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    print(f"✓ Duplicate rows removed: {removed:,}")

    return df


def clean_transaction_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize transaction type values."""

    df["type"] = (
        df["type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    print("✓ Transaction types standardized")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values based on business rules."""

    missing_before = df.isnull().sum().sum()

    print(f"Missing values before cleaning: {missing_before:,}")

    # Critical transaction fields
    critical_columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest"
    ]

    df = df.dropna(subset=critical_columns)

    # Financial balances
    balance_columns = [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    df[balance_columns] = df[balance_columns].fillna(0)

    missing_after = df.isnull().sum().sum()

    print(f"✓ Missing values remaining: {missing_after:,}")

    return df


def remove_invalid_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove transactions violating basic business rules."""

    before = len(df)

    # Amount cannot be negative
    df = df[df["amount"] >= 0]

    # Step cannot be negative
    df = df[df["step"] >= 0]

    # Fraud columns must be binary
    df = df[df["isFraud"].isin([0, 1])]
    df = df[df["isFlaggedFraud"].isin([0, 1])]

    removed = before - len(df)

    print(f"✓ Invalid transactions removed: {removed:,}")

    return df


def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage using appropriate data types."""

    df["step"] = df["step"].astype("int32")

    df["amount"] = df["amount"].astype("float32")

    balance_columns = [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    for column in balance_columns:
        df[column] = df[column].astype("float32")

    df["type"] = df["type"].astype("category")

    df["isFraud"] = df["isFraud"].astype("int8")
    df["isFlaggedFraud"] = df["isFlaggedFraud"].astype("int8")

    print("✓ Data types optimized")

    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete transaction cleaning pipeline."""

    print("\n" + "=" * 60)
    print("STARTING DATA CLEANING PIPELINE")
    print("=" * 60)

    validate_schema(df)

    df = remove_duplicates(df)

    df = clean_transaction_types(df)

    df = handle_missing_values(df)

    df = remove_invalid_transactions(df)

    df = optimize_data_types(df)

    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETED")
    print("=" * 60)

    print(f"\nFinal rows: {len(df):,}")

    return df