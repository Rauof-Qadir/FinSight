# src/data/profiling.py

import pandas as pd


def profile_data(df: pd.DataFrame) -> None:
    """Generate a basic data quality profile."""

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nRows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nCOLUMN DATA TYPES")
    print("-" * 60)
    print(df.dtypes)

    print("\nMISSING VALUES")
    print("-" * 60)

    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    missing_report = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_percent
    })

    print(missing_report[missing_report["missing_count"] > 0])

    print("\nDUPLICATE ROWS")
    print("-" * 60)
    print(df.duplicated().sum())

    print("\nNUMERICAL SUMMARY")
    print("-" * 60)
    print(df.describe())

    print("\nUNIQUE VALUES")
    print("-" * 60)

    for column in df.columns:
        print(f"{column}: {df[column].nunique():,}")